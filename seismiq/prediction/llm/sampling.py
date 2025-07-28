import dataclasses
import time
from collections.abc import Iterator
from copy import deepcopy
from typing import cast

import numpy as np
import torch
from loguru import logger
from rdkit import Chem, DataStructs

from seismiq.prediction.llm.data_module import EncoderDecoderLlmDataModule, ModelSample
from seismiq.prediction.llm.training_module import EncoderDecoderLlmTrainingModule
from seismiq.utils import rdkit_wrapper as wrdkit
from seismiq.utils.mol_utils import get_mol_formula


@dataclasses.dataclass
class GeneratedSample:
    wrapped_sample: ModelSample
    logp: float
    prompt_logp: float
    perplexity: float
    prompt_perplexity: float
    tok_probs: list[float]
    tok_unc: list[float]
    pred_smiles: str | None
    mol: wrdkit.Mol | None
    fingerprints: DataStructs.cDataStructs.ExplicitBitVect | None
    tanimoto: float  # wrt real molecule
    generation_count: int


@dataclasses.dataclass
class SamplingStatus:
    sampling_step: int
    max_sampling_steps: int
    num_finished_samples: int
    num_ongoing_samples: int


def sample_predictions_beam_search(
    model: EncoderDecoderLlmTrainingModule,
    data: EncoderDecoderLlmDataModule,
    spec: list[tuple[float, float]],
    mol_atoms: dict[int, int],
    max_sampling_steps: int = 500,
    num_beams: int = 3,
    smiles_prefix: str | None = None,
    adduct_shift: float = 1.007825032,
    skip_wrong_atom_count: bool = True,
    return_partial_samples: bool = False,
) -> list[GeneratedSample]:
    progress = sample_predictions_beam_search_with_progress(
        model,
        data,
        spec,
        mol_atoms,
        max_sampling_steps,
        num_beams,
        smiles_prefix,
        adduct_shift,
        skip_wrong_atom_count,
        return_partial_samples,
    )
    for status in progress:
        pass

    assert isinstance(status, list)
    return status


def sample_predictions_beam_search_with_progress(
    model: EncoderDecoderLlmTrainingModule,
    data: EncoderDecoderLlmDataModule,
    spec: list[tuple[float, float]],
    mol_atoms: dict[int, int],
    max_sampling_steps: int = 500,
    num_beams: int = 128,
    smiles_prefix: str | None = None,
    adduct_shift: float = 1.007825032,
    skip_wrong_atom_count: bool = True,
    return_partial_samples: bool = False,
) -> Iterator[SamplingStatus | list[GeneratedSample]]:
    """Beam search sampling from the model, progress bar support.

    Args:
        model (DecoderOnlyTrainingModule): The model to sample from.
        data (DecoderOnlyDataModule): Data module to be used for encoding inputs and decoding outputs.
        spec (list[tuple[float, float]]): The mass spectrum, as list of (mass, intensity) tuples.
        mol_atoms (dict[int, int]): The number of atoms for each atomic number.
        max_sampling_steps (int, optional): Maximum number of tokens to sample. Defaults to 500.
        num_beams (int, optional): Number of beams to use. Defaults to 3.
        smiles_prefix (Optional[str], optional): Initial prompt for the model. Defaults to None.
        skip_wrong_atom_count (bool, optional): Wether to discard predicted molecules with incorrect sum formula.
            Defaults to False.
        return_partial_samples (bool, optional): Whether to return SMILES that do not parse correctly.
            Defaults to False.

    Yields:
        Iterator[dict[str, Any], list[GeneratedSample[ModelSample]]]: Each sampling step yields
            a dictionary with information on the sampling process. The very last yielded item contains
            the final model predictions.
    """

    tokenizer = data.dencoder._smiles_tokenizer  # type: ignore
    stop_token = tokenizer.token_to_id[tokenizer.TOKEN_END]

    attachment_token = data.val_dset.dencoder._smiles_tokenizer.token_to_id.get("[*]")
    spacer_token = data.val_dset.dencoder._smiles_tokenizer.token_to_id.get(".")
    if smiles_prefix is not None:
        prefix_tokens = data.val_dset.dencoder.encode(smiles_prefix)
        initial_tokens = torch.tensor(prefix_tokens, dtype=torch.long)[:-1]
        smiles_has_no_fragments = attachment_token not in prefix_tokens
    else:
        smiles_prefix = ""
        initial_tokens = torch.ones(1, dtype=torch.long)
        smiles_has_no_fragments = True

    assert smiles_prefix is not None and mol_atoms is not None

    pt = Chem.GetPeriodicTable()  # type: ignore
    monoiso_mass = sum(n * pt.GetMostCommonIsotopeMass(a) for a, n in mol_atoms.items())
    peaks = data.val_dset.transform_peaks(spec, adduct_shift, monoiso_mass)

    all_samples = [
        GeneratedSample(
            wrapped_sample=ModelSample(
                idx=-1,
                smiles=smiles_prefix,
                tokens=initial_tokens,
                peaks=peaks,
                remaining_atoms=data.val_dset.compute_remaining_atoms(mol_atoms, initial_tokens, validate=False),
                atoms=mol_atoms,
                weight=1.0,
                is_experimental=True,
            ),
            logp=0.0,
            prompt_logp=0.0,
            perplexity=0.0,
            prompt_perplexity=0.0,
            tok_probs=[],
            tok_unc=[],
            pred_smiles="",
            mol=None,
            fingerprints=None,
            tanimoto=0.0,
            generation_count=1,
        )
        for _ in range(num_beams)
    ]

    finished_samples: list[GeneratedSample] = []

    tgpu = tcpu = 0
    t0 = t1 = time.monotonic_ns()
    for sampling_step in range(max_sampling_steps):
        batch = data.val_dset.collate([s.wrapped_sample for s in all_samples]).to(model.device)

        t1 = time.monotonic_ns()
        tcpu += t1 - t0

        with torch.no_grad():
            t0 = t1
            preds = model(batch)
            t1 = time.monotonic_ns()
            tgpu += t1 - t0
            t0 = t1

        next_token_logits = preds.tokens[:, -1, :]
        next_token_probs = next_token_logits.softmax(-1)
        next_token_logp = torch.log(1e-28 + next_token_probs)  # (num_beams, num_tokens)
        num_tokens = next_token_logp.shape[1]

        if sampling_step == 0:
            # compute tok_probs and logp of initial tokens, including start token
            for i, s in enumerate(all_samples):
                real_tokens = batch.tokens[i].long()
                probs = preds.tokens[i].softmax(-1)
                tok_probs = probs[torch.arange(probs.shape[0]), real_tokens]
                s.tok_probs = tok_probs.tolist()
                s.prompt_logp = s.logp = torch.log(1e-28 + tok_probs).sum().item()
                s.prompt_perplexity = np.exp(-s.logp / len(s.tok_probs))

        logps = torch.tensor([s.logp for s in all_samples], device=next_token_logp.device)  # (num_beams,)
        next_seq_logps, next_idx = torch.sort(
            (next_token_logp + logps.unsqueeze(-1)).view(-1),  # (num_beams * num_tokes,)
            descending=True,
        )

        i = 0
        sample_next: list[GeneratedSample] = []
        new_seqs = set()
        while len(sample_next) < num_beams and i < next_idx.shape[0]:
            sid, token = int(next_idx[i] // num_tokens), int(next_idx[i] % num_tokens)
            new_seq_logp = next_seq_logps[i]
            i += 1

            if smiles_has_no_fragments and token in (attachment_token, spacer_token):
                continue

            # make sure each beam contains an unique sequence
            new_seq = tuple(all_samples[sid].wrapped_sample.tokens.tolist() + [token])
            if new_seq in new_seqs:
                continue
            new_seqs.add(new_seq)

            # update this sample with the new token
            # we deepcopy because this sample could be part of several beams in the next iteration
            sample = deepcopy(all_samples[sid])
            sample.wrapped_sample.tokens = torch.tensor(new_seq)

            # update remaining atoms
            new_remaining = sample.wrapped_sample.remaining_atoms[-1].clone()
            at = tokenizer.token_to_atom(token)
            if at is not None:
                new_remaining[at[1]] -= 1
                if skip_wrong_atom_count and new_remaining[at[1]] < 0:
                    # skip this SMILES as the molecule would have too many atoms of this element
                    continue
            sample.wrapped_sample.remaining_atoms = torch.cat(
                [sample.wrapped_sample.remaining_atoms, new_remaining.unsqueeze(0)]
            )

            # compute token probability and uncertainty
            sample.logp = new_seq_logp.item()
            sample.tok_probs.append(next_token_probs[sid][token].item())
            ps = next_token_probs[sid]
            sample.tok_unc.append(-torch.sum(ps * torch.log(ps + 1e-28)).item())

            if token == stop_token:
                if not skip_wrong_atom_count or sample.wrapped_sample.remaining_atoms[-1, 2:].max() == 0:
                    finished_samples.append(sample)
            else:
                sample_next.append(sample)

        yield SamplingStatus(sampling_step, max_sampling_steps, len(finished_samples), len(sample_next))

        all_samples = sample_next
        if not all_samples:
            break

    t1 = time.monotonic_ns()
    tcpu += t1 - t0
    logger.debug(
        f"sampled for {sampling_step} steps with {num_beams} beams"
        f" - beaming time: {tcpu / 1e9:.2f}s, model time: {tgpu / 1e9:.2f}s"
    )

    ret = (finished_samples + all_samples) if return_partial_samples else finished_samples
    for s in ret:
        s.pred_smiles = cast(str, data.dencoder.decode(s.wrapped_sample.tokens.int().tolist(), return_mol=False))
        s.perplexity = np.exp(-s.logp / len(s.tok_probs))

    yield ret


def postprocess_generated_samples(
    generated_samples: list[GeneratedSample],
    match_hydrogen_count: bool,
    real_mol: wrdkit.Mol | None = None,
) -> list[GeneratedSample]:
    valid_samples: dict[str, GeneratedSample] = {}
    for s in generated_samples:
        if s.pred_smiles is None:
            continue

        s.mol = wrdkit.mol_from_smiles(s.pred_smiles, sanitize=False)
        if s.mol is None:
            continue

        # deduplicate keeping lowest perplexity sample for each mol
        canon = wrdkit.mol_to_smiles(s.mol)
        if canon not in valid_samples:
            valid_samples[canon] = s
        elif s.perplexity < valid_samples[canon].perplexity:
            s.generation_count += valid_samples[canon].generation_count
            valid_samples[canon] = s
        else:
            valid_samples[canon].generation_count += 1

    real_fp = wrdkit.fingerprint_mol(real_mol) if real_mol is not None else None
    res: list[GeneratedSample] = []
    for s in valid_samples.values():
        if match_hydrogen_count:
            pred_hs = get_mol_formula(mol=s.mol).get(1, 0)
            real_hs = int(s.wrapped_sample.remaining_atoms[0, 1])
            if pred_hs != real_hs:
                continue

        s.tanimoto = float("nan")
        if s.mol is not None:
            s.fingerprints = wrdkit.fingerprint_mol(s.mol)
            if real_fp is not None:
                s.tanimoto = wrdkit.tanimoto_similarity(s.fingerprints, real_fp)  # type: ignore
        res.append(s)

    return res
