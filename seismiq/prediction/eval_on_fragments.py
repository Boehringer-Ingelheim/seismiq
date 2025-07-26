import dataclasses
import os
import pickle
import sys
from collections.abc import Iterable

import click
import pandas as pd
from loguru import logger

from seismiq.prediction.data import test_datasets
from seismiq.prediction.llm.data_module import EncoderDecoderLlmDataModule
from seismiq.prediction.llm.model_loading import get_available_models, load_checkpoint
from seismiq.prediction.llm.sampling import (
    postprocess_generated_samples,
    sample_predictions_beam_search,
)
from seismiq.prediction.llm.training_module import EncoderDecoderLlmTrainingModule
from seismiq.utils import rdkit_wrapper as wrdkit
from seismiq.utils.mol_utils import fragment_mol, get_mol_formula


@click.group
def main() -> None:
    pass


@main.command()
@click.argument(
    "result-folder",
    type=click.Path(file_okay=False, writable=True),
    default="./development/test_results_fragments/",
)
@click.option("--slurm-flags", type=str, default="")
@click.option("--skip-existing/--no-skip-existing", default=True)
def make_slurm_command(result_folder: str, slurm_flags: str, skip_existing: bool) -> None:
    models = get_available_models()
    challenges = ["casmi_2016", "casmi_2017", "casmi_2022"]
    challenges += [f"massspecgym.{i}.10" for i in range(10)]

    os.makedirs(result_folder, exist_ok=True)
    for modl in models:
        for cha in challenges:
            out_file = os.path.join(result_folder, f"{modl}-{cha}.pkl")
            py_cmd = f'python {sys.argv[0]} run-single "{modl}" "{cha}" "{out_file}"'
            cmd = f"srun {slurm_flags}" if slurm_flags else ""

            if not skip_existing or not os.path.exists(out_file):
                print("nohup", cmd, py_cmd, "&")


@main.command()
@click.argument("model-name", type=str)
@click.argument("challenge-name", type=str)
@click.argument("result-file", type=click.Path(dir_okay=False, writable=True))
def run_single(model_name: str, challenge_name: str, result_file: str) -> None:
    logger.info(f"will save results to {result_file}")

    challenge_name, *challenge_parts = challenge_name.split(".")
    challenges = list(getattr(test_datasets, challenge_name)())
    logger.info(f"loaded {len(challenges)} challenges for test dataset {challenge_name}")

    if challenge_parts:
        # the name of the challenge is of the form "name.i.p" where
        # i is the part number and p is the total number of parts
        i, p = map(int, challenge_parts)
        n = len(challenges) // p
        challenges = challenges[i * n :] if i == p - 1 else challenges[i * n : (i + 1) * n]

        logger.info(f"will evaluate part {i + 1} of {p} ({len(challenges)} challenges)")

    logger.info(f"loading model {model_name}")
    model, data = load_checkpoint(model_name)
    res = eval_model(model, data, challenges).assign(model=model_name)

    with open(result_file, "wb") as f:
        pickle.dump(res, f)


def eval_model(
    model: EncoderDecoderLlmTrainingModule,
    data: EncoderDecoderLlmDataModule,
    challenges: Iterable[test_datasets.TestChallenge],
) -> pd.DataFrame:
    challenges = list(challenges)

    all_preds = []
    for i, ch in enumerate(challenges):
        logger.info(f"Doing challenge {ch.challenge} ({i + 1} / {len(challenges)}) of dataset {ch.dataset}")
        try:
            model_preds = eval_model_on_challenge(model, data, ch.spectrum, ch.smiles, ch.adduct_shift)
        except KeyError:
            logger.error(f'could not encode SMILES "{ch.smiles}", skipping')
        else:
            if model_preds is not None:
                all_preds.append(model_preds.assign(challenge=ch.challenge))

    res = pd.concat(all_preds).assign(dataset=ch.dataset)
    return res


def eval_model_on_challenge(
    model: EncoderDecoderLlmTrainingModule,
    data: EncoderDecoderLlmDataModule,
    spec: list[tuple[float, float]],
    smiles: str,
    adduct_shift: float,
    num_predictions: int = 128,
    peak_mz_noise: float = 1e-2,
) -> pd.DataFrame:
    data.val_dset.peak_mz_noise = peak_mz_noise
    # assert isinstance(model, DecoderOnlyTrainingModule) and isinstance(data, DecoderOnlyDataModule)

    rows = []
    frags = list(fragment_mol(smiles))
    real_mol = wrdkit.mol_from_smiles(smiles)
    formula = get_mol_formula(smiles)

    for i, (bond_idx, smiles1, fragment1, smiles2, fragment2) in enumerate(frags):
        logger.debug(f"doing fragments {2 * i} and {2 * i + 1} of {2 * len(frags)}")
        for given_smiles, given_fragment, missing_smiles, missing_fragment in [
            (smiles1, fragment1, smiles2, fragment2),
            (smiles2, fragment2, smiles1, fragment1),
        ]:
            if len(missing_fragment.GetAtoms()) <= 2:
                # skip fragments with a single atom (plus the dummy)
                continue

            dummy = list(given_fragment.GetAtoms())[-1]
            assert dummy.GetSymbol() == "*"

            generated = postprocess_generated_samples(
                sample_predictions_beam_search(
                    model,
                    data,
                    spec,
                    formula,
                    adduct_shift=adduct_shift,
                    smiles_prefix=given_smiles,
                    max_sampling_steps=20 + len(missing_smiles),
                    num_beams=num_predictions,
                ),
                match_hydrogen_count=False,
                real_mol=real_mol,
            )

            best_tani = 0.0
            for gen in generated:
                best_tani = max(best_tani, gen.tanimoto)

                d = dataclasses.asdict(gen)
                res = {k: d[k] for k in ["perplexity", "tanimoto", "pred_smiles", "generation_count"]}

                # used to uniquely identify the fragment
                res["bond_idx"] = bond_idx
                res["dummy_idx"] = dummy.GetIdx()
                res["smiles_prompt"] = given_smiles
                res["missing_smiles"] = missing_smiles

                # used for performance analysis
                res["given_atoms"] = len(given_fragment.GetAtoms()) - 1
                res["missing_atoms"] = len(missing_fragment.GetAtoms()) - 1

                rows.append(res)

            logger.debug(f"best Tanimoto: {best_tani}")

    df = pd.DataFrame(rows)
    return df.reset_index()


if __name__ == "__main__":
    main()
