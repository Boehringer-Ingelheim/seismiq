import dataclasses
import json
from typing import Any, TextIO

import click
import pandas as pd
from loguru import logger

from seismiq.prediction.llm.data_module import EncoderDecoderLlmDataModule
from seismiq.prediction.llm.model_loading import load_checkpoint
from seismiq.prediction.llm.sampling import postprocess_generated_samples, sample_predictions_beam_search
from seismiq.prediction.llm.training_module import EncoderDecoderLlmTrainingModule
from seismiq.utils import rdkit_wrapper as wrdkit
from seismiq.utils.mol_utils import parse_mol_formula


@click.command
@click.argument("model-name", type=str)
@click.argument("input-file", type=click.File("r"))
@click.argument("result-file", type=click.File("w"))
@click.option("--num-beams", type=int, default=128, help="Number of beams for beam search")
@click.option("--max-sampling-steps", type=int, help="Maximum number of sampling steps")
@click.option("--peak-mz-noise", type=float, default=1e-2, help="Noise level for peak m/z values")
@click.option(
    "--skip-wrong-atom-count/--keep-wrong-atom-count", default=True, help="Skip samples with wrong heavy atom count"
)
@click.option(
    "--keep-partial-samples/--skip-partial-samples", default=False, help="Keep samples that were not fully generated"
)
@click.option(
    "--match-hydrogen-count/--no-match-hydrogen-count",
    default=False,
    help="Match number of hydrogen atoms in generated samples",
)
def main(model_name: str, input_file: TextIO, result_file: TextIO, **kwargs: Any) -> None:
    model, data = load_checkpoint(model_name)
    challenges = json.load(input_file)
    logger.info(f"Read {len(challenges)} challenges from input file")

    df = eval_model_on_challenges(model, data, challenges, **kwargs)
    df.to_csv(result_file, index=False)


def eval_model_on_challenges(
    model: EncoderDecoderLlmTrainingModule,
    data: EncoderDecoderLlmDataModule,
    challenges: list[dict[str, Any]],
    num_beams: int = 128,
    max_sampling_steps: int | None = None,
    peak_mz_noise: float = 1e-2,
    skip_wrong_atom_count: bool = True,
    keep_partial_samples: bool = False,
    match_hydrogen_count: bool = False,
) -> pd.DataFrame:
    dfs = []
    for i, ch in enumerate(challenges):
        logger.debug(f"Predicting challenge {i + 1} / {len(challenges)} ...")
        preds = eval_model_on_challenge(
            model,
            data,
            ch["spectrum"],
            ch["sum_formula"],
            ch.get("smiles_prefix", None),
            ch.get("max_sampling_steps", max_sampling_steps),
            ch.get("true_smiles", None),
            ch.get("adduct_shift", 1.007825032),
            num_beams,
            peak_mz_noise,
            skip_wrong_atom_count,
            keep_partial_samples,
            match_hydrogen_count,
        )
        best = preds["tanimoto"].max()
        if best >= 0.0:
            logger.debug(f'Valid predictions: {len(preds)} - Best tanimoto: {preds["tanimoto"].max()}')
        else:
            logger.debug(f'Valid predictions: {len(preds)}')

        dfs.append(preds)

    df = pd.concat(dfs).reset_index(drop=True)
    return df


def eval_model_on_challenge(
    model: EncoderDecoderLlmTrainingModule,
    data: EncoderDecoderLlmDataModule,
    spec: list[tuple[float, float]],
    mol_formula: str | dict[int, int],
    smiles_prefix: str | None,
    max_sampling_steps: int | None,
    true_smiles: str | None,
    adduct_shift: float,
    num_predictions: int = 128,
    peak_mz_noise: float = 1e-2,
    skip_wrong_atom_count: bool = True,
    return_partial_samples: bool = False,
    match_hydrogen_count: bool = False,
) -> pd.DataFrame:
    if isinstance(mol_formula, str):
        mol_formula = parse_mol_formula(mol_formula)
    if max_sampling_steps is None:
        max_sampling_steps = 30 + sum(v for k, v in mol_formula.items() if k > 1)

    data.val_dset.peak_mz_noise = peak_mz_noise
    generated = postprocess_generated_samples(
        sample_predictions_beam_search(
            model,
            data,
            spec,
            mol_formula,
            max_sampling_steps,
            num_predictions,
            smiles_prefix,
            adduct_shift,
            skip_wrong_atom_count,
            return_partial_samples,
        ),
        match_hydrogen_count=match_hydrogen_count,
        real_mol=wrdkit.mol_from_smiles(true_smiles) if true_smiles is not None else None,
    )

    rows = []
    for gen in generated:
        d = dataclasses.asdict(gen)
        res = {k: d[k] for k in ["perplexity", "tanimoto", "pred_smiles", "generation_count"]}
        rows.append(res)

    df = pd.DataFrame(rows).reset_index()
    return df


if __name__ == "__main__":
    main()
