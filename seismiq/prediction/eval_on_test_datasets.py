import dataclasses
import os
import pickle
import sys
from collections.abc import Iterable

import click
import numpy as np
import pandas as pd
from loguru import logger

from seismiq.prediction.data import test_datasets
from seismiq.prediction.llm.data_module import EncoderDecoderLlmDataModule
from seismiq.prediction.llm.model_loading import get_available_models, load_checkpoint
from seismiq.prediction.llm.sampling import postprocess_generated_samples, sample_predictions_beam_search
from seismiq.prediction.llm.training_module import EncoderDecoderLlmTrainingModule
from seismiq.utils import rdkit_wrapper as wrdkit
from seismiq.utils.mol_utils import get_mol_formula


@click.group
def main() -> None:
    pass


@main.command()
@click.argument(
    "result-folder",
    type=click.Path(file_okay=False, writable=True),
    default="./development/test_results/",
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
            all_preds.append(model_preds.assign(challenge=ch.challenge))
            logger.debug(f'valid preds: {len(model_preds)} - best tanimoto: {model_preds["tanimoto"].max()}')

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

    generated = postprocess_generated_samples(
        sample_predictions_beam_search(
            model,
            data,
            spec,
            get_mol_formula(smiles),
            smiles_prefix=None,
            max_sampling_steps=20 + len(smiles),
            num_beams=num_predictions,
            adduct_shift=adduct_shift,
        ),
        match_hydrogen_count=False,
        real_mol=wrdkit.mol_from_smiles(smiles),
    )

    rows = []
    for gen in generated:
        d = dataclasses.asdict(gen)
        res = {k: d[k] for k in ["perplexity", "tanimoto", "pred_smiles", "generation_count"]}
        rows.append(res)

    df = pd.DataFrame(rows).reset_index()
    if "tanimoto" not in df.columns:
        df["tanimoto"] = np.nan
    return df


if __name__ == "__main__":
    main()
