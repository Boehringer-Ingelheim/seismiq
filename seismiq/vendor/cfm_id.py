import dataclasses
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from tempfile import TemporaryDirectory
from typing import Any, TextIO

import click
from loguru import logger

from seismiq.utils import config


@dataclasses.dataclass
class CfmPredictedPeak:
    mass: float  # charge assumed to be +/- 1
    abundance: float  # 100 for the most common
    fragment_ids: list[int]  # list of IDs of possible fragments generating this peak
    fragment_logits: list[float]  # list of logits (I assume) of each fragment


@dataclasses.dataclass
class CfmFragment:
    fragment_id: int
    mass: float  # charge assumed to be +/- 1
    smiles: str


@dataclasses.dataclass
class CfmPrediction:
    cfm_version: str
    spectra_type: str
    spectra_ionization: str
    mol_id: str
    mol_smiles: str
    mol_inchikey: str
    mol_formula: str
    precursor_mass: float
    peaks: dict[str, list[CfmPredictedPeak]]
    fragments: list[CfmFragment]


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("smiles_or_inchi_or_file")
@click.argument("ionization", type=click.Choice(["+", "-"]))
@click.argument("output-file", type=click.File("w"))
@click.option(
    "-p",
    "--probability-threshold",
    default=0.001,
    help="The probability below which to prune unlikely fragmentations during"
    "fragmentation graph generation (default 0.001).",
)
@click.option(
    "-a",
    "--annotate-fragments",
    default="1",
    type=click.Choice(["0", "1"]),
    help="Whether to include fragment information in the output spectra " "(0 = NO (DEFAULT), 1 = YES).",
)
@click.option(
    "-P",
    "--apply-postproc",
    default="1",
    type=click.Choice(["0", "1"]),
    help="Whether or not to post-process predicted spectra to take the top 80% of energy "
    "(at least 5 peaks), or the highest 30 peaks (whichever comes first) (0 = OFF, "
    "1 = ON (default)). If turned off, will output a peak for every possible fragment "
    "of the input molecule, as long as the prob_thresh argument above is set to 0.0.",
)
@click.option("-v", "--verbose", is_flag=True, help="Print stdout/stderr of cfm-predict")
def predict(output_file: TextIO, verbose: bool, **kwargs) -> None:
    assert output_file is not None

    status, preds = cfm_predict(**kwargs)
    if verbose or status.returncode != 0:
        print("==== ARGS", file=sys.stderr)
        print(status.args, file=sys.stderr)
        print("==== STDOUT", file=sys.stderr)
        print(status.stdout.decode("utf8"), file=sys.stderr)
        print("==== STDERR", file=sys.stderr)
        print(status.stderr.decode("utf8"), file=sys.stderr)
        print("====", file=sys.stderr)
        status.check_returncode()

    assert preds is not None
    json.dump({key: dataclasses.asdict(value) for key, value in preds.items()}, output_file)
    output_file.write("\n")


def cfm_predict(
    smiles_or_inchi_or_file: str,
    ionization: str,
    probability_threshold: float = 0.001,
    annotate_fragments: bool = True,
    apply_postproc: bool = True,
    cfm_program: str | None = None,
) -> tuple[subprocess.CompletedProcess[Any], dict[str, CfmPrediction]]:
    """
    Predicts ESI-MS/MS spectra for the given molecules using CFM-ID.

    Note (in case of multiple input SMILES): cfm-predict processes each SMILE
    string sequentially and stops at the first error it encounters. Therefore,
    all smiles after the one that generated the error will not be processed.

    Args:
        smiles_or_inchi_or_file (str): The smiles or inchi string of the
            structure whose spectra you want to predict. Or alternatively a .txt
            file containing a list of space-separated (id, smiles_or_inchi) pairs
            one per line.
        ionization (str): Whether to use positive ('+') or negative ('-')
            ionization.
        probability_threshold (float, optional): The probability below which to
            prune unlikely fragmentations during fragmentation graph generation.
            Defaults to 0.001.
        annotate_fragments (bool, optional): Whether to include fragment
            information in the output spectra. Defaults to True.
        apply_postproc (bool, optional): Whether or not to post-process
            predicted spectra to take the top 80% of energy (at least 5 peaks), or
            the highest 30 peaks (whichever comes first). If turned off, will
            output a peak for every possible fragment of the input molecule, as
            long as the prob_thresh argument above is set to 0.0. Defaults to True.

    Returns:
        tuple[subprocess.CompletedProcess, Optional[dict[str, CfmPrediction]]]:
        The final status of the process, including stdout, stderr, and the full
        command line, as well as the predictions of CFM-ID.
    """

    cfm_program = cfm_program or config.SEISMIQ_CFM_ID_PROGRAM()
    logger.trace(f"Using cfm-id program {cfm_program}")

    with TemporaryDirectory(dir=config.SEISMIQ_TEMP_FOLDER(default=None), prefix="cfmid_tmp_") as temp_dir:
        in_folder = os.path.join(temp_dir, "input")
        out_folder = os.path.join(temp_dir, "output")

        os.makedirs(in_folder)
        os.makedirs(out_folder)

        cfm_input = "input.txt"
        cfm_output = "NullId.log"

        if os.path.isfile(smiles_or_inchi_or_file):
            shutil.copy(smiles_or_inchi_or_file, os.path.join(in_folder, cfm_input))
        else:
            with open(os.path.join(in_folder, cfm_input), "w") as f:
                f.write(f"NullId {smiles_or_inchi_or_file}")

        args = [
            cfm_program,
            in_folder,
            out_folder,
            cfm_input,
            cfm_output,
            str(probability_threshold),
            ionization,
            str(int(annotate_fragments)),
            str(int(apply_postproc)),
        ]

        logger.trace(f"Running subprocess with the following args: {args}")
        proc_status = subprocess.run(args, capture_output=True)
        logger.trace(f"Process ended with status {proc_status}")

        predictions = {}
        tmp_content = os.listdir(out_folder)
        logger.trace(f"Content of temporary output folder {out_folder}: {tmp_content}")
        for fname in tmp_content:
            fpath = os.path.join(out_folder, fname)
            with open(fpath) as f:
                for pred in parse_cfm_output(f):
                    predictions[pred.mol_id] = pred

    return proc_status, predictions


@main.command()
@click.argument("log-file", type=click.File("r"))
@click.argument("output-file", type=click.File("w"))
@click.pass_context
def parse_output(ctx: click.Context, log_file: TextIO, output_file: TextIO) -> None:
    res = []
    for pred in parse_cfm_output(log_file):
        res.append(dataclasses.asdict(pred))
        logger.info(f"Read molecule {pred.mol_id} with formula {pred.mol_formula}")
        logger.debug(f"Obtained {len(pred.fragments)} fragments")
        for e, ps in pred.peaks.items():
            logger.debug(f"Energy level {e} has {len(ps)} peaks")
    json.dump(res, output_file)


def parse_cfm_output(outf: TextIO) -> Iterable[CfmPrediction]:
    rows = [r.strip() for r in outf]

    i = 0
    while i < len(rows):
        if rows[i]:
            i, pred = parse_one_cfm_output(i, rows)
            yield pred
        else:
            i += 1
            continue


def parse_one_cfm_output(i: int, rows: list[str]) -> tuple[int, CfmPrediction]:
    attrs: dict[str, str] = {}
    energies: dict[str, list[CfmPredictedPeak]] = {}
    cfm_version = spectra_type = spectra_ionization = fragments = None

    start = i
    while i < len(rows):
        if i - start == 0:
            parts = rows[i].split()
            if parts[0] != "#In-silico" or parts[3] != "Spectra":
                raise RuntimeError("file is not from CFM-ID 4.x")
            spectra_type, spectra_ionization = parts[1], parts[2]
            i += 1
        elif i - start == 1:
            if not rows[i].startswith("#PREDICTED BY CFM-ID 4."):
                raise RuntimeError("file is not from CFM-ID 4.x")
            cfm_version = rows[i].split()[-1]
            i += 1
        elif rows[i].startswith("#"):
            k, *v = rows[i][1:].split("=")
            attrs[k] = "=".join(v)
            i += 1
        elif rows[i].startswith("energy"):
            en_level = rows[i]
            i, peaks = parse_energy_peaks(i + 1, rows)
            energies[en_level] = peaks
        elif not rows[i]:
            # FIXME does this work when cfm-predict did not annotate fragments?
            i, fragments = parse_fragments(i + 1, rows)
            break
        else:
            raise RuntimeError("error parsing file")

    assert cfm_version is not None, "could not read cfm version (empty CFM-ID output file?)"
    assert spectra_type is not None, "could not read spectrum type (empty CFM-ID output file?)"
    assert spectra_ionization is not None, "could not read spectrum ionization (empty CFM-ID output file?)"
    assert fragments is not None, "empty fragments"

    try:
        pred = CfmPrediction(
            cfm_version=cfm_version,
            spectra_type=spectra_type,
            spectra_ionization=spectra_ionization,
            mol_id=attrs.pop("ID"),
            mol_smiles=attrs.pop("SMILES"),
            mol_inchikey=attrs.pop("InChiKey"),
            mol_formula=attrs.pop("Formula"),
            precursor_mass=float(attrs.pop("PMass")),
            peaks=energies,
            fragments=fragments,
        )
    except KeyError as exc:
        raise KeyError("Could not find all attributes, defined: %s" % attrs) from exc

    if attrs:
        print("WARNINGS: unknown attributes found:", ", ".join(attrs.keys()))
    return i, pred


def parse_energy_peaks(i: int, rows: list[str]) -> tuple[int, list[CfmPredictedPeak]]:
    peaks: list[CfmPredictedPeak] = []
    while not rows[i].startswith("energy") and rows[i]:
        parts = rows[i].split()
        mass = float(parts[0])
        abundance = float(parts[1])

        num_frags = (len(parts) - 2) // 2
        fragment_ids = [int(x) for x in parts[2 : 2 + num_frags]]

        assert parts[2 + num_frags][0] == "(" and parts[-1][-1] == ")"
        fragment_logits = [float(x.replace("(", "").replace(")", "")) for x in parts[2 + num_frags :]]
        peaks.append(
            CfmPredictedPeak(
                mass=mass,
                abundance=abundance,
                fragment_ids=fragment_ids,
                fragment_logits=fragment_logits,
            )
        )

        i += 1

    return i, peaks


def parse_fragments(i: int, rows: list[str]) -> tuple[int, list[CfmFragment]]:
    fragments: list[CfmFragment] = []
    while i < len(rows):
        if not rows[i]:
            break

        fragment_id, mass, smiles = rows[i].split()
        fragments.append(CfmFragment(int(fragment_id), float(mass), smiles))
        i += 1

    return i, fragments


def to_cfm_result(obj: dict[str, Any]) -> CfmPrediction:
    cp = CfmPrediction(**obj)
    cp.fragments = [CfmFragment(**f) for f in cp.fragments]  # type: ignore
    cp.peaks = {
        k: [CfmPredictedPeak(**p) for p in v]  # type: ignore
        for k, v in cp.peaks.items()  # type: ignore
    }
    return cp


if __name__ == "__main__":
    main()
