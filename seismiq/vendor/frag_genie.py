import json
import os
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from tempfile import TemporaryDirectory
from typing import TextIO

import click
import pandas as pd
from loguru import logger

from seismiq.utils import config


@dataclass
class FragGeniePrediction:
    smiles: str
    peaks: list[float]
    formulas: list[str]


@click.command()
@click.argument("smiles")
@click.argument("output-file", type=click.File("w"))
@click.option("-d", "--frag-recursion-depth", default=3)
@click.option("-m", "--min-frag-mass", default=80.0)
def main(smiles: str, output_file: TextIO, frag_recursion_depth: int, min_frag_mass: float) -> None:
    pred = FragGeniePredictor(frag_recursion_depth, min_frag_mass)
    spec = pred.predict_spectra([smiles])
    json.dump([asdict(s) for s in spec], output_file)


class FragGeniePredictor:
    def __init__(
        self,
        frag_recursion_depth: int = 3,
        min_frag_mass: float = 80,
        raise_errors: bool = False,
    ):
        self.frag_recursion_depth = frag_recursion_depth
        self.min_frag_mass = min_frag_mass
        self._raise_errors = raise_errors

    def predict_spectra(self, smiles: list[str]) -> list[FragGeniePrediction]:
        with TemporaryDirectory(dir=config.SEISMIQ_TEMP_FOLDER(default=None), prefix="fraggenie_tmp_") as temp_dir:
            infile = "input.txt"
            outfile = "output.txt"

            with open(os.path.join(temp_dir, infile), "w") as f:
                f.write("smiles\n")
                for sm in smiles:
                    f.write(f"{sm}\n")

            args = [
                config.SEISMIQ_FRAGGENIE_PROGRAM(),
                temp_dir,
                infile,
                outfile,
                str(self.frag_recursion_depth),
                str(self.min_frag_mass),
            ]

            logger.trace(f"Running subprocess with the following args: {args}")
            status = subprocess.run(args, capture_output=True)
            logger.trace(f"Process ended with status {status}")

            if status.returncode != 0:
                print("==== STDOUT")
                print(status.stdout.decode("utf8"))
                print("==== STDERR")
                print(status.stderr.decode("utf8"))
                print("==== ARGS")
                print(status.args)
                print("====")

            if self._raise_errors:
                status.check_returncode()
            return list(self._parse_frag_genie_output_file(os.path.join(temp_dir, outfile)))

    def _parse_frag_genie_output_file(self, outf: str) -> Iterator[FragGeniePrediction]:
        df = pd.read_csv(outf).dropna()
        all_smiles = df["smiles"].values.tolist()
        all_peaks = df["METFRAG_MZ"].values.tolist()
        all_formulas = df["METFRAG_FORMULAE"].values.tolist()
        for sm, ps, fm in zip(all_smiles, all_peaks, all_formulas):
            if ps and fm:
                form = fm[1:-1].split(", ")
                yield FragGeniePrediction(
                    smiles=sm,
                    peaks=json.loads(ps),
                    formulas=form,
                )


if __name__ == "__main__":
    main()
