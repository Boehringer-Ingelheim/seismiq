import json
import os
from collections.abc import Iterator
from dataclasses import dataclass

from seismiq.utils import config


@dataclass
class TestChallenge:
    smiles: str
    spectrum: list[tuple[float, float]]
    challenge: str
    dataset: str
    adduct_shift: float


def challenges_from_file(fname: str) -> Iterator[TestChallenge]:
    with open(fname) as f:
        data = json.load(f)
        for row in data:
            yield TestChallenge(**row)


def casmi_2016() -> Iterator[TestChallenge]:
    yield from challenges_from_file(os.path.join(config.SEISMIQ_TEST_DATA_FOLDER(), "casmi_2016.json"))


def casmi_2017() -> Iterator[TestChallenge]:
    yield from challenges_from_file(os.path.join(config.SEISMIQ_TEST_DATA_FOLDER(), "casmi_2017.json"))


def casmi_2022() -> Iterator[TestChallenge]:
    yield from challenges_from_file(os.path.join(config.SEISMIQ_TEST_DATA_FOLDER(), "casmi_2022.json"))


def massspecgym() -> Iterator[TestChallenge]:
    yield from challenges_from_file(os.path.join(config.SEISMIQ_TEST_DATA_FOLDER(), "massspecgym.json"))
