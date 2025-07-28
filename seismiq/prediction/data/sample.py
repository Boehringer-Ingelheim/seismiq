import dataclasses
from typing import TypeVar

import torch
from rdkit import Chem


@dataclasses.dataclass
class SeismiqSample:
    measurement_id: int
    compound_id: int
    measurement_origin: str
    is_experimental: bool
    smiles: str
    peaks: list[tuple[float, float]]
    mol: Chem.rdchem.Mol
    adduct_shift: float
    precursor_mass: float


TSample = TypeVar("TSample")
TInfo = TypeVar("TInfo")


torch.serialization.add_safe_globals(
    [
        Chem.Mol,
        SeismiqSample,
    ]
)
