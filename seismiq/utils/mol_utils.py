import re
from collections.abc import Iterator
from functools import lru_cache

from loguru import logger
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdchem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.MolStandardize import rdMolStandardize

from seismiq.utils import rdkit_wrapper as wrdkit


def tanimoto(r: str, s: str) -> float:
    rf = FingerprintMols.FingerprintMol(wrdkit.mol_from_smiles(r))
    sf = FingerprintMols.FingerprintMol(wrdkit.mol_from_smiles(s))
    tani = DataStructs.TanimotoSimilarity(rf, sf)  # type: ignore
    return tani


def get_mol_formula(smiles: str | None = None, mol: wrdkit.Mol | None = None) -> dict[int, int]:
    if mol is None and smiles is None:
        raise RuntimeError("at least one of mol and smi must be given")
    elif mol is not None and smiles is not None:
        raise RuntimeError("at most one of mol and smi must be given")

    if mol is None:
        assert smiles is not None
        mol = wrdkit.mol_from_smiles(smiles)

    mol_atoms: dict[int, int] = {}
    for a in Chem.AddHs(mol).GetAtoms():  # type: ignore
        n = a.GetAtomicNum()
        mol_atoms[n] = 1 + mol_atoms.get(n, 0)
    return mol_atoms


def parse_mol_formula(formula: str) -> dict[int, int]:
    mol_atoms: dict[int, int] = {}
    pt = Chem.GetPeriodicTable()  # type: ignore
    for sy, ns in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        n = int(ns) if ns else 1
        a = pt.GetAtomicNumber(sy)
        mol_atoms[a] = n
    return mol_atoms


@lru_cache(maxsize=1_000_000)
def parse_clean_smiles(smiles: str, min_atoms: int = 3) -> tuple[str, rdchem.Mol] | None:
    RDLogger.DisableLog("rdApp.*")  # type: ignore
    try:
        smiles = Chem.CanonSmiles(smiles, useChiral=0)
    except:
        return None

    mol = wrdkit.mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() < min_atoms:
        return None

    # remove explicit atom mapping if given
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    # inspiration from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
    # Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)  # type: ignore
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)

    # validate
    newsmi = wrdkit.mol_to_smiles(mol)
    if "H+" in newsmi:
        logger.debug(f"could not remove charged Hs, original {smiles} to {newsmi}")
        return None

    newmol = wrdkit.mol_from_smiles(newsmi)
    if newmol is None:
        logger.debug(f"could not reparse smiles after cleaning, original {smiles} to {newsmi}")

    return newsmi, mol


def get_mol_wt(mol: wrdkit.Mol) -> float:
    pt = Chem.GetPeriodicTable()  # type: ignore
    return sum(
        pt.GetMostCommonIsotopeMass(a.GetAtomicNum())
        for a in Chem.AddHs(mol).GetAtoms()  # type: ignore
    )


def fragment_mol(smiles: str) -> Iterator[tuple[int, str, Chem.Mol, str, Chem.Mol]]:
    # fragments a molecule on all single bonds
    # returns pairs of fragments each having more than a single atom,
    # plus their smiles strings such that the other fragment is last
    mol = Chem.MolFromSmiles(smiles)
    for i, b in enumerate(mol.GetBonds()):
        if b.GetBondTypeAsDouble() == 1.0:
            ff = rdmolops.FragmentOnBonds(mol, [i], addDummies=True)
            frags = rdmolops.GetMolFrags(ff, asMols=True, sanitizeFrags=True)
            if len(frags) == 2:
                try:
                    yield i, frag_to_smiles(frags[0]), frags[0], frag_to_smiles(frags[1]), frags[1]
                except RuntimeError:
                    pass


def frag_to_smiles(frag: Chem.Mol) -> str:
    # transforms a fragment to smiles such that the dummy atom is in last position
    frag_smiles = Chem.MolToSmiles(frag)
    assert frag_smiles[0] in ("[", "*"), frag_smiles  # by default the dummy atom is first

    mol_rev = Chem.MolFromSmiles(frag_smiles)
    mol_ord = AllChem.RenumberAtoms(mol_rev, list(range(len(frag.GetAtoms()) - 1, -1, -1)))
    computed_prefix = Chem.MolToSmiles(mol_ord, canonical=False)

    if computed_prefix[-1] != "]":
        msg = f'could not make prompt for fragment "{frag_smiles}"'
        logger.warning(msg)
        raise RuntimeError(msg)

    return re.sub(r"\[\d+\*\]$", "", computed_prefix)
