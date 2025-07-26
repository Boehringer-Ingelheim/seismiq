"""
Functions in this module simply wrap useful rdkit stuff adding the right type annotation
and saving me from ignoring types all over the place
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors  # type: ignore
from rdkit.Chem.Fingerprints import FingerprintMols  # type: ignore
from rdkit.DataStructs.cDataStructs import ExplicitBitVect as bit_vect  # type: ignore


class Mol(Chem.rdchem.Mol):
    # the Mol type in rdkit is defined in the C library and Python treats
    # this as "Unknown" type
    pass


class ExplicitBitVect(bit_vect):
    # the Mol type in rdkit is defined in the C library and Python treats
    # this as "Unknown" type
    pass


def mol_to_smiles(
    mol: Mol,
    doRandom: bool = False,
    kekuleSmiles: bool = False,
    rootedAtAtom: int = -1,
    canonical: bool = True,
    isomericSmiles: bool = True,
) -> str:
    return Chem.MolToSmiles(  # type: ignore
        mol,
        doRandom=doRandom,
        kekuleSmiles=kekuleSmiles,
        rootedAtAtom=rootedAtAtom,
        canonical=canonical,
        isomericSmiles=isomericSmiles,
    )


def mol_from_smiles(sm: str, sanitize: bool = True) -> Mol:
    return Chem.MolFromSmiles(sm, sanitize=sanitize)  # type: ignore


def mol_from_smarts(sm: str) -> Mol:
    return Chem.MolFromSmarts(sm)  # type: ignore


def mol_add_hs(mol: Mol) -> Mol:
    return Chem.AddHs(mol)  # type: ignore


def calc_mol_formula(mol: Mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)  # type: ignore


def add_hs(mol: Mol) -> Mol:
    return Chem.AddHs(mol)  # type: ignore


def fingerprint_mol(mol: Mol) -> ExplicitBitVect:
    return FingerprintMols.FingerprintMol(mol)


def canon_smiles(smiles: str) -> str:
    return Chem.CanonSmiles(smiles)


def tanimoto_similarity(fp1: ExplicitBitVect, fp2: ExplicitBitVect) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)  # type: ignore
