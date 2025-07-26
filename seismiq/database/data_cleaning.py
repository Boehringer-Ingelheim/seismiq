import re

from rdkit import Chem


def normalize_smiles(smiles: str) -> tuple[bool | None, str | None]:
    if smiles.upper().strip() in ("", "N/A", "NA"):
        return None, None
    else:
        norm = smiles.replace("&gt;", "").replace("&lt;", "").replace("N/A", "")
        try:
            if norm.startswith("InChI="):
                norm = Chem.MolToSmiles(Chem.MolFromInchi(norm[len("InChI=") :]))  # type: ignore
            else:
                norm = Chem.MolToSmiles(Chem.MolFromSmiles(norm))  # type: ignore
            return True, norm
        except:
            return False, smiles


def parse_ion_mode_positive(ion_mode: str) -> bool | None:
    ion_mode = ion_mode.strip().lower()
    if "pos" in ion_mode:
        return True
    elif "neg" in ion_mode:
        return False
    else:
        return None


def split_compound_name_collision_energy(text: str) -> tuple[str, float | None]:
    """sometimes the collision energy is recorded in the compound name"""
    m = re.match(r"(.*?)[^\w\d]+((\d+\.?\d*)\s*eV)", text)
    if m is not None:
        return m.group(1), float(m.group(3))
    else:
        return text, None
