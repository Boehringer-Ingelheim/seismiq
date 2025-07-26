from rxnmapper import RXNMapper

from seismiq.impurity_simulation.extract_template import extract_rxn_template, map_rxn, sanitize_reaction_smiles


def test_extract():
    rxn_mapper = RXNMapper()
    reaction_smiles = "OC(C1=CC=CC=C1)=O.CCCN>>O=C(NCCC)C2=CC=CC=C2.O"
    sanitized_reaction_smiles = sanitize_reaction_smiles(reaction_smiles)
    mapped_rxn = map_rxn(rxn_mapper, sanitized_reaction_smiles)
    rxn_template = extract_rxn_template(mapped_rxn)
    assert rxn_template == (
        "[C:1]-[NH;D2;+0:2]-[C;H0;D3;+0:4](=[O;D1;H0:3])-[c:6].[OH2;D0;+0:5]"
        ">>[C:1]-[NH2;D1;+0:2].[O;D1;H0:3]=[C;H0;D3;+0:4](-[OH;D1;+0:5])-[c:6]"
    )
