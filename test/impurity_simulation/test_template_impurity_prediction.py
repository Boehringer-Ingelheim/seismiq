from seismiq.impurity_simulation.template_impurity_prediction import TemplateImpurityPrediction


def test_predict():
    rxn_smiles = "OC(C1=CC(Cl)=CC=C1)=O.NCCC(C)O"
    list_of_smiles = rxn_smiles.split(".")
    list_of_templates = [
        "[C:1]-[NH2;D1;+0:2].[O;D1;H0:3]=[C;H0;D3;+0:4](-[OH;D1;+0:5])-[c:6]>>[C:1]-[NH;D2;+0:2]-[C;H0;D3;+0:4](=[O;D1;H0:3])-[c:6].[OH2;D0;+0:5]",
        "Cl-[c;H0;D3;+0:1](:[c:2]):[c:3].[C:4]-[NH2;D1;+0:5]>>[C:4]-[NH;D2;+0:5]-[c;H0;D3;+0:1](:[c:2]):[c:3]",
        "[C:1]-[OH;D1;+0:2].[O;D1;H0:3]=[C;H0;D3;+0:4](-[OH;D1;+0:5])-[c:6]>>[C:1]-[O;H0;D2;+0:2]-[C;H0;D3;+0:4](=[O;D1;H0:3])-[c:6].[OH2;D0;+0:5]",
    ]
    impurities = TemplateImpurityPrediction(
        list_of_smiles=list_of_smiles,
        list_of_templates=list_of_templates,
        number_of_cycles=2,
    )

    assert set(impurities.set_of_products) == {
        "CC(O)CCNC(=O)c1cccc(Cl)c1",
        "CC(O)CCNc1cccc(C(=O)NCCC(C)OC(=O)c2cccc(Cl)c2)c1",
        "CC(O)CCNc1cccc(C(=O)OC(=O)c2cccc(Cl)c2)c1",
        "CC(O)CCNc1cccc(C(=O)OC(C)CCN)c1",
        "CC(O)CCNC(=O)c1cccc(NCCC(C)OC(=O)c2cccc(Cl)c2)c1",
        "CC(CCNc1cccc(C(=O)OC(=O)c2cccc(Cl)c2)c1)OC(=O)c1cccc(Cl)c1",
        "CC(O)CCNc1cccc(C(=O)OC(C)CCNc2cccc(C(=O)O)c2)c1",
        "CC(O)CCNc1cccc(C(=O)OC(=O)c2cccc(NCCC(C)O)c2)c1",
        "CC(CCN)OC(=O)c1cccc(NCCC(C)OC(=O)c2cccc(Cl)c2)c1",
        "O=C(OC(=O)c1cccc(Cl)c1)c1cccc(Cl)c1",
        "CC(O)CCNc1cccc(C(=O)O)c1",
        "CC(CCN)OC(=O)c1cccc(Cl)c1",
        "CC(O)CCNC(=O)c1cccc(NCCC(C)O)c1",
        "CC(CCNC(=O)c1cccc(Cl)c1)OC(=O)c1cccc(Cl)c1",
        "CC(O)CCNc1cccc(C(=O)OC(C)CCNC(=O)c2cccc(Cl)c2)c1",
        "CC(CCNc1cccc(C(=O)O)c1)OC(=O)c1cccc(Cl)c1",
        "O",
    }
