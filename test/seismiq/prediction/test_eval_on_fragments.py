from seismiq.prediction.data.test_datasets import casmi_2016
from seismiq.prediction.eval_on_fragments import eval_model


def test_eval_model(model_checkpoint) -> None:
    model, data = model_checkpoint
    challenges = list(casmi_2016())[6:7]
    res = eval_model(model, data, challenges)

    assert set(res.columns.tolist()) == {
        "index",
        "perplexity",
        "tanimoto",
        "pred_smiles",
        "generation_count",
        "bond_idx",
        "dummy_idx",
        "smiles_prompt",
        "missing_smiles",
        "given_atoms",
        "missing_atoms",
        "challenge",
        "dataset",
    }
    assert res["perplexity"].gt(0).all()
    assert res["tanimoto"].ge(0).all()
    assert res["tanimoto"].le(1).all()
    assert res["generation_count"].gt(0).all()
    assert list(sorted(res["bond_idx"].unique().tolist())) == [1, 3, 4]
    assert list(sorted(res["dummy_idx"].unique().tolist())) == [2, 4, 5, 9, 10, 12]
    assert res["given_atoms"].gt(0).all()
    assert res["missing_atoms"].gt(0).all()
    assert res["challenge"].unique().tolist() == [challenges[0].challenge]
    assert res["dataset"].unique().tolist() == [challenges[0].dataset]
