from seismiq.prediction.data.test_datasets import casmi_2016
from seismiq.prediction.eval_on_test_datasets import eval_model


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
        "challenge",
        "dataset",
    }
    assert res["perplexity"].gt(0).all()
    assert res["tanimoto"].ge(0).all()
    assert res["tanimoto"].le(1).all()
    assert res["generation_count"].gt(0).all()
    assert res["challenge"].unique().tolist() == [challenges[0].challenge]
    assert res["dataset"].unique().tolist() == [challenges[0].dataset]
