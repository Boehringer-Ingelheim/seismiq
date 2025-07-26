from seismiq.prediction.llm.data_module import EncoderDecoderLlmDataModule
from seismiq.prediction.llm.training_module import EncoderDecoderLlmTrainingModule


def test_load_checkpoint(model_checkpoint) -> None:
    model, data = model_checkpoint
    assert isinstance(model, EncoderDecoderLlmTrainingModule)
    assert isinstance(data, EncoderDecoderLlmDataModule)
