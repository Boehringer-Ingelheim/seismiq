import pytest

from seismiq.prediction.llm.data_module import EncoderDecoderLlmDataModule
from seismiq.prediction.llm.model_loading import get_available_models, load_checkpoint
from seismiq.prediction.llm.training_module import EncoderDecoderLlmTrainingModule


@pytest.fixture(scope="session", params=get_available_models())
def model_checkpoint(request) -> tuple[EncoderDecoderLlmTrainingModule, EncoderDecoderLlmDataModule]:
    return load_checkpoint(request.param)
