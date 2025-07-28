import os

import pytest

from seismiq.prediction.llm.data_module import EncoderDecoderLlmDataModule
from seismiq.prediction.llm.model_loading import get_available_models, load_checkpoint
from seismiq.prediction.llm.training_module import EncoderDecoderLlmTrainingModule


@pytest.fixture(autouse=True, scope="session")
def dev_env():
    default_settings = {
        "SEISMIQ_CHECKPOINTS_FOLDER": "dev/checkpoints",
        "SEISMIQ_TEST_DATA_FOLDER": "dev/test_datasets",
    }

    for k, v in default_settings.items():
        if k not in os.environ:
            os.environ[k] = v


@pytest.fixture(scope="session", params=get_available_models())
def model_checkpoint(request) -> tuple[EncoderDecoderLlmTrainingModule, EncoderDecoderLlmDataModule]:
    return load_checkpoint(request.param)
