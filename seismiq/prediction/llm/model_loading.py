import os
from io import BytesIO
from typing import Any

import torch
from loguru import logger

from seismiq.prediction.data.storage import InMemoryDataStorage, OnDiskBlockDataStorage
from seismiq.prediction.llm.data_module import EncoderDecoderLlmDataModule
from seismiq.prediction.llm.training_module import EncoderDecoderLlmTrainingModule
from seismiq.utils import config


def get_available_models() -> list[str]:
    return [
        "seismiq_pretrained",
        "seismiq_finetuned_casmi",
    ]


def load_checkpoint(
    ckpt_path: str,
    setup_stage: str | None = "predict",
    map_location: str | None = None,
    data_hparams_override: dict[str, Any] | None = None,
    model_hparams_override: dict[str, Any] | None = None,
) -> tuple[EncoderDecoderLlmTrainingModule, EncoderDecoderLlmDataModule]:
    if map_location is None and not torch.cuda.is_available():
        map_location = "cpu"

    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config.SEISMIQ_CHECKPOINTS_FOLDER(), ckpt_path) + ".ckpt"
    logger.info(f"Loading checkpoint from {ckpt_path} ...")
    with open(ckpt_path, "rb") as f:
        ckpt = BytesIO(f.read())

    if data_hparams_override is None:
        data_hparams_override = {}
    if 'tokenizer_override' not in data_hparams_override:
        ov = config.SEISMIQ_TOKENIZER_OVERRIDE(None)
        if ov:
            data_hparams_override["tokenizer_override"] = ov

    model = load_model_from_checkpoint(ckpt, setup_stage, map_location, **(model_hparams_override or {}))
    data = load_data_from_checkpoint(ckpt, setup_stage, map_location, **(data_hparams_override or {}))

    return model, data


def load_model_from_checkpoint(
    ckpt: BytesIO,
    setup_stage: str | None = "predict",
    map_location: str | None = None,
    strict: bool | None = False,
    **hparams_override: Any,
) -> EncoderDecoderLlmTrainingModule:
    logger.debug("Loading model ..")
    ckpt.seek(0)
    model = EncoderDecoderLlmTrainingModule.load_from_checkpoint(
        ckpt, map_location=map_location, strict=strict, **hparams_override
    )

    if setup_stage is not None:
        logger.debug(f"Setting up model for stage {setup_stage} ...")
        model.setup(setup_stage)
        if setup_stage in ("predict", "test"):
            logger.debug("setting model to eval mode")
            model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.debug("Model loaded to GPU")
    else:
        model = model.to("cpu")
        logger.debug("Model loaded on CPU")

    logger.info("Model loaded successfully from checkpoint!")
    return model


def load_data_from_checkpoint(
    ckpt: BytesIO,
    setup_stage: str | None = "predict",
    map_location: str | None = None,
    data_path_override: str | None = None,
    **hparams_override: Any,
) -> EncoderDecoderLlmDataModule:
    logger.debug("Loading data from checkpoint ...")
    ckpt.seek(0)
    data = EncoderDecoderLlmDataModule.load_from_checkpoint(ckpt, map_location=map_location, **hparams_override)

    if data_path_override:
        # FIXME this is a horrible solution
        logger.debug(f"Will look for data samples in {data_path_override} ...")
        sto = data.storage  # type: ignore
        if isinstance(sto, OnDiskBlockDataStorage):
            sto.base_folder = data_path_override
        elif isinstance(sto, InMemoryDataStorage):
            sto.datafile = data_path_override

    if setup_stage is not None:
        data.setup(setup_stage)

    logger.info("Data loaded successfully from checkpoint!")
    return data
