import os
from typing import overload

RAISE_IF_UNDEFINED = ""


class ConfigFromEnv:
    # all of this just because I like proper type checking:
    #
    # >>> SEISMIQ_DATA_FOLDER = ConfigFromEnv('SEISMIQ_DATA_FOLDER')
    # >>> xx = SEISMIQ_DATA_FOLDER()      # type: str (or exception if not in env)
    # >>> yy = SEISMIQ_DATA_FOLDER("yy")  # type: str (either value in env or 'yy')
    # >>> zz = SEISMIQ_DATA_FOLDER(None)  # type: Optional[str]  (either value in env or None)

    def __init__(self, key: str) -> None:
        self.key = key

    @overload
    def __call__(self) -> str: ...

    @overload
    def __call__(self, default: str) -> str: ...

    @overload
    def __call__(self, default: None) -> str | None: ...

    def __call__(self, default: str | None = RAISE_IF_UNDEFINED) -> str | None:
        if default is RAISE_IF_UNDEFINED:
            return os.environ[self.key]
        else:
            return os.environ.get(self.key, default)


SEISMIQ_DATABASE_FOLDER = ConfigFromEnv("SEISMIQ_DATABASE_FOLDER")
SEISMIQ_LOGS_FOLDER = ConfigFromEnv("SEISMIQ_LOGS_FOLDER")
SEISMIQ_RAW_DATA_FOLDER = ConfigFromEnv("SEISMIQ_RAW_DATA_FOLDER")
SEISMIQ_FRAGGENIE_PROGRAM = ConfigFromEnv("SEISMIQ_FRAGGENIE_PROGRAM")
SEISMIQ_CFM_ID_PROGRAM = ConfigFromEnv("SEISMIQ_CFM_ID_PROGRAM")
SEISMIQ_CHECKPOINTS_FOLDER = ConfigFromEnv("SEISMIQ_CHECKPOINTS_FOLDER")
SEISMIQ_TEST_DATA_FOLDER = ConfigFromEnv("SEISMIQ_TEST_DATA_FOLDER")
SEISMIQ_TOKENIZER_OVERRIDE = ConfigFromEnv("SEISMIQ_TOKENIZER_OVERRIDE")
SEISMIQ_TEMP_FOLDER = ConfigFromEnv("SEISMIQ_TEMP_FOLDER")
