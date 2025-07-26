import os
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Generic

import numpy as np
import torch
from loguru import logger

from seismiq.prediction.data.sample import TInfo, TSample


class DataStorage(Generic[TSample, TInfo], ABC):
    @abstractmethod
    def is_prepared(self) -> bool:
        pass

    @abstractmethod
    def start_preparation(self) -> None:
        pass

    @abstractmethod
    def save_sample(self, sample: TSample) -> None:
        pass

    @abstractmethod
    def finish_preparation(self, info: TInfo) -> None:
        pass

    @abstractmethod
    def load_dataset_info(self, reload: bool = True) -> TInfo:
        # the checkpoints saved by lightning will also save any cached data
        # of all children of the data module, but sometimes we may want to
        # reload such cached information from scratch ... ask me how I know that
        pass

    @abstractmethod
    def load_sample(self, idx: int) -> TSample:
        pass


class InMemoryDataStorage(DataStorage[TSample, TInfo]):
    def __init__(self, datafile: str):
        self.datafile = datafile
        self._samples: list[TSample] | None = None
        self._info = None

    def is_prepared(self) -> bool:
        return os.path.exists(self.datafile)

    def start_preparation(self) -> None:
        self._samples = []

    def save_sample(self, sample: TSample) -> None:
        if self._samples is None:
            raise RuntimeError("please call start_preparation once before save_sample")
        self._samples.append(sample)

    def finish_preparation(self, info: TInfo) -> None:
        torch.save((self._samples, info), self.datafile)

    def load_dataset_info(self, reload: bool = True) -> TInfo:
        if reload or self._info is None:
            logger.debug(f"reading dataset from {self.datafile}")
            self._samples, self._info = torch.load(self.datafile)
        if self._info is None:
            raise RuntimeError("dataset preparation must be completed before loading info")
        return self._info

    def load_sample(self, idx: int) -> TSample:
        if self._info is None:
            self._samples, self._info = torch.load(self.datafile)
        if self._samples is None:
            raise RuntimeError("no samples were saved when the dataset was prepared")

        return self._samples[idx]


class OnDiskBlockDataStorage(DataStorage[TSample, TInfo]):
    """
    Saves samples on disk, storing many samples in the same file.
    For efficient loading and shuffling during model training,
    it is recommended to use the BlockSampler down below.
    """

    def __init__(self, base_folder: str, block_size: int = 1000):
        self.base_folder = base_folder
        self._info = None
        self._sample_count = self.block_count = 0
        self.block_size = block_size
        self._current_saving_block: list[TSample] | None = None
        self._loaded_block: list[TSample] | None = None
        self._loaded_block_idx: int | None = None

    @property
    def datafile(self) -> str:
        return os.path.join(self.base_folder, "dataset_info.pkl")

    def is_prepared(self) -> bool:
        return os.path.exists(self.datafile)

    def start_preparation(self) -> None:
        self._current_saving_block = []

    def save_sample(self, sample: TSample) -> None:
        if self._current_saving_block is None:
            raise RuntimeError("please call start_preparation before saving samples")

        self._current_saving_block.append(sample)
        self._sample_count += 1

        if len(self._current_saving_block) >= self.block_size:
            self._save_current_block()
            self._current_saving_block = []
            self.block_count += 1
            logger.info(f"saved {self._sample_count} samples so far ...")

    def _save_current_block(self) -> None:
        dest_path = self._block_path(self.block_count)
        dest_folder, _ = os.path.split(dest_path)
        os.makedirs(dest_folder, exist_ok=True)

        retry_count = 0
        try:
            torch.save(self._current_saving_block, dest_path)
        except RuntimeError as exc:
            time.sleep(2**retry_count)
            retry_count += 1
            if retry_count >= 7:
                raise RuntimeError(f"failed to save file {dest_path} after {retry_count - 1} attempts") from exc

    def _block_path(self, block_id: int) -> str:
        sample_idx_parts = [f"{block_id:09d}"[i : i + 3] for i in range(0, 8, 3)]
        return os.path.join(self.base_folder, *sample_idx_parts) + ".pkl"

    def finish_preparation(self, info: TInfo) -> None:
        if self._current_saving_block:
            self._save_current_block()
            self.block_count += 1

        with open(self.datafile, "wb") as f:
            pickle.dump(
                {
                    "info": info,
                    "block_count": self.block_count,
                    "block_size": self.block_size,
                },
                f,
                protocol=4,
            )

    def load_dataset_info(self, reload: bool = False) -> TInfo:
        if reload or self._info is None:
            logger.debug(f"reading dataset info from {self.datafile}")
            with open(self.datafile, "rb") as f:
                data = pickle.load(f)
            self._info, self.block_count = data["info"], data["block_count"]

        if self._info is None:
            raise RuntimeError("dataset preparation must be completed before loading info")

        return self._info

    def load_sample(self, idx: int) -> TSample:
        block_id, offset = self._split_sample_id(idx)
        if self._loaded_block_idx != block_id:
            block_path = self._block_path(block_id)
            self._loaded_block = torch.load(block_path)
            self._loaded_block_idx = block_id

        assert self._loaded_block is not None
        return self._loaded_block[offset]

    def _split_sample_id(self, idx: int) -> tuple[int, int]:
        block = idx // self.block_size
        offset = idx % self.block_size
        return block, offset


class InMemoryDataStorage(DataStorage[TSample, TInfo]):
    def __init__(self, datafile: str):
        self.datafile = datafile
        self._samples: list[TSample] | None = None
        self._info = None

    def is_prepared(self) -> bool:
        return os.path.exists(self.datafile)

    def start_preparation(self) -> None:
        self._samples = []

    def save_sample(self, sample: TSample) -> None:
        if self._samples is None:
            raise RuntimeError("please call start_preparation once before save_sample")
        self._samples.append(sample)

    def finish_preparation(self, info: TInfo) -> None:
        torch.save((self._samples, info), self.datafile)

    def load_dataset_info(self, reload: bool = True) -> TInfo:
        if reload or self._info is None:
            logger.debug(f"reading dataset from {self.datafile}")
            self._samples, self._info = torch.load(self.datafile)
        if self._info is None:
            raise RuntimeError("dataset preparation must be completed before loading info")
        return self._info

    def load_sample(self, idx: int) -> TSample:
        if self._info is None:
            self._samples, self._info = torch.load(self.datafile)
        if self._samples is None:
            raise RuntimeError("no samples were saved when the dataset was prepared")

        return self._samples[idx]


class BlockSampler:
    def __init__(self, indices: list[int], block_size: int, shuffle: bool) -> None:
        self._block_size = block_size
        self._shuffle = shuffle
        self._indices = indices
        self._blocks: dict[int, list[int]] = {}

        # `indices` contain the "global" sample indices contained by the dataset
        # these global indices are used to derive the block that contains the sample
        # however a dataset with n samples will map the "local" indices [0...n) to these
        # global sample indices to determine the actual sample to load
        # therefore we also need to do the same: construct blocks based on global indices
        # but yield indices in [0...n)
        # why like this? because train/validation split means that each dataset contains
        # an arbitrary random subset of the "global" samples
        for local_idx, global_idx in enumerate(indices):
            b = global_idx // block_size
            if b not in self._blocks:
                self._blocks[b] = []
            self._blocks[b].append(local_idx)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[int]:
        block_sequence = list(self._sequence(self._blocks.keys()))  # type: ignore
        for block in block_sequence:
            sample_sequence = list(self._sequence(self._blocks[block]))
            yield from sample_sequence

    def _sequence(self, indices: Sequence[int]) -> Iterator[int]:
        sorted_indices = list(sorted(indices))
        if self._shuffle:
            yield from np.random.choice(sorted_indices, size=len(sorted_indices), replace=False)
        else:
            yield from sorted_indices
