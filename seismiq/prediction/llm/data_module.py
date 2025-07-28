import dataclasses
import pickle
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from loguru import logger
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from seismiq.prediction.data import test_datasets
from seismiq.prediction.data.preparation import DataPreparer, SeismiqSample
from seismiq.prediction.data.storage import BlockSampler, DataStorage, OnDiskBlockDataStorage
from seismiq.prediction.llm.tokenizers import SmilesAtomTokenizer, SmilesDencoder
from seismiq.utils import rdkit_wrapper as wrdkit


@dataclasses.dataclass
class LlmDatasetInfo:
    smiles_samples: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    smiles_tokenizer: SmilesAtomTokenizer = dataclasses.field(default_factory=SmilesAtomTokenizer)


@dataclasses.dataclass
class DecoderOnlyDataSample:
    idx: int
    smiles: str
    tokens: Tensor
    remaining_atoms: Tensor
    atoms: dict[int, int]
    peaks: Tensor
    weight: float
    is_experimental: bool


torch.serialization.add_safe_globals(
    [
        LlmDatasetInfo,
    ]
)


class PeakMassEncoder:
    def __init__(self, dim: int, base: float = 3.0, log_freq_min: float = -5, log_freq_max: float = 7) -> None:
        self.dim = dim

        pt = Chem.GetPeriodicTable()  # type: ignore
        atom_freqs = torch.tensor(
            [pt.GetMostCommonIsotopeMass(at) for at in ["H", "C", "N", "O", "Cl", "S", "P", "K", "F", "Br"]]
        )

        num_mass_freqs = dim // 2 - atom_freqs.shape[-1]
        mass_freqs = base ** torch.linspace(log_freq_min, log_freq_max, steps=num_mass_freqs)

        self.freqs = torch.cat([atom_freqs, mass_freqs])
        assert 2 * self.freqs.shape[-1] == dim, "pls choose a better dimensionality"

    def encode(self, masses: Tensor) -> Tensor:
        s = masses.unsqueeze(-1) / self.freqs
        enc = torch.cat([torch.sin(s), torch.cos(s)], dim=-1)
        return enc


@dataclasses.dataclass
class EncoderDecoderDataBatch:
    indices: list[int]
    smiles_str: list[str]
    token_dencoder: SmilesDencoder
    tokens: Tensor
    token_mask: Tensor
    peaks: Tensor
    peak_mask: Tensor
    remaining_atoms: Tensor
    mol_atoms: list[dict[int, int]]
    weights: Tensor
    is_experimental: Tensor

    def to(self, device: str | torch.device) -> "EncoderDecoderDataBatch":
        for k in dir(self):
            v = getattr(self, k)
            if isinstance(v, Tensor):
                setattr(self, k, v.to(device))
        return self


class EncoderDecoderLlmDataset(Dataset[DecoderOnlyDataSample]):
    def __init__(
        self,
        info: LlmDatasetInfo | None,
        storage: DataStorage[SeismiqSample, Any],
        dencoder: SmilesDencoder,
        indices: list[int],
        peaks_nfreq: int,
        uniform_peak_sampling: bool,
        smiles_augment: bool,
        smiles_augment_prob: float,
        sample_min_num_peaks: int,
        sample_max_num_peaks: int,
        peak_mz_noise: float | None,
        encode_peak_intensity: bool,
        part: str,
        **kwargs: Any,
    ):
        super().__init__()

        self.info = info
        self.dencoder = dencoder
        self.storage = storage
        self.indices = indices
        self.peaks_nfreq = peaks_nfreq
        self.part = part
        self.smiles_augment = smiles_augment
        self.smiles_augment_prob = smiles_augment_prob
        self.uniform_peak_sampling = uniform_peak_sampling
        self.peak_mz_noise = peak_mz_noise
        self.sample_min_num_peaks = sample_min_num_peaks
        self.sample_max_num_peaks = sample_max_num_peaks
        self.encode_peak_intensity = encode_peak_intensity

        self.peak_mz_encoder = PeakMassEncoder(self.peaks_nfreq // 2)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int | Sequence[int]) -> DecoderOnlyDataSample | list[DecoderOnlyDataSample]:  # type: ignore
        if isinstance(idx, int):
            return self._load_one(idx)
        else:
            return [self._load_one(i) for i in idx]

    def _load_one(self, idx: int) -> DecoderOnlyDataSample:
        assert self.info is not None
        datum = self.storage.load_sample(self.indices[idx])

        mol_atoms: dict[int, int] = {}
        monoiso_mass = 0.0
        pt = Chem.GetPeriodicTable()  # type: ignore
        for at in wrdkit.mol_add_hs(datum.mol).GetAtoms():
            monoiso_mass += pt.GetMostCommonIsotopeMass(at.GetSymbol())
            a = at.GetAtomicNum()
            if a not in mol_atoms:
                mol_atoms[a] = 0
            mol_atoms[a] += 1

        sm = self._augment_smiles(datum.mol, datum.smiles)
        tokens = torch.tensor(self.dencoder.encode(sm), dtype=torch.long)
        remaining = self.compute_remaining_atoms(mol_atoms, tokens, validate=True)

        return DecoderOnlyDataSample(
            smiles=sm,
            tokens=tokens,
            remaining_atoms=remaining,
            peaks=self.transform_peaks(datum.peaks, datum.adduct_shift, monoiso_mass),
            atoms=mol_atoms,
            idx=idx,
            weight=len(self.indices) / len(self.info.smiles_samples[datum.smiles]),
            is_experimental=datum.is_experimental,
        )

    def _augment_smiles(self, mol: wrdkit.Mol, smi: str) -> str:
        if not self.smiles_augment or np.random.random() > self.smiles_augment_prob:
            return smi

        for _ in range(10):
            try:
                aug = wrdkit.mol_to_smiles(mol, doRandom=True, kekuleSmiles=(np.random.random() > 0.5))
                if len(aug) > 200:
                    # filter too long (=pointless) augmentations to avoid OOM and use larger batch sizes
                    return smi
                else:
                    return aug
            except:
                pass
        else:
            return smi

    def compute_remaining_atoms(self, mol_atoms: dict[int, int], tokens: Tensor, validate: bool) -> Tensor:
        tokenizer = self.dencoder._smiles_tokenizer

        remaining_oh = torch.zeros(tokens.shape[0], 100)
        remaining_count = dict(mol_atoms)
        for j in range(tokens.shape[0]):
            tok = int(tokens[j])
            if tok > 0:
                at = tokenizer.token_to_atom(tok)
                if at is not None:
                    _, num = at
                    if num not in remaining_count:
                        remaining_count[num] = 0
                    remaining_count[num] -= 1
            elif validate:
                assert all(n == 0 for n in remaining_count.values())

            for k, v in remaining_count.items():
                remaining_oh[j, k] = v

        return remaining_oh

    def transform_peaks(self, peaks: list[tuple[float, float]], adduct_shift: float, mol_monoiso_mass: float) -> Tensor:
        if isinstance(peaks[0], float):
            # FIXME COMPATIBILITY WITH OLD DATA AND CHECKPOINT
            peaks_mz = [cast(float, p) for p in peaks]
            peaks_int = [cast(float, p) for p in peaks]
        else:
            peaks_mz = [float(p[0]) for p in peaks]
            peaks_int = [float(p[1]) for p in peaks]

        # choose how many peaks to sample
        a, b = self.sample_min_num_peaks, self.sample_max_num_peaks
        if a < b:
            num_peaks = np.random.randint(a, min(b, len(peaks_mz))) if len(peaks_mz) > a else a
        else:
            num_peaks = min(b, len(peaks_mz))

        # randomly sample the given number of peaks
        if len(peaks_mz) > num_peaks:
            if not self.uniform_peak_sampling:
                # sampling probability proportional to peak intensity
                peak_probs = np.array(peaks_int)
                peak_probs -= peak_probs.min()
                peak_probs += 1e-4 + 1e-3 * peak_probs.max()
                peak_probs /= peak_probs.sum()
            else:
                # uniform sampling probability
                peak_probs = None

            idx = np.random.choice(np.arange(len(peaks_mz)), size=num_peaks, p=peak_probs, replace=False)
            ps = torch.tensor([peaks_mz[i] for i in idx])
            pi = torch.tensor([peaks_int[i] for i in idx])
        else:
            ps = torch.tensor(peaks_mz)
            pi = torch.tensor(peaks_int)

        # remove mass shift due to the ionization adduct
        ps = ps - adduct_shift

        # add the mass of the neutral loss and intensities
        ps = torch.stack([ps, mol_monoiso_mass - ps, pi], dim=-1)

        # add some noise to peak/loss mass
        if self.peak_mz_noise is not None:
            ps = ps + self.peak_mz_noise * (2 * torch.rand_like(ps) - 1)

        # sort peaks by m/z
        psi = torch.sort(ps, descending=True, dim=0)
        ps = ps[psi.indices[:, 0], :]

        # sinusoidal mass/intensity encoding
        enc = self.peak_mz_encoder.encode(ps)
        enc_mz = enc[:, :2, :]
        if self.encode_peak_intensity:
            enc_is = enc[:, -1, :]
            enc_mz += enc_is.unsqueeze(1)  # add intensity encoding to masses
        enc_mz = torch.flatten(enc_mz, -2, -1)  # put peak and loss encodings side-by-side

        return enc_mz

    def collate(self, samples: list[DecoderOnlyDataSample]) -> EncoderDecoderDataBatch:
        if not isinstance(samples, list):
            samples = [samples]

        padded_tokens = pad_sequence(
            [s.tokens.float() for s in samples],
            padding_value=torch.nan,
            batch_first=True,
        )
        padded_peaks = pad_sequence([s.peaks.float() for s in samples], padding_value=torch.nan, batch_first=True)

        padded_remaining = pad_sequence(
            [s.remaining_atoms.float() for s in samples],
            padding_value=0.0,
            batch_first=True,
        )

        ws = torch.tensor([s.weight for s in samples])
        ws = ws / ws.sum()

        b = EncoderDecoderDataBatch(
            token_dencoder=self.dencoder,
            smiles_str=[s.smiles for s in samples],
            tokens=padded_tokens.nan_to_num(),
            token_mask=padded_tokens.isnan(),
            peaks=padded_peaks.nan_to_num(),
            peak_mask=padded_peaks[:, :, 0].isnan(),
            mol_atoms=[s.atoms for s in samples],
            indices=[s.idx for s in samples],
            weights=ws,
            remaining_atoms=padded_remaining,
            is_experimental=torch.tensor([s.is_experimental for s in samples]),
        )

        return b


class EncoderDecoderLlmDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        storage: DataStorage,  # type: ignore
        peaks_nfreq: int = 64,
        preparer: DataPreparer | None = None,  # type: ignore
        subsample: int | float | None = None,
        subsample_mols: int | None = None,
        smiles_augment: bool = False,
        smiles_augment_prob: float = 0.2,
        uniform_peak_sampling: bool = True,
        max_selfies_len: int = 150,
        split_seed: int = 134214,
        sample_min_num_peaks: int = 5,
        sample_max_num_peaks: int = 50,
        peak_mz_noise: float | None = 2e-3,
        train_val_split_ratio: int = 24,
        tokenizer_override: str | None = None,  # real type Optional[SmilesAtomTokenizer | str]
        train_val_smiles_override: list[str] | None = None,
        encode_peak_intensity: bool = False,
        max_mol_mass: float | None = None,
        remove_test_datasets: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.peaks_nfreq = peaks_nfreq
        self.prepared = False
        # we cast this here as jsonargparse is unable to handle
        # generic type hints in the __init__ declaration
        self.preparer = cast(DataPreparer[SeismiqSample], preparer)
        self.storage = cast(DataStorage[SeismiqSample, LlmDatasetInfo], storage)
        self.subsample = subsample
        self.subsample_mols = subsample_mols
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.smiles_augment = smiles_augment
        self.smiles_augment_prob = smiles_augment_prob
        self.max_selfies_len = max_selfies_len
        self.uniform_peak_sampling = uniform_peak_sampling
        self.split_seed = split_seed
        self.sample_max_num_peaks = sample_max_num_peaks
        self.sample_min_num_peaks = sample_min_num_peaks
        self.peak_mz_noise = peak_mz_noise
        self.train_val_split_ratio = train_val_split_ratio
        # we cast this here as declaring the real type in the constructor would make
        # the moronic jsonargparse library interpret the str as a path (which it is) and
        # try loading it as a utf8 text file (which it isn't)
        self.tokenizer_override = cast(str | SmilesAtomTokenizer, tokenizer_override)
        self.train_val_smiles_override = train_val_smiles_override
        self.encode_peak_intensity = encode_peak_intensity
        self.max_mol_mass = max_mol_mass
        self.remove_test_datasets = remove_test_datasets
        self._info = None

    def prepare_data(self) -> None:
        if self.storage.is_prepared():
            return
        elif self.preparer is None:
            raise RuntimeError("cannot prepare dataset without preparer and storage")

        info = LlmDatasetInfo()
        self.storage.start_preparation()
        for i, sample in enumerate(self.preparer.prepare_data()):
            if sample.smiles not in info.smiles_samples:
                info.smiles_samples[sample.smiles] = []

            info.smiles_samples[sample.smiles].append(i)
            self.storage.save_sample(sample)

        info.smiles_tokenizer = SmilesAtomTokenizer(smiles=info.smiles_samples.keys())
        self.storage.finish_preparation(info)  # type: ignore

    @property
    def dataset_info(self) -> LlmDatasetInfo:
        if self._info is None:
            logger.debug("reading info ...")
            self._info = cast(LlmDatasetInfo, self.storage.load_dataset_info())
        return self._info

    def setup(self, stage: str) -> None:
        super().setup(stage)

        if self.tokenizer_override is None:
            logger.debug("using tokenizer saved with dataset")
            tokenizer = self.dataset_info.smiles_tokenizer
        elif isinstance(self.tokenizer_override, str):
            logger.debug(f"loading tokenizer from file {self.tokenizer_override}")
            with open(self.tokenizer_override, "rb") as f:
                tokenizer = pickle.load(f)
        else:
            logger.debug("*overriding* tokenizer saved with dataset")
            tokenizer = self.tokenizer_override

        tokenizer.add_token("[*]")
        tokenizer.add_token(".")
        logger.info(f"got {len(tokenizer.id_to_token)} different tokens!")

        self.dencoder = SmilesDencoder(tokenizer)

        if stage != "predict":
            info = self.dataset_info
            logger.info("creating train/val split ...")
            self.train_mols, self.train_idx, self.val_mols, self.val_idx = self.split_data(info)

            logger.info(f"Loaded {len(self.train_idx)} training samples ({len(self.train_mols)} molecules)")
            logger.info(f"Loaded {len(self.val_idx)} validation samples ({len(self.val_mols)} molecules)")
        else:
            # this is useful for inference on new data that is not contained in the original training dataset
            logger.info(f"Not loading any samples for stage {stage}")
            self.train_mols, self.train_idx, self.val_mols, self.val_idx = [], [], [], []
            info = None

        self.train_dset = EncoderDecoderLlmDataset(
            info,
            self.storage,
            self.dencoder,
            self.train_idx,
            self.peaks_nfreq,
            self.uniform_peak_sampling,
            self.smiles_augment,
            self.smiles_augment_prob,
            self.sample_min_num_peaks,
            self.sample_max_num_peaks,
            self.peak_mz_noise,
            self.encode_peak_intensity,
            "train",
        )

        self.val_dset = EncoderDecoderLlmDataset(
            info,
            self.storage,
            self.dencoder,
            self.val_idx,
            self.peaks_nfreq,
            encode_peak_intensity=self.encode_peak_intensity,
            part="val",
            # disable augmentations for the validation dataset
            uniform_peak_sampling=False,
            smiles_augment=False,
            smiles_augment_prob=0.0,
            sample_max_num_peaks=self.sample_max_num_peaks,
            sample_min_num_peaks=self.sample_max_num_peaks,
            peak_mz_noise=None,
        )

        logger.info("Data module setup done")

    def split_data(self, info: LlmDatasetInfo) -> tuple[list[str], list[int], list[str], list[int]]:
        if self.train_val_smiles_override:
            logger.debug("*overriding* molecules used for train/val splitting")
            all_smiles = self.train_val_smiles_override
        else:
            logger.debug("using molecules in dataset info for train/val splitting")
            all_smiles = list(info.smiles_samples.keys())

        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(all_smiles)

        train_idx: list[int] = []
        val_idx: list[int] = []

        train_mols: list[str] = []
        val_mols: list[str] = []

        # remove test molecules unless requested
        # NB - MassSpecGym published after this model was trained, we keep it like this this for reproducibility
        if self.remove_test_datasets:
            remove_dataset = [test_datasets.casmi_2016(), test_datasets.casmi_2017(), test_datasets.casmi_2022()]
            test_smiles = set(Chem.CanonSmiles(ch.smiles) for ds in remove_dataset for ch in ds)
        else:
            test_smiles = set()

        count_excluded_test = count_excluded_mass = 0
        for s in all_smiles:
            if s in test_smiles:
                count_excluded_test += 1
                continue
            elif (
                self.max_mol_mass is not None and Descriptors.ExactMolWt(wrdkit.mol_from_smiles(s)) > self.max_mol_mass
            ):
                count_excluded_mass += 1
                continue

            sample_ids = set(info.smiles_samples[s])
            if self.max_selfies_len is None or len(s) <= self.max_selfies_len:
                if not train_idx or (val_idx and len(train_idx) / len(val_idx) < self.train_val_split_ratio):
                    train_mols.append(s)
                    train_idx.extend(sample_ids)
                else:
                    val_mols.append(s)
                    val_idx.extend(sample_ids)

            if (
                self.subsample is not None
                and len(train_idx) + len(val_idx) > self.subsample
                or self.subsample_mols is not None
                and len(train_mols) + len(val_mols) >= self.subsample_mols
            ):
                break

        logger.debug(f"excluded {count_excluded_test} test samples and {count_excluded_mass} samples by mass")

        return train_mols, train_idx, val_mols, val_idx

    def train_dataloader(self) -> DataLoader[DecoderOnlyDataSample]:
        sam = self._get_sampler(self.train_dset, shuffle=True)

        return DataLoader(
            cast(Dataset[DecoderOnlyDataSample], self.train_dset),
            num_workers=self.num_workers,
            collate_fn=self.train_dset.collate,
            batch_size=None,
            sampler=sam,
        )

    def val_dataloader(self) -> DataLoader[DecoderOnlyDataSample]:
        sam = self._get_sampler(self.val_dset, shuffle=False)

        return DataLoader(
            cast(Dataset[DecoderOnlyDataSample], self.val_dset),
            num_workers=self.num_workers,
            collate_fn=self.val_dset.collate,
            batch_size=None,
            sampler=sam,
        )

    def _get_sampler(self, dataset: EncoderDecoderLlmDataset, shuffle: bool) -> Any:
        inner_sampler: Iterable[int]
        if isinstance(self.storage, OnDiskBlockDataStorage):
            inner_sampler = BlockSampler(dataset.indices, self.storage.block_size, shuffle=shuffle)
        elif shuffle:
            inner_sampler = RandomSampler(dataset)
        else:
            inner_sampler = SequentialSampler(dataset)

        return BatchSampler(inner_sampler, self.batch_size, drop_last=False)
