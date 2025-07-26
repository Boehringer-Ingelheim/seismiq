import csv
import itertools
import json
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from getpass import getuser
from typing import Generic

import numpy as np
from loguru import logger
from rdkit import Chem, RDLogger

from seismiq.prediction.data import test_datasets
from seismiq.prediction.data.sample import SeismiqSample, TSample
from seismiq.utils import rdkit_wrapper as wrdkit
from seismiq.utils.mol_utils import get_mol_wt, parse_clean_smiles
from seismiq.utils.parallel_utils import parallel_threads
from seismiq.vendor.cfm_id import cfm_predict
from seismiq.vendor.frag_genie import FragGeniePredictor


class DataPreparer(Generic[TSample], ABC):
    @abstractmethod
    def prepare_data(self) -> Iterator[TSample]:
        pass


class CsvDataPreparer(DataPreparer[SeismiqSample]):
    """
    This data preparer will read a CSV file and yield SeismiqSample objects.
    The CSV file should contain columns for measurement_id, compound_id, measurement_origin,
    is_experimental, smiles, peaks, adduct_shift, and precursor_mass.
    """

    def __init__(self, csv_file: str):
        self.csv_file = csv_file

    def prepare_data(self) -> Iterator[SeismiqSample]:
        csv.field_size_limit(sys.maxsize)
        with open(self.csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield SeismiqSample(
                    measurement_id=int(row["measurement_id"]),
                    compound_id=int(row["compound_id"]),
                    measurement_origin=row["measurement_origin"],
                    is_experimental=row["is_experimental"].lower() == "true",
                    smiles=row["smiles"],
                    peaks=json.loads(row["peaks"]),
                    adduct_shift=float(row.get("adduct_shift", 1.007825032)),
                    mol=Chem.MolFromSmiles(row["smiles"]),
                    precursor_mass=row.get("precursor_mass"),
                )


class SyntheticDataPreparer(DataPreparer[SeismiqSample]):
    """
    This data preparer will process the given list of smiles
    and predict their spectra with cfm-id and fraggenie
    """

    # TODO take a tokenizer as argument and exlude all smiles that can't be tokenized

    def __init__(
        self,
        smiles: list[str] | None = None,
        min_peaks: int = 4,
        smiles_origins: dict[str, str] | None = None,
        n_workers: int = 4,
        job_size: int = 16,
    ):
        self.smiles = smiles
        self.min_peaks = min_peaks
        self.smiles_origins = smiles_origins or {}
        self.n_workers = n_workers
        self.job_size = job_size

    def prepare_data(self) -> Iterator[SeismiqSample]:
        RDLogger.DisableLog("rdApp.*")  # type: ignore
        if not self.smiles:
            raise RuntimeError("no smiles given")
        logger.info(f"will process {len(self.smiles)} molecules")

        sample_count = num_compounds = 0
        tasks = [(num_compounds + i, x) for i, x in enumerate(self.smiles)]
        prepared = parallel_threads(
            tasks,
            self._worker,
            job_size=self.job_size,
            n_jobs=self.n_workers,
            progress_bar=False,
        )

        batch_count = int(np.ceil(len(tasks) / self.job_size))
        for i, batch in enumerate(prepared):
            for sam in batch:
                sam.measurement_id = sample_count
                sample_count += 1
                yield sam

            logger.debug(f"Processed {i + 1} / {batch_count} batches, produced {sample_count} samples ...")

        RDLogger.EnableLog("rdApp.*")
        logger.info("Done preparing!")

    def _worker(self, batch: list[tuple[int, str]]) -> list[SeismiqSample]:
        clean_smiles: dict[str, tuple[int, wrdkit.Mol]] = {}
        for idx, smi in batch:
            cleaned = parse_clean_smiles(smi)
            if cleaned is not None:
                smi, mol = cleaned
                clean_smiles[smi] = idx, mol
            else:
                logger.warning(f'failed to clean "{smi}", ignoring ...')

        prepared = itertools.chain(
            self._cfm_id_predictions(list(clean_smiles.keys())),
            self._frag_genie_predictions(list(clean_smiles.keys())),
        )

        res = []
        for i, prep in enumerate(prepared):
            if len(prep.peaks) < self.min_peaks:
                continue

            smiles_origin = self.smiles_origins.get(prep.smiles)
            if smiles_origin:
                prep.measurement_origin = f"{smiles_origin}/{prep.measurement_origin}"
            prep.compound_id, prep.mol = clean_smiles[prep.smiles]
            prep.measurement_id = i
            res.append(prep)

        return res

    def _cfm_id_predictions(self, smiles: list[str]) -> Iterator[SeismiqSample]:
        with tempfile.NamedTemporaryFile(mode="w+", dir=f"/scratch/{getuser()}") as tmp:
            for i, s in enumerate(smiles):
                tmp.write(f"{i} {s}\n")
            tmp.seek(0)

            status, preds = cfm_predict(
                smiles_or_inchi_or_file=tmp.name,
                ionization="+",
                probability_threshold=0.001,
                annotate_fragments=True,
                apply_postproc=True,
            )
            status.check_returncode()

        for sid, pr in preds.items():
            for energy, spectrum in pr.peaks.items():
                spec = [(p.mass, p.abundance) for p in spectrum]
                yield SeismiqSample(
                    measurement_id=-1,
                    compound_id=-1,
                    measurement_origin=f"cfm_id/{energy}",
                    is_experimental=False,
                    smiles=smiles[int(sid)],
                    peaks=spec,
                    mol=None,
                    adduct_shift=1.007825032,
                    precursor_mass=pr.precursor_mass,
                )

    def _frag_genie_predictions(self, smiles: list[str]) -> Iterator[SeismiqSample]:
        for depth in [1, 2, 3]:
            all_preds = FragGeniePredictor(
                frag_recursion_depth=depth,
                min_frag_mass=40,
                raise_errors=True,
            ).predict_spectra(smiles)

            for pred in all_preds:
                mol = wrdkit.mol_from_smiles(pred.smiles)
                monoiso_mol_mass = get_mol_wt(mol)
                monoiso_h_mass = 1.007825032

                spec = [(p, 1.0) for p in pred.peaks]

                yield SeismiqSample(
                    measurement_id=-1,
                    compound_id=-1,
                    measurement_origin=f"frag_genie/{depth}",
                    is_experimental=False,
                    smiles=pred.smiles,
                    peaks=spec,
                    mol=None,
                    adduct_shift=monoiso_h_mass,
                    precursor_mass=monoiso_h_mass + monoiso_mol_mass,
                )


class TestDatasetsPreparer(DataPreparer[SeismiqSample]):
    def __init__(self, dataset_names: list[str], base_preparer: SyntheticDataPreparer):
        self.dataset_names = dataset_names
        self.base_preparer = base_preparer

    def prepare_data(self) -> Iterator[SeismiqSample]:
        smiles_origins = {}
        for name in self.dataset_names:
            ds = getattr(test_datasets, name)
            for ch in ds():
                smiles_origins[ch.smiles] = f"{ch.dataset}/{ch.challenge}"

        logger.info(f"read {len(smiles_origins)} distinct molecules")
        self.base_preparer.smiles_origins = smiles_origins
        self.base_preparer.smiles = list(smiles_origins.keys())

        yield from self.base_preparer.prepare_data()
