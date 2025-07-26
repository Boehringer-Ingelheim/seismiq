import csv
import dataclasses
import json
import os
from typing import Any

import click
from pyteomics import mgf
from rdkit import Chem
from tqdm import tqdm

import seismiq.database.data_cleaning as data_cleaning
from seismiq.database.database import (
    Compound,
    Device,
    Measurement,
    MeasurementMetadata,
    get_db,
)
from seismiq.utils import config


@click.command()
def main(**kwargs: Any) -> None:
    prog = Program(**kwargs)
    prog.run()


class Program:
    def __init__(self) -> None:
        self.devices_cache: dict[tuple[str, str, str], int] = {}
        self.compounds_cache: dict[str | None, int] = {}
        self.spectrum_cache: set[int] = set()

    def run(self) -> None:
        self.db = get_db()
        self.prefill_compound_cache()
        self.import_pubchem10m_data()
        self.import_zinc250k_data()
        self.import_chembl_data()
        self.import_matchms_data()
        self.import_massbank_data()
        self.import_gnps_data()

    def prefill_compound_cache(self) -> None:
        print("Prefilling compounds cache ...")
        cur = self.db.executesql("SELECT DISTINCT id, smiles FROM compounds")
        for cid, smiles in tqdm(cur):
            self.compounds_cache[smiles] = cid

    def import_pubchem10m_data(self) -> None:
        fname = os.path.join(config.SEISMIQ_RAW_DATA_FOLDER(), "pubchem-10m.txt")
        print("importing from", fname)
        last_commit = 0
        with open(fname) as f:
            count_imported = 0
            for row in tqdm(f):
                try:
                    smi = Chem.CanonSmiles(row, useChiral=0)
                except:
                    smi = None

                if smi is not None:
                    _ = self.get_compound_id(Compound(smi, True))  # type: ignore
                    count_imported += 1

                last_commit += 1
                if last_commit > 10000:
                    self.db.commit()
                    last_commit = 0

        self.db.commit()
        print(f"Imported {count_imported} new SMILES from PubChem")

    def import_zinc250k_data(self) -> None:
        fname = os.path.join(config.SEISMIQ_RAW_DATA_FOLDER(), "zinc250k.csv")
        print("importing from", fname)
        with open(fname) as f:
            count_imported = 0
            for row in tqdm(csv.DictReader(f)):
                mol = Chem.MolFromSmiles(row["smiles"])  # type: ignore
                if mol is not None:
                    _ = self.get_compound_id(Compound(Chem.MolToSmiles(mol), True))  # type: ignore
                    count_imported += 1
        self.db.commit()
        print(f"Imported {count_imported} new SMILES from ZINC 250k")

    def import_chembl_data(self) -> None:
        fname = os.path.join(config.SEISMIQ_RAW_DATA_FOLDER(), "chembl-666k.csv")
        print("importing from", fname)
        with open(fname) as f:
            count_imported = 0
            for row in tqdm(csv.DictReader(f)):
                mol = Chem.MolFromSmiles(row["SMILES"])  # type: ignore
                if mol is not None:
                    _ = self.get_compound_id(Compound(Chem.MolToSmiles(mol), True))  # type: ignore
                    count_imported += 1
        self.db.commit()
        print(f"Imported {count_imported} new SMILES from ChEMBL data")

    def import_matchms_data(self) -> None:
        cnt = self.db(self.db.measurements.origin.startswith("matchms/")).count()
        if cnt > 0:
            print(f"{cnt} measurements with origin matchms already in DB, not overwriting!")
            return
        else:
            print("Importing matchms data ...")

        base_file = os.path.join(
            config.SEISMIQ_RAW_DATA_FOLDER(), "matchms/cleaned_libraries_matchms/ALL_GNPS_NO_PROPOGATED_21_08_2023.mgf"
        )
        with open(base_file) as f:
            data = mgf.read(f)
            for obj in tqdm(data, ncols=0, leave=False, disable=None):
                self.import_matchms_object(obj)

    def import_matchms_object(self, obj: dict[str, Any]) -> None:
        device_id = self.get_device_id(Device("unknown", "unknown", obj["params"]["source_instrument"]))

        compound_id = self.get_compound_id(Compound(obj["params"]["smiles"], True))

        measurement_id = Measurement(
            origin="matchms/%s" % obj["params"]["organism"],
            energy=0,
            is_ionization_positive=obj["params"]["ionmode"] == "Positive",
            is_experimental=True,
            peaks_json=json.dumps([(m, h) for m, h in zip(obj["m/z array"], obj["intensity array"])]),
            ref_compound=compound_id,
            ref_device=device_id,
        ).insert()

        for k, v in obj["params"].items():
            MeasurementMetadata(ref_measurement=measurement_id, key=k, value=v).insert()

    def import_massbank_data(self) -> None:
        base_dir = os.path.join(config.SEISMIQ_RAW_DATA_FOLDER(), "MassBank/MassBank-data-2023.09")
        for dir in tqdm(os.listdir(base_dir), ncols=0):
            fdir = os.path.join(base_dir, dir)
            if not os.path.isdir(fdir):
                continue

            origin = f"MSBNK-{dir}"
            cnt = self.db(self.db.measurements.origin == origin).count()
            if cnt > 0:
                print(f"{cnt} measurements with origin {origin} already in DB, not overwriting!")
            else:
                for fname in tqdm(os.listdir(fdir), ncols=0, leave=False):
                    if fname.endswith(".txt"):
                        self.save_massbank_object(origin, os.path.join(base_dir, dir, fname))
                self.db.commit()

    def save_massbank_object(self, origin: str, fname: str) -> None:
        with open(fname) as f:
            data = parse_massbank_file(list(f))

        if data.smiles is None or data.is_smiles_normalized is None:
            return

        device_id = self.get_device_id(Device("unknown", data.device_model, data.device_technique))

        compound_id = self.get_compound_id(Compound(data.smiles, data.is_smiles_normalized))

        measurement_id = Measurement(
            origin=origin,
            energy=0,
            is_ionization_positive=data.is_ionization_positive,
            is_experimental=True,
            peaks_json=json.dumps(data.peaks),
            ref_compound=compound_id,
            ref_device=device_id,
        ).insert()

        for k, v in data.metadata.items():
            MeasurementMetadata(ref_measurement=measurement_id, key=k, value=v).insert()

    def import_gnps_data(self) -> None:
        gnps_path = os.path.join(config.SEISMIQ_RAW_DATA_FOLDER(), "GNPS/")
        files = [f for f in os.listdir(gnps_path) if f.endswith(".json")]
        for i, fname in enumerate(files):
            print(f"Processing file {i + 1} / {len(files)} : {fname}")

            origin = f"GNPS/{fname}"
            cnt = self.db(self.db.measurements.origin == origin).count()
            if cnt > 0:
                print(f"{cnt} measurements with origin {origin} already in DB, not overwriting!")
            else:
                with open(os.path.join(gnps_path, fname)) as f:
                    for row in tqdm(f):
                        self.save_gnps_object(origin, json.loads(row))

                print(
                    "Database now has %d devices, %d compounds, %d measurements, %d metadata"
                    % (
                        self.db(self.db.devices).count(),
                        self.db(self.db.compounds).count(),
                        self.db(self.db.measurements).count(),
                        self.db(self.db.measurement_metadatas).count(),
                    )
                )

        self.db.commit()

    def save_gnps_object(self, origin: str, data: dict[str, Any]) -> None:
        if data["spectrum_id"] in self.spectrum_cache:
            return
        self.spectrum_cache.add(data["spectrum_id"])

        device_id = self.get_device_id(
            Device(
                manufacturer="generic",
                model=data["Instrument"],
                mass_spec_technique=data["Ion_Source"],
            )
        )

        is_smiles_normalized, smiles = data_cleaning.normalize_smiles(data["Smiles"])
        _, energy = data_cleaning.split_compound_name_collision_energy(data["Compound_Name"])
        compound_id = self.get_compound_id(
            Compound(
                smiles=smiles,
                is_smiles_normalized=is_smiles_normalized,
            )
        )

        measurement_id = Measurement(
            origin=origin,
            energy=energy,
            is_experimental=True,
            peaks_json=data["peaks_json"],
            ref_device=device_id,
            is_ionization_positive=data_cleaning.parse_ion_mode_positive(data["Ion_Mode"]),
            ref_compound=compound_id,
        ).insert()

        for k, v in data.items():
            MeasurementMetadata(ref_measurement=measurement_id, key=k, value=v).insert()

    def get_device_id(self, device: Device) -> int:
        """returns the id of the given device in the database, inserting if necessary"""
        key = device.manufacturer, device.model, device.mass_spec_technique
        if key not in self.devices_cache:
            self.devices_cache[key] = device.ensure_exists()
        return self.devices_cache[key]

    def get_compound_id(self, compound: Compound) -> int:
        """returns the id of the given compound in the database, inserting if necessary"""
        key = compound.smiles
        if key not in self.compounds_cache:
            self.compounds_cache[key] = compound.ensure_exists()
        return self.compounds_cache[key]


@dataclasses.dataclass
class MassBankDataFile:
    device_model: str
    device_technique: str
    is_ionization_positive: bool
    smiles: str | None
    is_smiles_normalized: bool | None
    metadata: dict[str, str]
    peaks: list[list[float]]


def parse_massbank_file(rows: list[str]) -> MassBankDataFile:
    metadata: dict[str, Any] = {}
    device_model = device_technique = is_ionization_positive = is_smiles_normalized = smiles = peaks = None

    i = cnt = 0
    while i < len(rows):
        cnt += 1
        assert cnt < 100000, "infinite loop detected :("

        row = rows[i].rstrip("\n")
        if row == "//":
            break
        rk, rv = row.split(": ", 1)
        if rk == "AC$INSTRUMENT":
            device_model = rv
            i += 1
            continue
        elif rk == "AC$INSTRUMENT_TYPE":
            device_technique = rv
            i += 1
            continue
        elif rk == "AC$MASS_SPECTROMETRY":
            rp1, rp2 = rv.split(maxsplit=1)
            if rp1 == "ION_MODE":
                is_ionization_positive = rp2 == "POSITIVE"
                i += 1
                continue
        elif rk == "CH$SMILES":
            is_smiles_normalized, smiles = data_cleaning.normalize_smiles(rv)
            i += 1
            continue
        elif rk == "PK$ANNOTATION":
            header = rv.split(maxsplit=1)
            peak_annotations = []
            j = i + 1
            while j < len(rows) and rows[j][0] == " ":
                peak_annotations.append(dict(zip(header, rows[j].strip().split())))
                j += 1
            i = j
            metadata["peak_annotations"] = json.dumps(peak_annotations)
            continue
        elif rk == "PK$PEAK":
            header = rv.split()
            assert header[0] == "m/z" and header[2] == "rel.int."
            peaks = []
            j = i + 1
            while j < len(rows) and rows[j][0] == " ":
                ps = rows[j].split()
                peaks.append([float(ps[0]), float(ps[2])])
                j += 1
            i = j
            continue
        elif rk == "MS$FOCUSED_ION":
            rp1, rp2 = rv.split(maxsplit=1)
            if rp1 == "PRECURSOR_M/Z":
                metadata["Precursor_MZ"] = rp2
                i += 1
                continue
            elif rp1 == "PRECURSOR_TYPE":
                metadata["Adduct"] = rp2
                i += 1
                continue

        if rk not in metadata:
            metadata[rk] = []
        metadata[rk].append(rv)
        i += 1

    for k in metadata:
        v = metadata[k]
        if isinstance(v, list):
            if len(v) == 1:
                metadata[k] = v[0]
            else:
                metadata[k] = json.dumps(v)

    assert (
        device_model is not None
        and device_technique is not None
        and is_ionization_positive is not None
        and peaks is not None
    )

    return MassBankDataFile(
        device_model,
        device_technique,
        is_ionization_positive,
        smiles,
        is_smiles_normalized,
        metadata,
        peaks,
    )


if __name__ == "__main__":
    main()
