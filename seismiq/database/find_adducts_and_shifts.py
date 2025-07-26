import collections
import dataclasses
import json
import sqlite3
import typing
from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import Any, TypeVar, cast

from rdkit import Chem
from tqdm import tqdm


@dataclasses.dataclass
class MeasurementAdductShift:
    ref_measurement: int
    quality_control_status: str
    adduct: str | None
    precursor_mass: float | None
    orig_compound_monoiso_mass: float | None
    adduct_theoretical_shift: float | None
    observed_shift: float | None


T = TypeVar("T")


class DataclassSqliteWrapper(typing.Generic[T]):
    def __init__(self, cls: type[T], conn: sqlite3.Connection) -> None:
        self.typ = cls
        self.table_name, self.table_field_types = self.db_table_definition_for_dataclass(cls)
        self.field_order = list(self.table_field_types.keys())
        self.conn = conn

    def insert_many(self, objs: list[T]) -> None:
        if not objs:
            return

        row_data = [dataclasses.asdict(x) for x in objs]  # type: ignore
        fields = ", ".join(self.field_order)
        values = ", ".join(f":{k}" for k in self.field_order)

        cur = self.conn.cursor()
        query = f"INSERT INTO {self.table_name}({fields}) VALUES({values})"
        cur.executemany(query, row_data)
        cur.close()

    def drop_table(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        cursor.close()

    def create_table(self) -> None:
        create_query = [f"CREATE TABLE IF NOT EXISTS {self.table_name}("]
        for k, v in self.table_field_types.items():
            create_query.append(f"{k} {v}, ")
        create_query[-1] = create_query[-1][:-2]
        create_query.append(")")

        cursor = self.conn.cursor()
        cursor.execute("".join(create_query))
        cursor.close()

    @staticmethod
    def db_table_definition_for_dataclass(
        cls: type[T],
        table_name: str | None = None,
        field_types: dict[str, str] | None = None,
    ) -> tuple[str, dict[str, str]]:
        fields = {}
        for f in dataclasses.fields(cls):  # type: ignore
            type_mapping = {str: "TEXT", float: "DOUBLE", bool: "BOOLEAN", int: "INTEGER"}
            if field_types and f.name in field_types:
                # highest priority to user-specified field types
                typ = field_types[f.name]
                notnull = False
            elif f.name.startswith("ref_"):
                # fields named 'ref_x_y' will be translated to foreign keys towards table x
                # assuming x has a primary key field named 'id'
                typ = f"{type_mapping[f.type]} REFERENCES {f.name.split('_')[1]}s(id)"
                typ += " ON DELETE CASCADE ON UPDATE CASCADE"
                notnull = True
            else:
                # guess pyDal's type based on Python's field type, also inferring notnull
                is_optional_type = typing.get_origin(f.type) is typing.Union and type(None) in typing.get_args(f.type)
                if is_optional_type:
                    # find out T for Python's Optional[T] type
                    notnull = False
                    inner = next(arg for arg in typing.get_args(f.type) if arg is not type(None))
                else:
                    notnull = True
                    inner = f.type
                typ = type_mapping[inner]  # extend over time

            fields[f.name] = f"{typ} NOT NULL" if notnull else typ

        if table_name is None:
            # convert CamelCase to snake_case and pluralize
            # e.g., DeviceMetadata -> device_metadatas
            table_name = "".join(("_" + c.lower()) if c.isupper() else c for c in cls.__name__)[1:] + "s"

        return table_name, fields


class Program:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.pt = Chem.GetPeriodicTable()  # type: ignore
        self.adduct_masses = {
            ("M+H", "[M+H]", "[M+H]+", "[M]+H+"): self.pt.GetMostCommonIsotopeMass("H"),
            ("M+Na", "[M+Na]+", "[M+Na]"): self.pt.GetMostCommonIsotopeMass("Na"),
        }

    def run(self) -> None:
        if self.dry_run:
            print("DRY RUN - database will not be changed")

        self.conn = sqlite3.connect("database/database.sqlite")
        self.conn.row_factory = sqlite3.Row  # This allows us to access rows as dictionaries

        try:
            db = DataclassSqliteWrapper(MeasurementAdductShift, self.conn)
            db.drop_table()
            db.create_table()

            batch = []
            for meas in self.process_all_measurements():
                batch.append(meas)
                if len(batch) > 1000:
                    db.insert_many(batch)
                    batch = []
            if batch:
                db.insert_many(batch)

            if self.dry_run:
                self.conn.rollback()
                print("DRY RUN - transaction rolled back")
            else:
                self.conn.commit()
        finally:
            self.conn.close()

    def process_all_measurements(self) -> Iterator[MeasurementAdductShift]:
        counters: dict[str, int] = collections.defaultdict(int)

        all_adducts: dict[str | None, int] = collections.defaultdict(int)
        good_compounds = set()
        all_compounds = set()
        bar = tqdm(enumerate(aggregate_data(get_data(self.conn))))

        rows = []

        for i, meas in bar:
            all_compounds.add(meas["compound_id"])
            proc = postprocess_measurement(meas)
            if proc is None:
                continue

            result = self.process_one_measurement(proc)
            assert result.quality_control_status
            assert result.quality_control_status != "good" or result.adduct in ("M+H", "M+Na")

            yield result

            counters[result.quality_control_status] += 1
            all_adducts[result.adduct] += 1
            good_compounds.add(proc["compound_id"])
            rows.append(result)

            if (i + 1) % 10000 == 0:
                counters["good_com"] = len(good_compounds)
                counters["bad_com"] = len(all_compounds - good_compounds)
                desc = " - ".join(f"[{k[:7]}]: {v}" for k, v in counters.items())
                tqdm.write(desc)

        print("=== processing finished")
        print("final counters")
        desc = "\n - ".join(f"[{k}]: {v}" for k, v in counters.items())
        print(desc)
        print("adduct counts:")
        for k, v in sorted(all_adducts.items(), key=lambda x: -x[1]):
            print(f" - {k} : {v}")

    def process_one_measurement(self, meas: dict[str, Any]) -> MeasurementAdductShift:
        result = MeasurementAdductShift(
            meas["measurement_id"],
            "",
            cast(str, meas.get("adduct")),
            cast(float, meas.get("precursor_mass")),
            None,
            None,
            None,
        )

        if "adduct" not in meas and "precursor_mass" not in meas:
            result.quality_control_status = "no_data"
            return result

        mm = self.get_mol_wt(meas["compound_smiles"])
        if mm is None:
            result.quality_control_status = "bad_smiles"
            return result
        monoiso_mass = result.orig_compound_monoiso_mass = mm[1]

        if "adduct" not in meas:
            result.adduct = meas["adduct"] = self.shift_to_adduct(monoiso_mass - float(meas.get("precursor_mass", 0.0)))

        # clean adduct
        if result.adduct is not None:
            for k in self.adduct_masses:
                if result.adduct in k:
                    result.adduct = k[0]

        adduct_shift = result.adduct_theoretical_shift = self.adduct_to_shift(meas["adduct"])
        if adduct_shift is None:
            result.quality_control_status = "bad_addu"
            return result

        if "precursor_mass" not in meas:
            result.precursor_mass = meas["precursor_mass"] = monoiso_mass + adduct_shift

        # check that precursor/exact mass are consistent with adduct
        pm = float(meas["precursor_mass"])
        result.observed_shift = shift = pm - monoiso_mass
        if abs(shift - adduct_shift) < 0.5:
            max_mass = pm
        else:
            result.quality_control_status = "bad_shift"
            return result

        # check that peaks have reasonable masses
        mass_lo = mass_hi = False
        for m in meas["peaks_json"]:
            if isinstance(m, (tuple, list)):
                m = m[0]

            if m - adduct_shift <= -0.5:
                mass_lo = True
            elif m > max_mass + 0.5:
                # todo should we also put a filter based on intensity?
                # eg https://massbank.eu/MassBank/RecordDisplay?id=MSBNK-Eawag-EQ325001
                # has one peak larger than the precursor but with very low intensity
                # could be noise?
                mass_hi = True

        if mass_lo or mass_hi:
            result.quality_control_status = "bad_mz"
            return result

        result.quality_control_status = "good"
        return result

    def adduct_to_shift(self, adduct: str | None) -> float | None:
        for k, v in self.adduct_masses.items():
            if adduct in k:
                return v
        return None

    def shift_to_adduct(self, shift: float) -> str | None:
        for k, v in self.adduct_masses.items():
            if abs(v - shift) < 0.1:
                return k[0]
        return None

    @lru_cache
    def get_mol_wt(self, sm: str) -> tuple[float, float] | None:
        mol = Chem.MolFromSmiles(sm)  # type: ignore
        if mol is None:
            return None

        avg_mass = monoiso_mass = 0.0
        for a in Chem.AddHs(mol).GetAtoms():  # type: ignore
            avg_mass += self.pt.GetAtomicWeight(a.GetSymbol())
            monoiso_mass += self.pt.GetMostCommonIsotopeMass(a.GetSymbol())

        return avg_mass, monoiso_mass


def get_data(conn: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    query = """
    SELECT
        c.id AS compound_id,
        c.smiles as compound_smiles,
        m.id AS measurement_id,
        m.energy AS measurement_energy,
        m.origin AS measurement_origin,
        m.peaks_json AS measurements_peaks_json,
        mm.key AS measurement_metadata_key,
        mm.value AS measurement_metadata_value
    FROM
        compounds c, measurements m, measurement_metadatas mm
    WHERE
        c.id = m.ref_compound
        AND c.smiles IS NOT NULL
        AND c.is_smiles_normalized = 'T'
        AND c.smiles <> 'N/A'
        AND m.id = mm.ref_measurement
        AND m.origin NOT LIKE 'cfm_id_%'
        AND m.is_ionization_positive = 'T'
    ORDER BY c.id, m.id
    """

    cursor = conn.cursor()
    try:
        cursor.execute(query)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield dict(row)
    finally:
        cursor.close()


def aggregate_data(data: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    # aggregate multiple rows into the same measurement
    # NB - assumes that rows are ordered by measurement id

    finished_measurements: set[int] = set()
    cur = None
    for row in data:
        mid = row["measurement_id"]
        assert mid not in finished_measurements, "sort rows by measurement plz"

        if cur is None or mid != cur["measurement_id"]:
            if cur is not None:
                finished_measurements.add(cur["measurement_id"])
                yield cur

            cur = row
            cur["peaks_json"] = json.loads(row["measurements_peaks_json"])
            cur["metadata"] = {}

        mk, mv = row["measurement_metadata_key"], row["measurement_metadata_value"]
        if row["measurement_metadata_key"] is not None:
            cur["metadata"][mk] = mv

    if cur is not None:
        yield cur


def parse_massbank_meta(mt: str | None) -> dict[str, str]:
    if not mt:
        return {}

    if mt[0] == "[":
        mv = cast(list[str], json.loads(mt))
    else:
        mv = [mt]

    r = {}
    for m in mv:
        k, *vs = m.split()
        r[k] = " ".join(vs)
    return r


def postprocess_measurement(m: dict[str, Any]) -> dict[str, Any] | None:
    # find origin
    if m["measurement_origin"].startswith("cfm_id"):
        m["measurement_origin"] = f'cfm_id-{int(m["measurement_energy"])}'
    elif m["measurement_origin"].startswith("GNPS"):
        m["measurement_origin"] = "GNPS"
    elif m["measurement_origin"].startswith("MSBNK"):
        m["measurement_origin"] = "MSBNK"
    elif m["measurement_origin"].startswith("matchms"):
        m["measurement_origin"] = "matchms"

    # find adduct and precursor mass from the metadata
    if "cfm_id" in m["measurement_origin"]:
        m["adduct"] = m["metadata"]["spectra_ionization"]
        m["precursor_mass"] = float(m["metadata"]["precursor_mass"])
    elif "GNPS" in m["measurement_origin"]:
        m["adduct"] = m["metadata"]["Adduct"]
        m["precursor_mass"] = float(m["metadata"]["Precursor_MZ"])
    elif "FragGenie" in m["measurement_origin"]:
        m["adduct"] = m["metadata"]["spectra_ionization"]
    elif "MSBNK" in m["measurement_origin"]:
        acm = parse_massbank_meta(m["metadata"].get("AC$MASS_SPECTROMETRY"))
        mfi = parse_massbank_meta(m["metadata"].get("MS$FOCUSED_ION"))

        if acm.get("MS_TYPE", "MS2") != "MS2" or acm.get("IONIZATION", "ESI") != "ESI":
            return None

        if "Adduct" in m["metadata"]:
            m["adduct"] = m["metadata"]["Adduct"]
        elif "ION_TYPE" in mfi:
            m["adduct"] = mfi["ION_TYPE"]
        elif "DERIVATIVE_FORM" in mfi:
            m["adduct"] = mfi["DERIVATIVE_FORM"]
        else:
            pass

        pmz = m["metadata"].get("Precursor_MZ")
        if pmz:
            if all(c.isnumeric() or c == "." for c in pmz):
                m["precursor_mass"] = float(m["metadata"]["Precursor_MZ"])
        elif "DERIVATIVE_MASS" in mfi:
            m["precursor_mass"] = float(mfi["DERIVATIVE_MASS"])
        elif "PRECURSOR_M/Z" in mfi:
            m["precursor_mass"] = float(mfi["PRECURSOR_M/Z"])
        else:
            pass
    elif "matchms" in m["measurement_origin"]:
        m["adduct"] = m["metadata"]["name"].split()[-1]
        pepmass = m["metadata"]["pepmass"]
        if pepmass[0] == "(" and pepmass[-1] == ")":
            m["precursor_mass"] = float(pepmass[1:-1].split(",")[0])
        else:
            pass

    return m


if __name__ == "__main__":
    Program().run()


ad = {
    # with H
    "M+H": 229121,
    "[M+H]+": 299326,
    "[M+H]": 12424,
    "M+2H]": 561,
    " M+H": 2,
    "2M+H": 3286,
    "[M+2H]": 198,
    "M+2H": 920,
    "[M+2H]++": 125,
    "[2M+H]+": 192,
    "[M+2H]2+": 18,
    "3M+H": 2,
    "M+H]": 36,
    # with Na
    "[M+Na]+": 226546,
    "M+Na": 30373,
    "[M+Na]": 248,
    "2M+Na": 1756,
    "M+H+Na": 126,
    "[2M+Na]+": 466,
    "M-H+2Na": 363,
    "M+2Na-H": 3,
    "[M+Na]*+": 2,
    "[M-H+Na]*+": 4,
    "M+2Na": 3,
    "[2M+Na]": 88,
    "[M+Na]+*": 1,
    "[M-H+Na]+*": 2,
    "M+2Na]": 27,
    "[M+H+Na]2+": 2,
    "[2M-H+2Na]+": 6,
    "[M-H+2Na]+": 54,
    "[3M+Na]+": 14,
    "M2+Na": 14,
    "3M+Na": 2,
    "M-H+Na": 63,
    # H2O
    "M-H2O+H": 6698,
    "M+H-H2O": 4585,
    "M-2H2O+H": 2379,
    "[M-H2O+H]+": 1029,
    "[M+H-H2O]+": 27,
    "M+H-2H2O": 159,
    "[M-H2O]": 2,
    "M-H2O-H": 19,
    "[M-2H2O+H]+": 235,
    "M+H-H20": 3,
    "M+H-3H2O": 27,
    # NH4
    "M+NH4": 4393,
    "[M+NH4]+": 782,
    "[M+NH4]": 116,
    "[2M+NH4]+": 188,
    "2M+NH4": 16,
    "3M+NH4": 2,
    # K
    "M+K": 1161,
    "[M+K]": 16,
    "[M+K]+": 262,
    "2M+K": 152,
    "[2M+K]": 4,
    "[2M+K]+": 26,
    "[3M+K]+": 2,
    "M+H+K": 13,
    # just charge
    "[M]+": 1017,
    "M": 1226,
    "[M]+*": 379,
    "[M]*+": 96,
    "M-e": 813,
    "M+": 572,
    "M++": 15,
    # !!!
    None: 20772,
    "Unknown": 852,
    # Ca
    "[2M+Ca-H]+": 6,
    "[2M+Ca]2+": 66,
    "[3M+Ca]2+": 14,
    "[M+Ca]2+": 140,
    "[4M+Ca]2+": 10,
    "[5M+Ca]2+": 4,
    "[3M+Ca-H]+": 2,
    "[M+Ca-H]+": 22,
    # below 1000
    "M-H": 144,
    "[M+Li]*+": 2,
    "[M-H+Li]*+": 4,
    "[M+15]+": 12,
    "[M+H-(C12H20O9)]+": 4,
    "M+H-NH3": 3,
    "M-C6H10O5+H": 3,
    "M-2H": 2,
    "M+H-C9H10O5": 3,
    "M+H-99": 3,
    "?": 2,
    "M+Cl": 22,
    "M-3H2O+H": 12,
    "M+CH3OH+H": 16,
    "M-H2O": 15,
    "M-HAc": 3,
    "M2+H": 4,
    "M-C3H8O+H": 3,
    "M-H20+H": 3,
    "M-H2+H": 3,
    "M+2H+2": 3,
    "M+NH3": 12,
    "M+2": 3,
    "M+ACN+H": 162,
    "[2M-H2O+H]+": 12,
    "[3M+NH4]+": 2,
    "[2M-2H2O+H]+": 10,
    "[M+ACN+NH4]+": 4,
    "[M-3H2O+H]+": 30,
    "[M-3H2O+2H]2+": 2,
    "[M-2H2O+2H]2+": 2,
    "[M-4H2O+H]+": 8,
    "[M-5H2O+H]+": 4,
    "[M+ACN+H]+": 2,
    "[2M-3H2O+H]+": 2,
    "[M-2H2O+NH4]+": 2,
    "[M+Na+CH3CN]": 20,
    "[2M+H]": 34,
    "[2M+NH4]": 78,
    "[M+H+CH3OH]": 62,
    "[M+H+C2H6OS]": 2,
    "[M+H+CH3CN]": 16,
    "[M+H+HCOOH]": 10,
    "[2M+H+CH3CN]": 2,
    "M-2(H2O)+H": 6,
    "M+DMSO+H": 4,
    "[M+HFA+H]+": 2,
    "[M+HFA+NH4]+": 2,
    "[M-H2O+NH4]+": 2,
    "[2M-H2O+Na]+": 2,
    "[3M+H]+": 2,
    "[M-H]-": 110,
    "[M]-": 28,
    "[M-OH]+": 9,
    "[M-H]+": 1,
    "[M+Li]+*": 3,
    "[M-H+Li]+*": 2,
    "[M+H-NH3]+": 31,
    "[M+H+O]+": 2,
    "[M+H-C8H10O]+": 1,
    "[M+H-C9H10O5]+": 1,
    "[M-C6H10O5+H]+": 1,
    "[M+H-C12H20O9]+": 2,
    "[M]++": 5,
    "[M+CH3]+": 6,
    "M+3H": 2,
    "M+23": 1,
}
