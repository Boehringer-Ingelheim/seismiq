import dataclasses
import typing
from collections.abc import Iterator
from typing import ClassVar, TypeVar, cast

import pydal
from pydal import DAL

from seismiq.utils import config

T = TypeVar("T", bound="PyDalDataClassMixin")


@dataclasses.dataclass
class PyDalDataClassMixin:
    _table: ClassVar[pydal.objects.Table]
    _fields: ClassVar[dict[str, pydal.objects.Field]]

    @classmethod
    def define_table(
        cls: type[T],
        db: DAL,
        table_name: str | None = None,
        field_types: dict[str, str] | None = None,
    ) -> None:
        cls._fields = {}
        for f in dataclasses.fields(cls):
            if f.name == "id":
                # id fields are automatically added by pyDAL
                continue

            type_mapping = {
                str: "text",
                float: "double",
                bool: "boolean",
                int: "integer",
            }
            if field_types and f.name in field_types:
                # highest priority to user-specified field types
                typ = field_types[f.name]
                notnull = False
            elif f.name.startswith("ref_"):
                # fields named 'ref_x_y' will be translated to foreign keys towards table x
                typ = "reference " + f.name.split("_")[1] + "s"
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

            cls._fields[f.name] = pydal.Field(f.name, typ, notnull=notnull)

        if table_name is None:
            # convert CamelCase to snake_case and pluralize
            # e.g., DeviceMetadata -> device_metadatas
            table_name = "".join(("_" + c.lower()) if c.isupper() else c for c in cls.__name__)[1:] + "s"

        cls._table = cast(pydal.objects.Table, db.define_table(table_name, *cls._fields.values()))
        cls._fields["id"] = cls._table.id  # type: ignore
        cls.create_indexes(db, table_name)

    @classmethod
    def create_indexes(cls: type[T], db: DAL, table: str) -> None:
        for f in dataclasses.fields(cls):
            if f.name.startswith("ref_"):
                # FIXME the index should not be on the foreign key but on the primary key it references
                db.executesql(f"CREATE INDEX IF NOT EXISTS {table}_{f.name} ON {table}({f.name});")

    @classmethod
    def from_select(cls: type[T], row: pydal.objects.Row) -> Iterator[T]:
        yield cls(**{k: row[v] for k, v in cls._fields.items()})

    def select(self) -> pydal.objects.Row | None:
        if not hasattr(self, "_table"):
            raise RuntimeError("call define_table before use")
        cols = {k: v for k, v in dataclasses.asdict(self).items() if k != "id" or v is not None}
        return self._table(**cols)  # type: ignore

    def insert(self) -> int:
        if not hasattr(self, "_table"):
            raise RuntimeError("call define_table before use")
        return self._table.insert(**dataclasses.asdict(self))

    def ensure_exists(self) -> int:
        row = self.select()
        return cast(int, row.id if row is not None else self.insert())


@dataclasses.dataclass
class Compound(PyDalDataClassMixin):
    smiles: str | None
    is_smiles_normalized: bool | None
    id: int | None = None


@dataclasses.dataclass
class Device(PyDalDataClassMixin):
    manufacturer: str
    model: str
    mass_spec_technique: str
    id: int | None = None


@dataclasses.dataclass
class Measurement(PyDalDataClassMixin):
    origin: str
    energy: float | None
    is_ionization_positive: bool | None
    is_experimental: bool
    peaks_json: str
    ref_compound: int  # reference compounds
    ref_device: int  # reference devices
    id: str | None = None


@dataclasses.dataclass
class MeasurementMetadata(PyDalDataClassMixin):
    key: str
    value: str
    ref_measurement: int  # rerence measurement
    id: int | None = None


@dataclasses.dataclass
class DeviceMetadata(PyDalDataClassMixin):
    key: str
    value: str
    ref_device: int  # reference device
    id: int | None = None


def get_db() -> pydal.DAL:
    db = DAL("sqlite://database.sqlite", folder=config.SEISMIQ_DATABASE_FOLDER())

    Compound.define_table(db)
    Device.define_table(db)
    Measurement.define_table(db)
    MeasurementMetadata.define_table(db)
    DeviceMetadata.define_table(db)

    return db
