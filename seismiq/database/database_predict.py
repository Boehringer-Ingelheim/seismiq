import abc
import itertools
import json
import traceback
from collections.abc import Iterator
from typing import cast

import click
from pydal import DAL
from tqdm import tqdm

from seismiq.database.database import (
    Device,
    Measurement,
    MeasurementMetadata,
    get_db,
)
from seismiq.utils.parallel_utils import parallel_threads
from seismiq.vendor.cfm_id import CfmPrediction, cfm_predict
from seismiq.vendor.frag_genie import FragGeniePredictor


@click.command()
@click.option("-m", "--n-jobs", default=64, help="How many jobs to run in parallel")
@click.option(
    "-s",
    "--job-size",
    default=16,
    help="How many compounds to predict in each job",
)
@click.option(
    "-n",
    "--dry-run",
    is_flag=True,
    help="Run predictions but do not insert items in the database.",
)
def main(n_jobs: int, job_size: int, dry_run: bool) -> None:
    assert n_jobs >= 1
    assert job_size >= 1

    db = get_db()
    predictors = [
        DatabaseFragGeniePredictor(FragGeniePredictor(1, 40)),
        DatabaseFragGeniePredictor(FragGeniePredictor(2, 40)),
        DatabaseFragGeniePredictor(FragGeniePredictor(3, 40)),
        CfmIdPredictor(apply_postprocessing=True, positive_ionization=True),
    ]

    all_predictions: Iterator[list[tuple[Measurement, dict[str, str]]]] = iter([])
    for pred in predictors:
        smiles = pred.find_compounds_without_predictions(db)
        print(f"Predictor {pred.measurement_origin} will predict spectra for {len(smiles)} compounds")

        all_predictions = itertools.chain(
            all_predictions,
            parallel_threads(
                smiles,
                pred.predict_spectra,
                job_size,
                n_jobs // len(predictors),
                progress_bar=False,
            ),
        )

    bar = tqdm(all_predictions, ncols=0)
    processed = 0
    for preds in bar:
        if not dry_run:
            for meas, meta in preds:
                mid = meas.insert()
                for k, v in meta.items():
                    MeasurementMetadata(k, v, mid).insert()
        processed += len(preds)
        bar.set_description(f"Saved {processed} spectra")
        db.commit()


class FragmentationPredictor(abc.ABC):
    @abc.abstractmethod
    def find_compounds_without_predictions(self, db: DAL) -> list[tuple[int, str]]:
        """Finds all compounds in the database that do not have any predictions from this method.

        Args:
            db (DAL): The database.

        Returns:
            list[tuple[int, str]]: A list of compound IDs and their SMILES.
        """

    @abc.abstractmethod
    def predict_spectra(self, smiles: list[tuple[int, str]]) -> list[tuple[Measurement, dict[str, str]]]:
        """Predicts one or more spectra for the given SMILES molecules.

        Args:
            smiles (list[tuple[int, str]]): A list with compound ID and SMILES string
            for each compound.

        Returns:
            list[tuple[Measurement, dict[str, str]]]: The predictions for all compounds.
            Each prediction is a Measurement object and a dictionary of metadata items.
        """

    @property
    @abc.abstractmethod
    def measurement_origin(self) -> str:
        """
        Returns the origin of this measurement to distinguish different predictors
        in the database

        Returns:
            str: Origin of this measurement
        """


class CfmIdPredictor(FragmentationPredictor):
    def __init__(
        self,
        apply_postprocessing: bool = True,
        positive_ionization: bool = True,
    ):
        self.apply_postprocessing = apply_postprocessing
        self.positive_ionization = positive_ionization
        self.device_id = Device(
            manufacturer="CFM-ID",
            model="4.4.7",
            mass_spec_technique="ESI-MS/MS",
        ).ensure_exists()

    def __repr__(self) -> str:
        return "/".join(
            [
                "CFM-ID/4.4.7",
                "postprocessing:%s" % ("yes" if self.apply_postprocessing else "no"),
                "ionization:%s" % ("+" if self.positive_ionization else "-"),
            ]
        )

    @property
    def measurement_origin(self) -> str:
        og = "cfm_id"
        if not self.apply_postprocessing:
            og += "_no_postprocessing"
        if not self.positive_ionization:
            og += "_negion"
        return og

    def find_compounds_without_predictions(self, db: DAL) -> list[tuple[int, str]]:
        return find_compounds_without_measurements_from_origin(db, self.measurement_origin)

    def predict_spectra(self, compounds: list[tuple[int, str]]) -> list[tuple[Measurement, dict[str, str]]]:
        res: list[tuple[Measurement, dict[str, str]]] = []
        for compound_id, smiles in compounds:
            status, predictions = cfm_predict(
                smiles,
                ionization="+" if self.positive_ionization else "-",
                apply_postproc=self.apply_postprocessing,
            )
            if status.returncode == 0 and "NullId" in predictions:
                try:
                    parsed = self._parse_predictions(predictions["NullId"], compound_id)
                except:
                    traceback.print_exc()
                else:
                    res.extend(parsed)
        return res

    def _parse_predictions(
        self, preds: CfmPrediction, compound_id: int
    ) -> Iterator[tuple[Measurement, dict[str, str]]]:
        # if these fail, just change the device_id to something appropriate
        assert preds.spectra_type == "ESI-MS/MS"
        assert preds.cfm_version == "4.4.7"

        for energy, spectrum in preds.peaks.items():
            meas = Measurement(
                origin=self.measurement_origin,
                energy=(10 if energy == "energy0" else 20 if energy == "energy1" else 40),
                is_ionization_positive=(preds.spectra_ionization == "[M+H]+"),
                is_experimental=False,
                peaks_json=json.dumps([[p.mass, p.abundance] for p in spectrum]),
                ref_compound=compound_id,
                ref_device=self.device_id,
            )

            frags = {f.fragment_id: f for f in preds.fragments}
            metadata = {
                "spectra_ionization": str(preds.spectra_ionization),
                "spectra_type": str(preds.spectra_type),
                "precursor_mass": str(preds.precursor_mass),
                "mol_formula": preds.mol_formula,
                "mol_smiles": preds.mol_smiles,
                "mol_inchikey": preds.mol_inchikey,
                "peaks_fragments_json": json.dumps(
                    [
                        [  # for each peak in the predicted spectrum
                            {  # dump all possible fragments
                                "fragment_mass": frags[fid].mass,
                                "fragment_smiles": frags[fid].smiles,
                                "fragment_logit": lit,
                            }
                            for (fid, lit) in zip(p.fragment_ids, p.fragment_logits)
                        ]
                        for p in spectrum
                    ]
                ),
            }

            yield meas, metadata


class DatabaseFragGeniePredictor(FragmentationPredictor):
    def __init__(self, fg: FragGeniePredictor):
        self.device_id = Device(
            manufacturer="FragGenie",
            model="9d62a3",  # git commit hash
            mass_spec_technique="unknown",  # TODO
        ).ensure_exists()
        self.fg = fg

    def __repr__(self) -> str:
        return "FragGenie"

    @property
    def measurement_origin(self) -> str:
        if self.fg.frag_recursion_depth == 3:
            return "FragGenie"
        else:
            return f"FragGenie-{self.fg.frag_recursion_depth}"

    def find_compounds_without_predictions(self, db: DAL) -> list[tuple[int, str]]:
        return find_compounds_without_measurements_from_origin(db, self.measurement_origin)

    def predict_spectra(self, compound_smiles: list[tuple[int, str]]) -> list[tuple[Measurement, dict[str, str]]]:
        smiles = []
        smiles_to_cid: dict[str, int] = {}
        for cid, sm in compound_smiles:
            smiles_to_cid[sm] = cid
            smiles.append(sm)

        preds = self.fg.predict_spectra(smiles)

        res = []
        for pr in preds:
            meas = Measurement(
                origin=self.measurement_origin,
                energy=-self.fg.frag_recursion_depth,  # "special encoding"
                is_ionization_positive=True,
                is_experimental=False,
                peaks_json=json.dumps(pr.peaks),
                ref_compound=smiles_to_cid[pr.smiles],
                ref_device=self.device_id,
            )

            metadata = {
                # according to their source code, MetFragFragmenter.java line 298
                "spectra_ionization": "[M]+H+",
                "peaks_formulae": json.dumps(pr.formulas),
            }

            res.append((meas, metadata))

        return res


def find_compounds_without_measurements_from_origin(db: DAL, origin: str) -> list[tuple[int, str]]:
    return cast(
        list[tuple[int, str]],
        db.executesql(
            """
        SELECT c.id, c.smiles
        FROM compounds c
            LEFT OUTER JOIN (
                SELECT DISTINCT ref_compound
                FROM measurements
                WHERE origin == ?
            ) m
            ON c.id = m.ref_compound
        WHERE m.ref_compound IS NULL AND c.is_smiles_normalized = 'T'
    """,
            [origin],
        ),
    )


if __name__ == "__main__":
    main()  # type: ignore
