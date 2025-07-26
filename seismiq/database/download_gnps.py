import json
import os

from seismiq.utils import config

GNPS_DATA = [
    {
        "jsonlink": "/gnpslibrary/GNPS-LIBRARY.json",
        "library": "GNPS-LIBRARY",
        "libraryname": "GNPS-LIBRARY",
        "mgflink": "/gnpslibrary/GNPS-LIBRARY.mgf",
        "msplink": "/gnpslibrary/GNPS-LIBRARY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-SELLECKCHEM-FDA-PART1.json",
        "library": "GNPS-SELLECKCHEM-FDA-PART1",
        "libraryname": "GNPS-SELLECKCHEM-FDA-PART1",
        "mgflink": "/gnpslibrary/GNPS-SELLECKCHEM-FDA-PART1.mgf",
        "msplink": "/gnpslibrary/GNPS-SELLECKCHEM-FDA-PART1.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-SELLECKCHEM-FDA-PART2.json",
        "library": "GNPS-SELLECKCHEM-FDA-PART2",
        "libraryname": "GNPS-SELLECKCHEM-FDA-PART2",
        "mgflink": "/gnpslibrary/GNPS-SELLECKCHEM-FDA-PART2.mgf",
        "msplink": "/gnpslibrary/GNPS-SELLECKCHEM-FDA-PART2.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-PRESTWICKPHYTOCHEM.json",
        "library": "GNPS-PRESTWICKPHYTOCHEM",
        "libraryname": "GNPS-PRESTWICKPHYTOCHEM",
        "mgflink": "/gnpslibrary/GNPS-PRESTWICKPHYTOCHEM.mgf",
        "msplink": "/gnpslibrary/GNPS-PRESTWICKPHYTOCHEM.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-NIH-CLINICALCOLLECTION1.json",
        "library": "GNPS-NIH-CLINICALCOLLECTION1",
        "libraryname": "GNPS-NIH-CLINICALCOLLECTION1",
        "mgflink": "/gnpslibrary/GNPS-NIH-CLINICALCOLLECTION1.mgf",
        "msplink": "/gnpslibrary/GNPS-NIH-CLINICALCOLLECTION1.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-NIH-CLINICALCOLLECTION2.json",
        "library": "GNPS-NIH-CLINICALCOLLECTION2",
        "libraryname": "GNPS-NIH-CLINICALCOLLECTION2",
        "mgflink": "/gnpslibrary/GNPS-NIH-CLINICALCOLLECTION2.mgf",
        "msplink": "/gnpslibrary/GNPS-NIH-CLINICALCOLLECTION2.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-NIH-NATURALPRODUCTSLIBRARY.json",
        "library": "GNPS-NIH-NATURALPRODUCTSLIBRARY",
        "libraryname": "GNPS-NIH-NATURALPRODUCTSLIBRARY",
        "mgflink": "/gnpslibrary/GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf",
        "msplink": "/gnpslibrary/GNPS-NIH-NATURALPRODUCTSLIBRARY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE.json",
        "library": "GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE",
        "libraryname": "GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE",
        "mgflink": "/gnpslibrary/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE.mgf",
        "msplink": "/gnpslibrary/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_NEGATIVE.json",
        "library": "GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_NEGATIVE",
        "libraryname": "GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_NEGATIVE",
        "mgflink": "/gnpslibrary/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_NEGATIVE.mgf",
        "msplink": "/gnpslibrary/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_NEGATIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-NIH-SMALLMOLECULEPHARMACOLOGICALLYACTIVE.json",
        "library": "GNPS-NIH-SMALLMOLECULEPHARMACOLOGICALLYACTIVE",
        "libraryname": "GNPS-NIH-SMALLMOLECULEPHARMACOLOGICALLYACTIVE",
        "mgflink": "/gnpslibrary/GNPS-NIH-SMALLMOLECULEPHARMACOLOGICALLYACTIVE.mgf",
        "msplink": "/gnpslibrary/GNPS-NIH-SMALLMOLECULEPHARMACOLOGICALLYACTIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-FAULKNERLEGACY.json",
        "library": "GNPS-FAULKNERLEGACY",
        "libraryname": "GNPS-FAULKNERLEGACY",
        "mgflink": "/gnpslibrary/GNPS-FAULKNERLEGACY.mgf",
        "msplink": "/gnpslibrary/GNPS-FAULKNERLEGACY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-EMBL-MCF.json",
        "library": "GNPS-EMBL-MCF",
        "libraryname": "GNPS-EMBL-MCF",
        "mgflink": "/gnpslibrary/GNPS-EMBL-MCF.mgf",
        "msplink": "/gnpslibrary/GNPS-EMBL-MCF.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-COLLECTIONS-PESTICIDES-POSITIVE.json",
        "library": "GNPS-COLLECTIONS-PESTICIDES-POSITIVE",
        "libraryname": "GNPS-COLLECTIONS-PESTICIDES-POSITIVE",
        "mgflink": "/gnpslibrary/GNPS-COLLECTIONS-PESTICIDES-POSITIVE.mgf",
        "msplink": "/gnpslibrary/GNPS-COLLECTIONS-PESTICIDES-POSITIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-COLLECTIONS-PESTICIDES-NEGATIVE.json",
        "library": "GNPS-COLLECTIONS-PESTICIDES-NEGATIVE",
        "libraryname": "GNPS-COLLECTIONS-PESTICIDES-NEGATIVE",
        "mgflink": "/gnpslibrary/GNPS-COLLECTIONS-PESTICIDES-NEGATIVE.mgf",
        "msplink": "/gnpslibrary/GNPS-COLLECTIONS-PESTICIDES-NEGATIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/MMV_POSITIVE.json",
        "library": "MMV_POSITIVE",
        "libraryname": "MMV_POSITIVE",
        "mgflink": "/gnpslibrary/MMV_POSITIVE.mgf",
        "msplink": "/gnpslibrary/MMV_POSITIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/MMV_NEGATIVE.json",
        "library": "MMV_NEGATIVE",
        "libraryname": "MMV_NEGATIVE",
        "mgflink": "/gnpslibrary/MMV_NEGATIVE.mgf",
        "msplink": "/gnpslibrary/MMV_NEGATIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/LDB_POSITIVE.json",
        "library": "LDB_POSITIVE",
        "libraryname": "LDB_POSITIVE",
        "mgflink": "/gnpslibrary/LDB_POSITIVE.mgf",
        "msplink": "/gnpslibrary/LDB_POSITIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/LDB_NEGATIVE.json",
        "library": "LDB_NEGATIVE",
        "libraryname": "LDB_NEGATIVE",
        "mgflink": "/gnpslibrary/LDB_NEGATIVE.mgf",
        "msplink": "/gnpslibrary/LDB_NEGATIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-NIST14-MATCHES.json",
        "library": "GNPS-NIST14-MATCHES",
        "libraryname": "GNPS-NIST14-MATCHES",
        "mgflink": "/gnpslibrary/GNPS-NIST14-MATCHES.mgf",
        "msplink": "/gnpslibrary/GNPS-NIST14-MATCHES.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-COLLECTIONS-MISC.json",
        "library": "GNPS-COLLECTIONS-MISC",
        "libraryname": "GNPS-COLLECTIONS-MISC",
        "mgflink": "/gnpslibrary/GNPS-COLLECTIONS-MISC.mgf",
        "msplink": "/gnpslibrary/GNPS-COLLECTIONS-MISC.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-MSMLS.json",
        "library": "GNPS-MSMLS",
        "libraryname": "GNPS-MSMLS",
        "mgflink": "/gnpslibrary/GNPS-MSMLS.mgf",
        "msplink": "/gnpslibrary/GNPS-MSMLS.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/PSU-MSMLS.json",
        "library": "PSU-MSMLS",
        "libraryname": "PSU-MSMLS",
        "mgflink": "/gnpslibrary/PSU-MSMLS.mgf",
        "msplink": "/gnpslibrary/PSU-MSMLS.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/BILELIB19.json",
        "library": "BILELIB19",
        "libraryname": "BILELIB19",
        "mgflink": "/gnpslibrary/BILELIB19.mgf",
        "msplink": "/gnpslibrary/BILELIB19.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/DEREPLICATOR_IDENTIFIED_LIBRARY.json",
        "library": "DEREPLICATOR_IDENTIFIED_LIBRARY",
        "libraryname": "DEREPLICATOR_IDENTIFIED_LIBRARY",
        "mgflink": "/gnpslibrary/DEREPLICATOR_IDENTIFIED_LIBRARY.mgf",
        "msplink": "/gnpslibrary/DEREPLICATOR_IDENTIFIED_LIBRARY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/PNNL-LIPIDS-POSITIVE.json",
        "library": "PNNL-LIPIDS-POSITIVE",
        "libraryname": "PNNL-LIPIDS-POSITIVE",
        "mgflink": "/gnpslibrary/PNNL-LIPIDS-POSITIVE.mgf",
        "msplink": "/gnpslibrary/PNNL-LIPIDS-POSITIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/PNNL-LIPIDS-NEGATIVE.json",
        "library": "PNNL-LIPIDS-NEGATIVE",
        "libraryname": "PNNL-LIPIDS-NEGATIVE",
        "mgflink": "/gnpslibrary/PNNL-LIPIDS-NEGATIVE.mgf",
        "msplink": "/gnpslibrary/PNNL-LIPIDS-NEGATIVE.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/MIADB.json",
        "library": "MIADB",
        "libraryname": "MIADB",
        "mgflink": "/gnpslibrary/MIADB.mgf",
        "msplink": "/gnpslibrary/MIADB.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/HCE-CELL-LYSATE-LIPIDS.json",
        "library": "HCE-CELL-LYSATE-LIPIDS",
        "libraryname": "HCE-CELL-LYSATE-LIPIDS",
        "mgflink": "/gnpslibrary/HCE-CELL-LYSATE-LIPIDS.mgf",
        "msplink": "/gnpslibrary/HCE-CELL-LYSATE-LIPIDS.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/UM-NPDC.json",
        "library": "UM-NPDC",
        "libraryname": "UM-NPDC",
        "mgflink": "/gnpslibrary/UM-NPDC.mgf",
        "msplink": "/gnpslibrary/UM-NPDC.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-NUTRI-METAB-FEM-POS.json",
        "library": "GNPS-NUTRI-METAB-FEM-POS",
        "libraryname": "GNPS-NUTRI-METAB-FEM-POS",
        "mgflink": "/gnpslibrary/GNPS-NUTRI-METAB-FEM-POS.mgf",
        "msplink": "/gnpslibrary/GNPS-NUTRI-METAB-FEM-POS.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-NUTRI-METAB-FEM-NEG.json",
        "library": "GNPS-NUTRI-METAB-FEM-NEG",
        "libraryname": "GNPS-NUTRI-METAB-FEM-NEG",
        "mgflink": "/gnpslibrary/GNPS-NUTRI-METAB-FEM-NEG.mgf",
        "msplink": "/gnpslibrary/GNPS-NUTRI-METAB-FEM-NEG.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-SCIEX-LIBRARY.json",
        "library": "GNPS-SCIEX-LIBRARY",
        "libraryname": "GNPS-SCIEX-LIBRARY",
        "mgflink": "/gnpslibrary/GNPS-SCIEX-LIBRARY.mgf",
        "msplink": "/gnpslibrary/GNPS-SCIEX-LIBRARY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-IOBA-NHC.json",
        "library": "GNPS-IOBA-NHC",
        "libraryname": "GNPS-IOBA-NHC",
        "mgflink": "/gnpslibrary/GNPS-IOBA-NHC.mgf",
        "msplink": "/gnpslibrary/GNPS-IOBA-NHC.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/BERKELEY-LAB.json",
        "library": "BERKELEY-LAB",
        "libraryname": "BERKELEY-LAB",
        "mgflink": "/gnpslibrary/BERKELEY-LAB.mgf",
        "msplink": "/gnpslibrary/BERKELEY-LAB.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/IQAMDB.json",
        "library": "IQAMDB",
        "libraryname": "IQAMDB",
        "mgflink": "/gnpslibrary/IQAMDB.mgf",
        "msplink": "/gnpslibrary/IQAMDB.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-SAM-SIK-KANG-LEGACY-LIBRARY.json",
        "library": "GNPS-SAM-SIK-KANG-LEGACY-LIBRARY",
        "libraryname": "GNPS-SAM-SIK-KANG-LEGACY-LIBRARY",
        "mgflink": "/gnpslibrary/GNPS-SAM-SIK-KANG-LEGACY-LIBRARY.mgf",
        "msplink": "/gnpslibrary/GNPS-SAM-SIK-KANG-LEGACY-LIBRARY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-D2-AMINO-LIPID-LIBRARY.json",
        "library": "GNPS-D2-AMINO-LIPID-LIBRARY",
        "libraryname": "GNPS-D2-AMINO-LIPID-LIBRARY",
        "mgflink": "/gnpslibrary/GNPS-D2-AMINO-LIPID-LIBRARY.mgf",
        "msplink": "/gnpslibrary/GNPS-D2-AMINO-LIPID-LIBRARY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/DRUGS-OF-ABUSE-LIBRARY.json",
        "library": "DRUGS-OF-ABUSE-LIBRARY",
        "libraryname": "DRUGS-OF-ABUSE-LIBRARY",
        "mgflink": "/gnpslibrary/DRUGS-OF-ABUSE-LIBRARY.mgf",
        "msplink": "/gnpslibrary/DRUGS-OF-ABUSE-LIBRARY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/ECG-ACYL-AMIDES-C4-C24-LIBRARY.json",
        "library": "ECG-ACYL-AMIDES-C4-C24-LIBRARY",
        "libraryname": "ECG-ACYL-AMIDES-C4-C24-LIBRARY",
        "mgflink": "/gnpslibrary/ECG-ACYL-AMIDES-C4-C24-LIBRARY.mgf",
        "msplink": "/gnpslibrary/ECG-ACYL-AMIDES-C4-C24-LIBRARY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/ECG-ACYL-ESTERS-C4-C24-LIBRARY.json",
        "library": "ECG-ACYL-ESTERS-C4-C24-LIBRARY",
        "libraryname": "ECG-ACYL-ESTERS-C4-C24-LIBRARY",
        "mgflink": "/gnpslibrary/ECG-ACYL-ESTERS-C4-C24-LIBRARY.mgf",
        "msplink": "/gnpslibrary/ECG-ACYL-ESTERS-C4-C24-LIBRARY.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/LEAFBOT.json",
        "library": "LEAFBOT",
        "libraryname": "LEAFBOT",
        "mgflink": "/gnpslibrary/LEAFBOT.mgf",
        "msplink": "/gnpslibrary/LEAFBOT.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/XANTHONES-DB.json",
        "library": "XANTHONES-DB",
        "libraryname": "XANTHONES-DB",
        "mgflink": "/gnpslibrary/XANTHONES-DB.mgf",
        "msplink": "/gnpslibrary/XANTHONES-DB.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/TUEBINGEN-NATURAL-PRODUCT-COLLECTION.json",
        "library": "TUEBINGEN-NATURAL-PRODUCT-COLLECTION",
        "libraryname": "TUEBINGEN-NATURAL-PRODUCT-COLLECTION",
        "mgflink": "/gnpslibrary/TUEBINGEN-NATURAL-PRODUCT-COLLECTION.mgf",
        "msplink": "/gnpslibrary/TUEBINGEN-NATURAL-PRODUCT-COLLECTION.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/NEO-MSMS.json",
        "library": "NEO-MSMS",
        "libraryname": "NEO-MSMS",
        "mgflink": "/gnpslibrary/NEO-MSMS.mgf",
        "msplink": "/gnpslibrary/NEO-MSMS.msp",
        "type": "GNPS",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-IIMN-PROPOGATED.json",
        "library": "GNPS-IIMN-PROPOGATED",
        "libraryname": "GNPS-IIMN-PROPOGATED",
        "mgflink": "/gnpslibrary/GNPS-IIMN-PROPOGATED.mgf",
        "msplink": "/gnpslibrary/GNPS-IIMN-PROPOGATED.msp",
        "type": "GNPS-PROPOGATED",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-SUSPECTLIST.json",
        "library": "GNPS-SUSPECTLIST",
        "libraryname": "GNPS-SUSPECTLIST",
        "mgflink": "/gnpslibrary/GNPS-SUSPECTLIST.mgf",
        "msplink": "/gnpslibrary/GNPS-SUSPECTLIST.msp",
        "type": "GNPS-PROPOGATED",
    },
    {
        "jsonlink": "/gnpslibrary/GNPS-BILE-ACID-MODIFICATIONS.json",
        "library": "GNPS-BILE-ACID-MODIFICATIONS",
        "libraryname": "GNPS-BILE-ACID-MODIFICATIONS",
        "mgflink": "/gnpslibrary/GNPS-BILE-ACID-MODIFICATIONS.mgf",
        "msplink": "/gnpslibrary/GNPS-BILE-ACID-MODIFICATIONS.msp",
        "type": "GNPS-PROPOGATED",
    },
    {
        "jsonlink": "/gnpslibrary/BMDMS-NP.json",
        "library": "BMDMS-NP",
        "libraryname": "BMDMS-NP",
        "mgflink": "/gnpslibrary/BMDMS-NP.mgf",
        "msplink": "/gnpslibrary/BMDMS-NP.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/MASSBANK.json",
        "library": "MASSBANK",
        "libraryname": "MASSBANK",
        "mgflink": "/gnpslibrary/MASSBANK.mgf",
        "msplink": "/gnpslibrary/MASSBANK.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/MASSBANKEU.json",
        "library": "MASSBANKEU",
        "libraryname": "MASSBANKEU",
        "mgflink": "/gnpslibrary/MASSBANKEU.mgf",
        "msplink": "/gnpslibrary/MASSBANKEU.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/MONA.json",
        "library": "MONA",
        "libraryname": "MONA",
        "mgflink": "/gnpslibrary/MONA.mgf",
        "msplink": "/gnpslibrary/MONA.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/RESPECT.json",
        "library": "RESPECT",
        "libraryname": "RESPECT",
        "mgflink": "/gnpslibrary/RESPECT.mgf",
        "msplink": "/gnpslibrary/RESPECT.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/HMDB.json",
        "library": "HMDB",
        "libraryname": "HMDB",
        "mgflink": "/gnpslibrary/HMDB.mgf",
        "msplink": "/gnpslibrary/HMDB.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/CASMI.json",
        "library": "CASMI",
        "libraryname": "CASMI",
        "mgflink": "/gnpslibrary/CASMI.mgf",
        "msplink": "/gnpslibrary/CASMI.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/SUMNER.json",
        "library": "SUMNER",
        "libraryname": "SUMNER",
        "mgflink": "/gnpslibrary/SUMNER.mgf",
        "msplink": "/gnpslibrary/SUMNER.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/BIRMINGHAM-UHPLC-MS-POS.json",
        "library": "BIRMINGHAM-UHPLC-MS-POS",
        "libraryname": "BIRMINGHAM-UHPLC-MS-POS",
        "mgflink": "/gnpslibrary/BIRMINGHAM-UHPLC-MS-POS.mgf",
        "msplink": "/gnpslibrary/BIRMINGHAM-UHPLC-MS-POS.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/BIRMINGHAM-UHPLC-MS-NEG.json",
        "library": "BIRMINGHAM-UHPLC-MS-NEG",
        "libraryname": "BIRMINGHAM-UHPLC-MS-NEG",
        "mgflink": "/gnpslibrary/BIRMINGHAM-UHPLC-MS-NEG.mgf",
        "msplink": "/gnpslibrary/BIRMINGHAM-UHPLC-MS-NEG.msp",
        "type": "IMPORT",
    },
    {
        "jsonlink": "/gnpslibrary/ALL_GNPS.json",
        "libraryname": "ALL_GNPS",
        "mgflink": "/gnpslibrary/ALL_GNPS.mgf",
        "msplink": "/gnpslibrary/ALL_GNPS.msp",
        "type": "AGGREGATION",
    },
    {
        "jsonlink": "/gnpslibrary/ALL_GNPS_NO_PROPOGATED.json",
        "libraryname": "ALL_GNPS_NO_PROPOGATED",
        "mgflink": "/gnpslibrary/ALL_GNPS_NO_PROPOGATED.mgf",
        "msplink": "/gnpslibrary/ALL_GNPS_NO_PROPOGATED.msp",
        "type": "AGGREGATION",
    },
]


def main() -> None:
    baseurl = "https://external.gnps2.org"
    basedir = os.path.join(config.SEISMIQ_RAW_DATA_FOLDER(), "GNPS")

    os.makedirs(basedir, exist_ok=True)

    for row in GNPS_DATA:
        fname = row["jsonlink"].split("/")[-1]
        dest = os.path.join(basedir, fname)
        if not os.path.exists(dest):
            cmd = f'wget {baseurl}{row["jsonlink"]} --no-check-certificate --directory-prefix={basedir}'
            print(cmd)
            os.system(cmd)
        else:
            print("not re-downloading", row["jsonlink"])

        try:
            with open(dest) as f:
                data = json.load(f)
        except:
            with open(dest) as f:
                for row in f:
                    json.loads(row)
                    break
            print("file", dest, "already processed")
            continue

        if isinstance(data, list):
            with open(dest, "w") as f:
                for row in data:
                    f.write(json.dumps(row))
                    f.write("\n")


if __name__ == "__main__":
    main()
