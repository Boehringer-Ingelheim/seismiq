#!/usr/bin/env python
import os

import bson
import py7zr

# This script extracts a .7z archive and reads .bson files from it.
# Ensure to clone the repository and have the required files in place.
# https://github.com/connorcoley/ochem_predict_nn.git, DOI: 10.1021/acscentsci.7b00064

cwd = os.getcwd()
print(f"Current working directory: {cwd}")


archive_path = f"{cwd}/dump.7z"
output_path = f"{cwd}/data_from_figshare/"

if not os.path.exists(os.path.join(output_path, "dump")):
    # Extract the .7z file
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=output_path)
    print(f"Contents extracted to: {output_path}")

else:
    print(f"Directory already exists: {os.path.join(output_path, 'dump')}")
    print("Skipping extraction.")


# Path to the .bson file
bson_file_path_refs = f"{cwd}/data_from_figshare/dump/askcos_transforms/lowe_refs_general_v3.bson"

# Open and read the second .bson file
with open(bson_file_path_refs, "rb") as file_refs:
    data_templates = bson.decode_all(file_refs.read())

# Print the first few entries of the second .bson file
print(f"Number of templates in refs: {len(data_templates)}")
print("First template in refs:", data_templates[0]["reaction_smarts"])
