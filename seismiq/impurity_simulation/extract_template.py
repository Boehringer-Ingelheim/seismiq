import click
from rdchiral import template_extractor
from rdkit import Chem
from rxnmapper import RXNMapper


def extract_rxn_template(mapped_rxn: str):
    split_mapped_smiles = mapped_rxn.split(">")
    reaction = [
        {
            "_id": 0,
            "reactants": split_mapped_smiles[0],
            "products": split_mapped_smiles[2],
        }
    ]
    try:
        template = template_extractor.extract_from_reaction(reaction[0])
        return template["reaction_smarts"]
    except Exception as e:
        print("Template could not be extractred.", mapped_rxn, e)


def map_rxn(rxn_mapper, sanitized_reaction_smiles: str):
    try:
        mapped_rxn = rxn_mapper.get_attention_guided_atom_maps([sanitized_reaction_smiles])
        return mapped_rxn[0]["mapped_rxn"]
    except Exception as e:
        print(
            "Atom mapping can not be performed by RXNMapper on the offered reaction smiles.",
            sanitized_reaction_smiles,
            e,
        )


def sanitize_reaction_smiles(reaction_smiles: str):
    starting_material_smiles, reactant_smiles, product_smiles = reaction_smiles.split(">")
    sanitized_starting_smiles = sanitize_smiles(starting_material_smiles)
    sanitized_reactant_smiles = sanitize_smiles(reactant_smiles)
    sanitized_product_smiles = sanitize_smiles(product_smiles)
    sanitized_reaction_smiles = (
        sanitized_starting_smiles + ">" + sanitized_reactant_smiles + ">" + sanitized_product_smiles
    )
    sanitized_reaction_smiles = sanitized_reaction_smiles
    return sanitized_reaction_smiles


def sanitize_smiles(smiles: str):
    try:
        mols = Chem.rdmolfiles.MolFromSmiles(smiles)
        sanitized_smiles = Chem.rdmolfiles.MolToSmiles(mols)
        return sanitized_smiles
    except Exception as e:
        print("Smiles are not sanitizable.", smiles, e)


@click.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
def main(input_file, output_file):
    rxn_mapper = RXNMapper()
    for row in input_file:
        reaction_smiles = row.strip()
        sanitized_reaction_smiles = sanitize_reaction_smiles(reaction_smiles)
        mapped_rxn = map_rxn(rxn_mapper, sanitized_reaction_smiles)
        rxn_template = extract_rxn_template(mapped_rxn)
        output_file.write(f"{rxn_template}\n")


if __name__ == "__main__":
    main()
