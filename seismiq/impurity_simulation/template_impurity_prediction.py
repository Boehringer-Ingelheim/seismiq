import copy
import itertools
import json

import click
from rdkit import Chem
from rdkit.Chem import rdChemReactions


class TemplateImpurityPrediction:
    def __init__(
        self,
        list_of_smiles: list,
        list_of_templates: list,
        number_of_cycles: int = 1,
    ):
        #
        self.list_of_smiles = copy.copy(list_of_smiles)
        self.sanitized_list_of_smiles = self.sanitize_list_of_smiles(self.list_of_smiles)
        #
        self.list_of_templates = copy.copy(list_of_templates)
        self.template_objects = [rdChemReactions.ReactionFromSmarts(template) for template in list_of_templates]
        #
        self.set_of_products = self.generate_impurities_from_rxns(
            self.sanitized_list_of_smiles,
            self.template_objects,
            number_of_cycles,
        )

    def generate_impurities_from_rxns(
        self,
        list_of_smiles,
        template_objects,
        number_of_cycles,
    ):
        set_of_products = set()
        predicted_products = set()
        list_of_reactants = []
        for smiles in list_of_smiles:
            list_of_reactants.append(smiles)

        for i in range(number_of_cycles):
            print("Cycle: ", i + 1)

            # Add products from previous cyclce to list of reactants
            for predicted_product in predicted_products:
                list_of_reactants.append(predicted_product)

            # Create all possible combinations of reactants
            reactants = list(itertools.product(list_of_reactants, repeat=2))

            # Generate tuples of rdkit mols for bimolecular reaction
            mols = []
            for reactant_tuple in reactants:
                mol_tuple = tuple(Chem.MolFromSmiles(smiles) for smiles in reactant_tuple)
                mols.append(mol_tuple)

            # Predict product based on a single template
            products = set()
            for mol in mols:
                for template in template_objects:
                    try:
                        # Forward prediction for a single template
                        product = template.RunReactants(mol)

                        # Extract smiles from product
                        for pp in product:
                            for p in pp:
                                try:
                                    smiles = Chem.MolToSmiles(p)
                                    products.add(smiles)
                                except Exception as e:
                                    print("Could not transform mol to smiles.", e)
                    except Exception as e:
                        print("Something did not work in RunReactants.", e)

            # Sanitize generated structures
            predicted_products = self.sanitize_list_of_smiles(products)
            predicted_products = set(predicted_products)

            # Add generated products from this cycle to the output set
            for product in predicted_products:
                set_of_products.add(product)

        return set_of_products

    def sanitize_list_of_smiles(self, list_of_smiles: list):
        sanitized_list_of_smiles = []
        for smiles in list_of_smiles:
            try:
                mol = Chem.rdmolfiles.MolFromSmiles(smiles)
                sanitized_smiles = Chem.rdmolfiles.MolToSmiles(mol)
                sanitized_list_of_smiles.append(sanitized_smiles)
            except Exception as e:
                print("Smiles are not sanitizable", smiles, e)
                continue
        return sanitized_list_of_smiles


@click.command()
@click.argument("rxn_smiles_file", type=click.File("r"))
@click.argument("templates_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
def main(rxn_smiles_file, templates_file, output_file):
    list_of_templates = [row.strip() for row in templates_file]

    res = {}
    for row in rxn_smiles_file:
        rxn_smiles = row.strip()
        list_of_smiles = rxn_smiles.split(".")
        impurities = TemplateImpurityPrediction(
            list_of_smiles=list_of_smiles,
            list_of_templates=list_of_templates,
            number_of_cycles=2,
        )

        res[rxn_smiles] = list(impurities.set_of_products)

    json.dump(res, output_file)


if __name__ == "__main__":
    main()
