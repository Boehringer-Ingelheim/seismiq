# SEISMiQ

A machine learning model elucidating molecular structures from tandem mass spectrometry data.

This is the official code release for [our paper](https://pubs.rsc.org/en/Content/ArticleLanding/2025/DD/D5DD00115C):

E. Dorigatti, J. Groß, J. Kühlborn, R. Möckel, F. Maier and J. Keupp, Enhancing automated drug substance impurity structure elucidation from tandem mass spectra through transfer learning and domain knowledge, Digital Discovery, 2025, DOI: 10.1039/D5DD00115C.

## Installation

Installation via [Poetry](https://python-poetry.org/):

```bash
poetry install
```

The supplementary material for our publication is on [Zenodo](https://doi.org/10.5281/zenodo.15790300) and contains, amongst others:
 - Model checkpoints for inference and fine-tuning ([download](https://zenodo.org/records/16438770/files/checkpoints.zip?download=1))
 - Training data in CSV format ([download](https://zenodo.org/records/16438770/files/training_data.csv.gz?download=1))


## Prediction

Predictions for new MS/MS spectra can be obtained via the following command:

```
> python seismiq/prediction/predict.py --help
Usage: predict.py [OPTIONS] MODEL_NAME INPUT_FILE RESULT_FILE

Options:
  --num-beams INTEGER             Number of beams for beam search
  --max-sampling-steps INTEGER    Maximum number of sampling steps
  --peak-mz-noise FLOAT           Noise level for peak m/z values
  --skip-wrong-atom-count / --keep-wrong-atom-count
                                  Skip samples with wrong heavy atom count
  --keep-partial-samples / --skip-partial-samples
                                  Keep samples that were not fully generated
  --match-hydrogen-count / --no-match-hydrogen-count
                                  Match number of hydrogen atoms in generated
                                  samples
  --help                          Show this message and exit.
```

Where the model name is the explicit path to a checkpoint file or the name of a checkpoint file, without extension, contained in the path given by the environment variable `SEISMIQ_CHECKPOINTS_FOLDER`.

We provide checkpoints for the pretrained model as well as a model finetuned on CASMI in our Zenodo dataset:

```bash
mkdir -p dev
wget -O dev/checkpoints.zip "https://zenodo.org/records/16438770/files/checkpoints.zip?download=1" \
  && unzip dev/checkpoints.zip -d dev/ \
  && rm dev/checkpoints.zip \
  && export SEISMIQ_CHECKPOINTS_FOLDER=dev/checkpoints

wget -O dev/tokenizer.pkl "https://zenodo.org/records/16438770/files/tokenizer.pkl?download=1" \
  && export SEISMIQ_TOKENIZER_OVERRIDE=dev/tokenizer.pkl
```

Then, to obtain predictions from the pretrained model (make sure to have enough RAM or a capable GPU):

```bash
python seismiq/prediction/predict.py \
  seismiq_pretrained \
  resources/examples/casmi_2016_5.json \
  predictions.csv
```

The input file should contain a JSON list of challenges including, at minimum, the spectrum and
sum formula, for example:

```json
[
  {
    "sum_formula": "C7H10N2",
    "spectrum": [
      [
        79.0414,    // mass
        1767018.1   // unnormalized intensity
      ],
      // etc. ...
    ]
    // additional optional keys
    //  - smiles_prefix (str)      : prompt for the model
    //  - adduct_shift (float)     : adduct shift, defaults to M+H
    //  - true_smiles (str)        : actual SMILES used to evaluate the predictions (if given)
    //  - max_sampling_steps (int) : maximum number of tokens to sample
  },
  // etc ...
]
```

And the output file will contain hte predictions in CSV format, for example:

```csv
index,perplexity,tanimoto,pred_smiles,generation_count
0,2.434374139169987,0.0,CCC=NNc1ncnc(N)n1,1
1,2.45338491081399,0.0,N#CNC(=NCCCN)NC#N,1
2,2.200015703998429,0.0,C=CCN=C(N)Nc1ncn[nH]1,1
3,2.3174031691083026,0.0,C=CCNC(N)=Nc1ncn[nH]1,1
```

Where `pred_smiles` contains the predicted smiles and `generation_count` the number of times the model predicted this molecule, including predictions with different SMILES (only the lowest perplexity SMILES is returned).

## Model training

Data preparation and model training are handled via [Lightning](https://lightning.ai/docs/pytorch/stable/) and configured through yaml files passed to the training script:

```bash
python seismiq/prediction/train.py fit --config configs/seismiq_pretrained.yaml
```

This configuration trains the pretrained model as described in our publication, using the dataset on Zenodo:

```bash
mkdir -p dev

wget -O dev/training_data.csv.gz "https://zenodo.org/records/16438770/files/training_data.csv.gz?download=1" \
  && gunzip dev/training_data.csv.gz

# Test datasets, necessary to remove test molecules from training data
wget -O dev/test_data.zip "https://zenodo.org/records/16438770/files/test_datasets.zip?download=1" \
  && unzip dev/test_data.zip -d dev/ \
  && rm dev/test_data.zip \
  && export SEISMIQ_TEST_DATA_FOLDER=dev/test_datasets

wget -O dev/tokenizer.pkl "https://zenodo.org/records/16438770/files/tokenizer.pkl?download=1" \
  && export SEISMIQ_TOKENIZER_OVERRIDE=dev/tokenizer.pkl
```

On first launch, the CSV dataset will be converted to pickle files.
It will take a while.

### Generating new training data

Data preparation is performed automatically upon invokation of the training script whenever the base folder indicated in the data storage configuration does not exist:

```yaml
data:
  class_path: seismiq.prediction.llm.data_module.EncoderDecoderLlmDataModule
  init_args:
    storage:
      class_path: seismiq.prediction.data.storage.OnDiskBlockDataStorage
      init_args:
        base_folder: dev/training_data
        block_size: 10000
    preparer:
      class_path: seismiq.prediction.data.preparation.CsvDataPreparer
      init_args:
        csv_file: dev/training_data.csv
    # other data module parameters ...
```

Data for training is saved as a sequence of pickle files, and the `storage` object takes care of saving and loading samples from this format.
The `preparer` object reads data from some source and transforms it into `SeismiqSample` objects, which eventually end up in the pickle files, containing all necessary information to train the model.

The default `CsvDataPreparer` reads the provided CSV.
We also provide a `SyntheticDataPreparer` class which can be used to produce a dataset of simulated mass spectra given a list of molecules in SMILES format.
This is meant to be subclassed or encapsulated by another class that loads these molecules from somewhere.

### Installing mass spectrum predictors

To simulate the mass spectra for training, please install CFM-ID and FragGenie, which can be
obtained from:

 - https://bitbucket.org/wishartlab/cfm-id-code/src/master/
 - https://github.com/neilswainston/FragGenie

The two predictors are wrapped by two scripts indicated by the environment variables
`SEISMIQ_FRAGGENIE_PROGRAM` and `SEISMIQ_CFM_ID_PROGRAM`.
Examples for these scripts are provided in the `./scripts` folder.

### Fine-tuning

To fine-tune the model on a dataset, use the pretrained configuration in conjunction with the
dataset configuration and the finetuning configuration. The latter also specifies a default path
for the pre-trained checkpoint, which can be overridden. If using another checkpoint, also
make sure that it is compatible with the specified pretrained configuration.

Concretely,
to fine-tune on simulated spectra of the CASMI challenges:

```bash
wget -O dev/all_casmi_simulated.pkl "https://zenodo.org/records/16438770/files/all_casmi_simulated.pkl?download=1"

python seismiq/prediction/train.py fit --config configs/seismiq_pretrained.yaml \
  --config configs/seismiq_finetuned.yaml \
  --config configs/data_casmi.yaml \
  --ckpt_path dev/checkpoints/seismiq_pretrained.ckpt
```

### Evaluation

Evaluation is performed by two scripts, one for de novo generation and one for fragmentation,
for example:

```bash
# or seismiq/prediction/eval_on_fragments.py
python seismiq/prediction/eval_on_test_datasets.py run-single \
  seismiq_pretrained casmi_2016 dev/test_results/pretrained-c16.pkl
```

These scripts can also launch a parallel evaluation of all models on all datasets on SLURM:

```bash
python seismiq/prediction/eval_on_test_datasets.py \
  make-slurm-command --slurm-flags "--partition gpu --mem 96G --gres gpu:1" | bash
```

To use the public test datasets, define the env variable `SEISMIQ_TEST_DATA_FOLDER` pointing
to a folder with the appropriate json files, and `SEISMIQ_CHECKPOINTS_FOLDER` pointing to
a folder with the checkpoints.

Our model's predictions are provided on Zenodo, and the code to download them and generate the
result figures in the paper is in the Jupyter notebook `notebooks/figures.ipynb`.

## Impurity simulation

Exemplary scripts and files are provided to demonstrate how to extract reaction templates from
a given reaction smiles string and apply templates for forward prediction on reactants, as it
was done for the results in the manuscript.

By subjecting a reaction string to the code, a template can be extracted, for example:

```bash
python seismiq/impurity_simulation/extract_template.py \
  resources/examples/reaction_smiles.txt ./templates.txt
```

By providing reactants in SMILES notation and reaction templates, possible impurity structures
can be generated, for example:

```bash
python seismiq/impurity_simulation/template_impurity_prediction.py \
  resources/examples/reaction_smiles.txt resources/examples/reaction_templates.txt ./impurities.py
```

## Citation

If you use our model or dataset, we would be grateful if you acknowledged our publication:

```bibtex
@article{Dorigatti_2025_seismiq,
  title = {Enhancing automated drug substance impurity structure elucidation from tandem mass spectra through transfer learning and domain knowledge.},
  ISSN = {2635-098X},
  url = {http://dx.doi.org/10.1039/D5DD00115C},
  DOI = {10.1039/d5dd00115c},
  journal = {Digital Discovery},
  publisher = {Royal Society of Chemistry (RSC)},
  author = {Dorigatti,  Emilio and Groß,  Jonathan and K\"{u}hlborn,  Jonas and M\"{o}ckel,  Robert and Maier,  Frank and Keupp,  Julian},
  year = {2025}
}
```
