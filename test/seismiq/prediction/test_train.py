import collections

from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch import Trainer

from seismiq.prediction.data.preparation import SeismiqDataPreparer
from seismiq.prediction.data.storage import InMemoryDataStorage
from seismiq.prediction.llm.data_module import EncoderDecoderLlmDataModule
from seismiq.prediction.llm.training_module import EncoderDecoderLlmTrainingModule


def test_train(tmp_path_factory):
    seed_everything(293857, workers=True)
    working_folder = tmp_path_factory.mktemp("model")

    model = EncoderDecoderLlmTrainingModule(
        d_model=16,
        peaks_dim=64,
        vocab_size=149,
        nhead=2,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        label_smoothing=0.01,
        log_on_step=True,
        sample_every_train_batches=5,
        sample_every_val_batches=2,
        continuous_validation_step=10,
    )

    data = EncoderDecoderLlmDataModule(
        batch_size=4,
        num_workers=0,
        storage=InMemoryDataStorage(
            datafile=str(working_folder / "dataset.pkl"),
        ),
        peaks_nfreq=64,
        preparer=SeismiqDataPreparer(
            sample_limit=1000,
            min_atoms=3,
            min_peaks=3,
            use_experimental_spectra=True,
            use_synthetic_spectra=True,
            only_molecules_with_experimental_spectra=False,
            only_samples_with_spectrum=True,
            num_workers=1,
        ),
        force_recreate=False,
        subsample=20,
        subsample_mols=50,
        smiles_augment=True,
        smiles_augment_prob=0.5,
        uniform_peak_sampling=False,
        mol_representation="smiles",
        split_seed=134214,
        sample_min_num_peaks=5,
        sample_max_num_peaks=50,
        peak_mz_noise=2e-3,
        precursor_mask_prob=0.5,
    )

    logged_losses = collections.defaultdict(list)

    original_log = model.log

    def log_override(k, v, *args, **kwargs):
        if "loss" in k:
            logged_losses[k].append(v.item())
        return original_log(k, v, *args, **kwargs)

    model.log = log_override  # type: ignore

    trainer = Trainer(accelerator="cpu", max_epochs=2, default_root_dir=str(working_folder))
    trainer.fit(model=model, datamodule=data)

    print(f'loss train 0 {logged_losses["loss/train"][0]:.3f}')
    print(f'loss train -1 {logged_losses["loss/train"][-1]:.3f}')
    print(f'loss val 0 {logged_losses["loss/val"][0]:.3f}')
    print(f'loss val -1 {logged_losses["loss/val"][-1]:.3f}')

    assert logged_losses["loss/train"][0] - logged_losses["loss/train"][-1] > 0.1
    assert logged_losses["loss/val"][0] - logged_losses["loss/val"][-1] > 0.1
