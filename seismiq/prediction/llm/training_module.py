import collections
import dataclasses
import resource
import time
import warnings
from collections.abc import Iterator
from typing import Any, cast

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from loguru import logger
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Fingerprints import FingerprintMols
from torch import Tensor, nn
from torch.nn import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.utils.data import DataLoader

from seismiq.prediction.layers import Mlp, PositionalEncoder
from seismiq.prediction.llm.data_module import DecoderOnlyDataSample, EncoderDecoderDataBatch
from seismiq.utils import rdkit_wrapper as wrdkit


@dataclasses.dataclass
class EncoderDecoderLlmPrediction:
    next_embedding: Tensor
    tokens: Tensor
    encoded_spectrum: Tensor


class EncoderDecoderLlmTrainingModule(LightningModule):
    def __init__(
        self,
        d_model: int = 256,
        peaks_dim: int = 64,
        vocab_size: int = 303,
        nhead: int = 2,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        label_smoothing: float = 0.01,
        log_on_step: bool = True,
        sample_every_train_batches: int = 50,
        sample_every_val_batches: int = 20,
        continuous_validation_step: int = 100,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        use_sample_weights: bool = True,
        predict_formula: bool = False,
        freeze_transformer: bool = False,
        is_finetuning: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        RDLogger.DisableLog("rdApp.*")  # type: ignore
        self.save_hyperparameters(
            ignore=[
                "optimizer",
                "lr_scheduler",  # the actual optimizer and scheduler are saved by the callback
            ]
        )

        self.log_on_step = log_on_step
        self.label_smoothing = label_smoothing
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.sample_every_train_batches = sample_every_train_batches
        self.sample_every_val_batches = sample_every_val_batches
        self.use_sample_weights = use_sample_weights
        self.freeze_transformer = freeze_transformer
        self.is_finetuning = is_finetuning

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.token_pos_encoder = PositionalEncoder(num_freqs=d_model // 2)

        self.peak_encoder = Mlp(peaks_dim, 2 * d_model, 1, d_model, nn.ReLU)
        self.token_remaining_embedding = Mlp(100 + d_model, 2 * d_model, 1, d_model, nn.ReLU)

        self.spec_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=nn.ReLU(),
                norm_first=True,
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=num_decoder_layers,
        )

        self.mol_decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=nn.ReLU(),
                norm_first=True,
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=num_decoder_layers,
        )

        self.token_predictor = Mlp(d_model, 2 * d_model, 1, vocab_size, nn.ReLU)

        self.cval_step = continuous_validation_step
        self._continuous_val_dataloader: Iterator[EncoderDecoderDataBatch] | None = None

        # not our fault, comes from within pandas
        warnings.filterwarnings(
            action="ignore",
            category=FutureWarning,
            message=r".*The behavior of DataFrame.sum with axis=None is deprecated.*",
        )

    def sample_from_batch(
        self,
        batch: EncoderDecoderDataBatch,
        max_sampling_steps: int = 500,
        temperature: float = 1.0,
        initial_tokens: int = 1,
    ) -> tuple[Tensor, Tensor]:
        tokenizer = batch.token_dencoder._smiles_tokenizer
        token_logits = []

        # keep a copy of original data
        batch.orig_token_mask = batch.token_mask  # type: ignore
        batch.orig_tokens = batch.tokens  # type: ignore
        batch.orig_remaining_atoms = batch.remaining_atoms  # type: ignore

        # initialize sampling variables
        batch.token_mask = torch.zeros(
            batch.tokens.shape[0], initial_tokens, dtype=torch.bool, device=batch.tokens.device
        )
        batch.tokens = batch.tokens[:, :initial_tokens]
        batch.remaining_atoms = batch.remaining_atoms[:, :initial_tokens]

        remaining: list[dict[int, int]] = []
        for i in range(batch.remaining_atoms.shape[0]):
            remaining.append(
                {
                    j: n
                    for j in range(batch.remaining_atoms.shape[2])
                    if (n := int(batch.remaining_atoms[i, initial_tokens - 1, j])) > 0
                }
            )

        encoded_spectrum = None
        with torch.no_grad():
            for _ in range(max_sampling_steps):
                # obtain logits for next token
                preds = self(batch, encoded_spectrum=encoded_spectrum)
                encoded_spectrum = preds.encoded_spectrum

                next_token_logits = preds.tokens[:, -1]
                token_logits.append(next_token_logits)

                # sample token
                if temperature != 0:
                    probs = torch.softmax(
                        next_token_logits / temperature,
                        dim=-1,
                    )
                    next_tokn = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokn = next_token_logits.argmax(dim=-1).unsqueeze(-1)

                # update remaining atoms
                new_remaining = torch.zeros(
                    batch.remaining_atoms.shape[0],
                    1,
                    batch.remaining_atoms.shape[2],
                    device=batch.remaining_atoms.device,
                )
                for j in range(next_tokn.shape[0]):
                    at = tokenizer.token_to_atom(int(next_tokn[j, 0]))
                    if at is not None:
                        _, num = at
                        if num not in remaining[j]:
                            remaining[j][num] = 0
                        remaining[j][num] -= 1
                    for a, n in remaining[j].items():
                        new_remaining[j, 0, a] = n

                # update sampling variables
                batch.tokens = torch.cat([batch.tokens, next_tokn], dim=1)
                batch.remaining_atoms = torch.cat([batch.remaining_atoms, new_remaining], dim=1)
                batch.token_mask = torch.cat(
                    [
                        batch.token_mask,
                        torch.zeros(batch.token_mask.shape[0], 1, dtype=torch.bool, device=batch.token_mask.device),
                    ],
                    dim=1,
                )

                if (batch.tokens == 2).sum(-1).min() > 0:
                    break

        return batch.tokens, torch.stack(token_logits, dim=1)

    def on_train_epoch_start(self) -> None:
        if self.freeze_transformer:
            for param in self.spec_encoder.parameters():
                param.requires_grad = False
            for param in self.mol_decoder.parameters():
                param.requires_grad = False

    def training_step(self, batch: EncoderDecoderDataBatch, batch_idx: int) -> dict[str, int | float | Tensor]:
        self._continuous_validation_step()
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch: EncoderDecoderDataBatch, batch_idx: int) -> dict[str, int | float | Tensor]:
        return self._step("val", batch, batch_idx)

    def _step(self, stage: str, x: EncoderDecoderDataBatch, batch_idx: int) -> dict[str, int | float | Tensor]:
        rus = resource.getrusage(resource.RUSAGE_SELF)
        ruc = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.log("memory/maxrss-self", float(rus.ru_maxrss))
        self.log("memory/maxrss-child", float(ruc.ru_maxrss))
        self.log("memory/maxrss-tot", float(rus.ru_maxrss + ruc.ru_maxrss))

        preds = cast(EncoderDecoderLlmPrediction, self(x))
        metrics = self.compute_metrics(x, preds, stage, batch_idx)
        assert "loss" in metrics

        for k, v in metrics.items():
            self.log(
                f"{k}/{stage}",
                v,
                prog_bar=(k == "loss"),
                on_step=self.log_on_step and (stage == "train" or stage == "cval"),
                on_epoch=(stage != "cval"),
                logger=True,
                batch_size=x.tokens.shape[0],
                sync_dist=(stage != "train"),
                rank_zero_only=(stage != "train"),
            )

        return metrics

    def _continuous_validation_step(self) -> None:
        def get_dataloader() -> DataLoader[DecoderOnlyDataSample]:
            if hasattr(self.trainer, "datamodule"):
                return self.trainer.datamodule.val_dataloader()  # type: ignore
            else:
                return self.val_dataloader()

        if self.cval_step > 0 and (self.global_step + 1) % self.cval_step == 0:
            if self._continuous_val_dataloader is None:
                self._continuous_val_dataloader = iter(get_dataloader())

            # get next batch
            try:
                assert self._continuous_val_dataloader is not None
                val_batch = next(self._continuous_val_dataloader)
            except StopIteration:
                self._continuous_val_dataloader = iter(get_dataloader())
                assert self._continuous_val_dataloader is not None
                val_batch = next(self._continuous_val_dataloader)

            # move tensors to the right device
            for k in dir(val_batch):
                v = getattr(val_batch, k)
                if isinstance(v, Tensor):
                    setattr(val_batch, k, v.to(self.device))

            # perform one step
            self.eval()
            with torch.no_grad():
                self._step("cval", val_batch, -1)
            self.train()

    def forward(
        self, batch: EncoderDecoderDataBatch, encoded_spectrum: Tensor | None = None
    ) -> EncoderDecoderLlmPrediction:
        if batch.tokens.max() >= self.token_embedding.num_embeddings:
            raise RuntimeError(f"not enough embeddings, need at least {batch.tokens.max()}")

        assert torch.isfinite(batch.tokens).all()
        assert torch.isfinite(batch.peaks).all()

        if encoded_spectrum is None:
            peaks_emb = self.peak_encoder(batch.peaks)
            zspec = self.spec_encoder(
                self.token_pos_encoder(peaks_emb),
                src_key_padding_mask=batch.peak_mask,
            )
        else:
            zspec = encoded_spectrum

        # predict smiles
        tok = torch.cat(
            [
                self.token_embedding(batch.tokens.long()),
                batch.remaining_atoms,
            ],
            dim=-1,
        )
        token_emb = self.token_remaining_embedding(tok)
        mask = Transformer.generate_square_subsequent_mask(token_emb.shape[1], device=token_emb.device)
        out = self.mol_decoder(
            self.token_pos_encoder(token_emb),
            memory=zspec,
            tgt_mask=torch.isinf(mask),
            tgt_is_causal=True,
            tgt_key_padding_mask=batch.token_mask,
            memory_key_padding_mask=batch.peak_mask,
        )

        assert torch.isfinite(out).all()

        ps = EncoderDecoderLlmPrediction(
            next_embedding=out,
            tokens=self.token_predictor(out),
            encoded_spectrum=zspec,
        )

        return ps

    def compute_metrics(
        self,
        batch: EncoderDecoderDataBatch,
        preds: EncoderDecoderLlmPrediction,
        stage: str,
        batch_idx: int,
    ) -> dict[str, int | float | Tensor]:
        pred_tokn = preds.tokens
        true_tokn = batch.tokens

        # weight of each sample depending on the frequency of the corresponding molecule in the dataset
        ws = batch.weights if self.use_sample_weights else torch.ones_like(batch.weights)
        ws = ws.unsqueeze(-1)
        den = batch.weights.sum()

        # token loss
        valid_tokens = 1 - batch.token_mask[..., :-1].float()
        tokn_loss = nn.functional.cross_entropy(
            pred_tokn[:, :-1, :].swapdims(-1, -2), true_tokn[:, 1:].long(), label_smoothing=0.01, reduction="none"
        )
        tokn_loss = tokn_loss * ws / den
        tokn_loss = tokn_loss * valid_tokens / valid_tokens.sum(dim=-1).unsqueeze(-1)
        tokn_loss = tokn_loss.sum()

        metrics: dict[str, float | Tensor] = {
            "loss_token": tokn_loss,
            "loss": tokn_loss,
        }

        assert torch.isfinite(cast(Tensor, metrics["loss"])).all()

        if (
            (stage == "train" and batch_idx % self.sample_every_train_batches == 0)
            or (stage == "val" and batch_idx % self.sample_every_val_batches == 0)
            or (stage == "cval")
        ):
            metrics.update(self.compute_tanimotos(batch))

        return metrics

    def compute_tanimotos(self, batch: EncoderDecoderDataBatch) -> dict[str, float]:
        ws: list[float] = (batch.weights if self.use_sample_weights else torch.ones_like(batch.weights)).tolist()

        # sample new molecule
        was_training = self.training
        if was_training:
            self.eval()
        tstart = time.time()
        toks, _ = self.sample_from_batch(batch, max_sampling_steps=cast(int, 10 + batch.tokens.shape[-1]))
        tend = time.time()
        ts = tend - tstart
        self.log("samplig/speed_tokpersec", toks.shape[0] / ts)
        self.log("samplig/total_time", ts)
        if was_training:
            self.train()

        # compute metrics
        metrics: dict[str, float] = collections.defaultdict(float)
        for i in range(toks.shape[0]):
            real_mol = Chem.MolFromSmiles(batch.smiles_str[i])  # type: ignore
            if real_mol is None:
                continue

            smi = cast(str, batch.token_dencoder.decode([int(t) for t in toks[i]], return_mol=False))

            try:
                pred_mol = wrdkit.mol_from_smiles(smi) if smi.count("[*]") == 0 else None
            except:
                logger.error(f'could not join fragments of predicted molecule with smiles "{smi}"')
                continue

            if pred_mol is None:
                continue

            real_fings = FingerprintMols.FingerprintMol(real_mol)
            pred_fings = FingerprintMols.FingerprintMol(pred_mol)
            tani = DataStructs.TanimotoSimilarity(real_fings, pred_fings)  # type: ignore

            key = "exp" if batch.is_experimental[i] else "syn"
            metrics[f"{key}_valid"] += ws[i] * 1.0
            metrics[f"{key}_valid_tanimoto"] += ws[i] * tani
            metrics[f"{key}_valid_meaningful_match"] += ws[i] * int(tani >= 0.400)
            metrics[f"{key}_valid_close_match"] += ws[i] * int(tani >= 0.675)
            metrics[f"{key}_valid_excellent_match"] += ws[i] * int(tani >= 0.800)
            metrics[f"{key}_valid_perfect_match"] += ws[i] * int(tani >= 0.999)

        # normalize by weight of valid
        metrics["exp_valid"] = metrics.get("exp_valid", 0) / sum(ws)
        metrics["syn_valid"] = metrics.get("syn_valid", 0) / sum(ws)
        for k in metrics:
            if k != "exp_valid" and k != "syn_valid":
                orig = k.split("_")[0]
                metrics[k] /= metrics[f"{orig}_valid"]

        return metrics

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore
        optimizer = self.optimizer(self.parameters())
        scheduler = self.lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_fit_start(self) -> None:
        if self.is_finetuning:
            # When finetuning on a different dataset we want to change the parameters of the optimizer,
            # in particular the learning rate, as well as the global step in the scheduler to do the
            # warmup again. The problem is that lightning forcefully restores these states from the checkpoint
            # sometime after calling this method, so even if we change the learning rate in the config
            # it will be ignored.
            # At this moment, the optimizer and scheduler are brand new as returned by configure_optimizers
            # above, so we can save their fresh state here and overwrite it later in on_train_start
            self._optimizer_states = [o.state_dict() for o in self.trainer.optimizers]
            self._optimizers_scheduler_states = [
                s.scheduler.state_dict() for s in self.trainer.strategy.lr_scheduler_configs
            ]

    def on_train_start(self) -> None:
        if self.is_finetuning:
            # At this point the optimizer and scheduler have been restored from the checkpoint;
            # If we are finetuning, we want to discard the restored state and start from scratch.
            logger.info("finetuning: discarding optimizer state in checkpoint and starting anew")
            for opt, state in zip(self.trainer.optimizers, self._optimizer_states):
                opt.load_state_dict(state)
            for sched, state in zip(self.trainer.strategy.lr_scheduler_configs, self._optimizers_scheduler_states):
                sched.scheduler.load_state_dict(state)
