import torch
from lightning.pytorch.cli import LightningCLI


def main() -> None:
    torch.set_float32_matmul_precision("medium")

    LightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
        subclass_mode_data=True,
        subclass_mode_model=True,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    main()
