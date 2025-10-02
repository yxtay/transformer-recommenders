from __future__ import annotations

from jsonargparse import auto_cli

from xfmr_rec.deploy import TestService
from xfmr_rec.mf import MODEL_NAME
from xfmr_rec.mf.data import MFDataModule
from xfmr_rec.mf.service import Service
from xfmr_rec.mf.trainer import MFRecLightningModule
from xfmr_rec.trainer import LightningCLI


def main(ckpt_path: str = "") -> None:
    """CLI entry point used to prepare, save, and validate a deployed model.

    When invoked, this function prepares a Trainer (optionally loading a
    checkpoint), saves the model into the BentoML model store, and runs a
    set of smoke tests against the bundled Bento service.

    Args:
        ckpt_path: Optional path to a Lightning checkpoint to load.
    """

    cli = LightningCLI(MFRecLightningModule, MFDataModule, model_name=MODEL_NAME)
    trainer = cli.prepare_trainer(ckpt_path=ckpt_path)
    cli.save_model(trainer=trainer)
    TestService(Service).test_queries()


def cli_main() -> None:
    auto_cli(main, as_positional=False)


if __name__ == "__main__":
    cli_main()
