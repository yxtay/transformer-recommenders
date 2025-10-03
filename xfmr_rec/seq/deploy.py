from __future__ import annotations

from jsonargparse import auto_cli

from xfmr_rec.deploy import test_queries
from xfmr_rec.seq import MODEL_NAME
from xfmr_rec.seq.data import SeqDataModule
from xfmr_rec.seq.service import Service
from xfmr_rec.seq.trainer import SeqRecLightningModule
from xfmr_rec.trainer import LightningCLI


def main(ckpt_path: str = "") -> None:
    """CLI helper: prepare a trainer, save the model, and run smoke tests.

    When called this function prepares a Lightning Trainer (optionally
    loading a checkpoint), saves the model into the BentoML store, and runs
    a small suite of sanity checks against the exported Bento service.

    Args:
        ckpt_path: Optional checkpoint path to load model/data configuration.
    """

    cli = LightningCLI(SeqRecLightningModule, SeqDataModule, model_name=MODEL_NAME)
    trainer = cli.prepare_trainer(ckpt_path=ckpt_path)
    cli.save_model(trainer=trainer)
    test_queries(Service)


def cli_main() -> None:
    auto_cli(main, as_positional=False)


if __name__ == "__main__":
    cli_main()
