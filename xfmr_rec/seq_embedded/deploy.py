from __future__ import annotations

from jsonargparse import auto_cli

from xfmr_rec.deploy import TestService
from xfmr_rec.seq.data import SeqDataModule
from xfmr_rec.seq_embedded import MODEL_NAME
from xfmr_rec.seq_embedded.service import Service
from xfmr_rec.seq_embedded.trainer import SeqEmbeddedLightningModule
from xfmr_rec.trainer import LightningCLI


def main(ckpt_path: str = "") -> None:
    """CLI helper: prepare trainer, save model, and run basic validation.

    When invoked this function prepares a Trainer (optionally loading a
    checkpoint), saves the Lightning module into the BentoML model store,
    and executes a short set of smoke tests against the exported Bento
    service to verify basic inference functionality.

    Args:
        ckpt_path: Optional path to Lightning checkpoint to load configuration.
    """
    cli = LightningCLI(SeqEmbeddedLightningModule, SeqDataModule, model_name=MODEL_NAME)
    trainer = cli.prepare_trainer(ckpt_path=ckpt_path)
    cli.save_model(trainer=trainer)
    TestService(Service).test_queries()


def cli_main() -> None:
    auto_cli(main, as_positional=False)


if __name__ == "__main__":
    cli_main()
