import pytest
import torch
import numpy as np
from xfmr_rec.data import SeqDataModule, SeqDataModuleConfig
from xfmr_rec.trainer import RecommenderLightningModule, LightningConfig
from xfmr_rec.models import RecommenderModel, ModelConfig

@pytest.fixture(scope="module")
def datamodule():
    config = SeqDataModuleConfig(num_workers=0, batch_size=2)
    dm = SeqDataModule(config)
    dm.prepare_data()
    dm.setup()
    return dm

def test_data_module(datamodule):
    assert datamodule.items_dataset is not None
    assert datamodule.users_dataset is not None

    batch = next(iter(datamodule.train_dataloader()))
    assert "history_item_idx" in batch
    assert "pos_item_idx" in batch
    assert "neg_item_idx" in batch
    assert batch["history_item_idx"].shape[0] == 2

def test_model_forward(datamodule):
    config = LightningConfig()
    module = RecommenderLightningModule(config)
    module.items_dataset = datamodule.items_dataset
    module.configure_model()

    batch = next(iter(datamodule.train_dataloader()))
    output = module(batch["history_item_idx"])
    assert "sentence_embedding" in output
    assert "token_embeddings" in output

def test_training_step(datamodule):
    config = LightningConfig()
    module = RecommenderLightningModule(config)
    module.items_dataset = datamodule.items_dataset
    module.configure_model()

    batch = next(iter(datamodule.train_dataloader()))
    loss = module.training_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_validation_step(datamodule):
    config = LightningConfig()
    module = RecommenderLightningModule(config)
    module.items_dataset = datamodule.items_dataset
    module.configure_model()
    module.items_index.index_data(datamodule.items_dataset)
    module.users_index.index_data(datamodule.users_dataset)

    val_batch = next(iter(datamodule.val_dataloader()))
    # Validation step expects a single row from the dataloader in this implementation
    metrics = module.validation_step(val_batch)
    assert "val/retrieval_normalized_dcg" in metrics
