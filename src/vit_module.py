import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_metric_learning as pml
from pytorch_lightning import Trainer
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from src.models import VisionTransformer
from src.accuracy_calculator import Calculator


class ViTModule(pl.LightningModule):
    def __init__(self, args, data_module):
        super().__init__()
        self.args = args
        self.model = VisionTransformer(config=vars(args))
        self.data_module = data_module
        if self.args.load_from:
            self.model.load_from(np.load(self.args.load_from))
        self.distance = distances.CosineSimilarity()
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss_func = losses.TripletMarginLoss(margin=0.2, distance=self.distance, reducer=self.reducer)
        self.mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=self.distance, type_of_triplets="semihard"
        )
        self.tester = testers.BaseTester(dataloader_num_workers=self.args.num_workers)
        self.accuracy_calculator = Calculator(include=("precision_at_1", "precision_at_3", "precision_at_5"), k=5)
    
    def forward(self, x, return_tokens_and_weights=False):
        embeddings, prepooled_tokens, attn_weights = self.model(x)
        if return_tokens_and_weights:
            return embeddings, prepooled_tokens, attn_weights
        return embeddings
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings, _, _ = self.model(images)
        indices_tuple = self.mining_func(embeddings, labels[:, 1])
        loss = self.loss_func(embeddings, labels[:, 1], indices_tuple)
        self.log("loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        embeddings, _, _ = self.model(images)
        indices_tuple = self.mining_func(embeddings, labels[:, 1])
        loss = self.loss_func(embeddings, labels[:, 1], indices_tuple)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return embeddings, labels
    
    def validation_epoch_end(self, validation_step_outputs):
        outputs = list(zip(*validation_step_outputs))
        val_embeddings, val_labels = tuple(map(lambda x: torch.cat(x), outputs))
        train_embeddings, train_labels = self.tester.get_all_embeddings(self.data_module.train_ds, self)
        cat_accuracy = self.accuracy_calculator.get_accuracy(val_embeddings, train_embeddings, val_labels[:, 0], train_labels[:, 0], False)
        imo_accuracy = self.accuracy_calculator.get_accuracy(val_embeddings, train_embeddings, val_labels[:, 1], train_labels[:, 1], False)
        self.log("validation/acc@1/imo", imo_accuracy["precision_at_1"], on_epoch=True)
        self.log("validation/acc@1/category", cat_accuracy["precision_at_1"], on_epoch=True)
        self.log("validation/acc@3/imo", imo_accuracy["precision_at_3"], on_epoch=True)
        self.log("validation/acc@3/category", cat_accuracy["precision_at_3"], on_epoch=True)
        self.log("validation/acc@5/imo", imo_accuracy["precision_at_5"], on_epoch=True)
        self.log("validation/acc@5/category", cat_accuracy["precision_at_5"], on_epoch=True)