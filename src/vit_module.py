import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_metric_learning as pml
from pytorch_lightning import Trainer
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from src.models import VisionTransformer


class ViTModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = VisionTransformer(config=vars(args))
        if self.args.load_from:
            self.model.load_from(np.load(self.args.load_from))
        self.distance = distances.CosineSimilarity()
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss_func = losses.TripletMarginLoss(margin=0.2, distance=self.distance, reducer=self.reducer)
        self.mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=self.distance, type_of_triplets="semihard"
        )
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    def training_step(self, batch, batch_idx):
        images, labels, imos = batch
        embeddings, _, _ = self.model(images)
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss_func(embeddings, labels, indices_tuple)
        self.log("loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels, imos = batch
        embeddings, _, _ = self.model(images)
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss_func(embeddings, labels, indices_tuple)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return embeddings, labels, imos
    
    def validation_epoch_end(self, validation_step_outputs):
        outputs = list(zip(*validation_step_outputs))
        embeddings, labels, imos = tuple(map(lambda x: torch.cat(x), outputs))
        accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
        imo_accuracy = accuracy_calculator.get_accuracy(embeddings, embeddings, imos, imos, True)
        cat_accuracy = accuracy_calculator.get_accuracy(embeddings, embeddings, labels, labels, True)
        self.log("validation/acc@1/imo", imo_accuracy, on_epoch=True)
        self.log("validation/acc@1/category", cat_accuracy, on_epoch=True)
    
    