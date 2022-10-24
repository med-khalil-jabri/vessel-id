import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_metric_learning as pml
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from src.models import VisionTransformer
from src.accuracy_calculator import Calculator
from src.visualizer import Visualizer
from src.metrics import SimilarityMetrics
from src.utils import BatchClassWeightedReducer


class ViTModule(pl.LightningModule):
    def __init__(self, args, data_module):
        super().__init__()
        self.args = args
        self.model = VisionTransformer(config=vars(args))
        self.data_module = data_module
        if self.args.load_from.endswith('.npz'):
            self.model.load_from(np.load(self.args.load_from))
        self.visualizer = Visualizer(self, args)
        self.val_similarity_metrics = SimilarityMetrics(top_k=(1, 3, 5))
        self.test_similarity_metrics = [SimilarityMetrics(top_k=(1, 3, 5)) for _ in range(2)]
    
    def setup(self, stage):
        self.distance = distances.CosineSimilarity()
        n_imos = len(self.data_module.imo2cat_weight)
        if self.args.class_weighting:
            self.reducer = BatchClassWeightedReducer(self.data_module.imo2cat_weight)
            self.loss_func = losses.CosFaceLoss(n_imos, self.args.output_size, distance=self.distance, reducer=self.reducer)
        else:
            self.loss_func = losses.CosFaceLoss(n_imos, self.args.output_size, distance=self.distance)
        self.mining_func = miners.AngularMiner(angle=self.args.cosface_margin)
    
    def forward(self, x, return_tokens_and_weights=False):
        x = x.to(self.device)
        if return_tokens_and_weights:
            return self.model(x, return_all=True)
        return self.model(x, return_all=False)
    
    def configure_optimizers(self):
        optmizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optmizer, eta_min=self.args.lr/10, T_0=5)
        return [optmizer], [scheduler]
    
    def log_gradient(self):
        grad_norm = [param.grad.data.norm(2) for param in self.model.parameters() if param.requires_grad
                     and param.grad is not None]
        if grad_norm:
            avg_grad_norm = torch.stack(grad_norm)
            self.log(f"gradient/average", torch.mean(avg_grad_norm), on_step=False, on_epoch=True)
            self.log(f"gradient/max", torch.max(avg_grad_norm), on_step=False, on_epoch=True)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self.model(images)
        indices_tuple = self.mining_func(embeddings, labels[:, 1])
        loss = self.loss_func(embeddings, labels[:, 1], indices_tuple)
        self.log("loss", loss)
        self.log_gradient()
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self.model(images)
        indices_tuple = self.mining_func(embeddings, labels[:, 1])
        loss = self.loss_func(embeddings, labels[:, 1], indices_tuple)
        self.log("val_loss", loss, prog_bar=True)
        self.val_similarity_metrics.update(embeddings, labels)

    def validation_epoch_end(self, _):
        top_k_accuracies = self.val_similarity_metrics.compute()
        for top_k, top_k_accuracy in top_k_accuracies.items():
            self.log(f"validation/acc@{top_k}/imo", top_k_accuracy, on_epoch=True)
        # Paired Image Similarity visualizaiton
        if self.global_rank == 0 and self.current_epoch % self.args.viz_freq == 0:
            for im_idx, im in enumerate(self.data_module.viz_images):
                fig, axes = plt.subplots(len(im), len(im), constrained_layout=True, figsize=(12,12))
                for i, key1 in enumerate(im):
                    for j, key2 in enumerate(im):
                        if i == j:
                            im_path = os.path.join(self.args.data_dir, str(im[key1]['id']) + '.jpg')
                            axes[i,j].imshow(Image.open(im_path))
                            axes[i,j].axis('off')
                        else:
                            im1_path = os.path.join(self.args.data_dir, str(im[key1]['id']) + '.jpg')
                            im2_path = os.path.join(self.args.data_dir, str(im[key2]['id']) + '.jpg')
                            map1, _ = self.visualizer.get_sim_maps(im1_path, im2_path)
                            axes[i,j].imshow(map1)
                            axes[i,j].axis('off')
                self.logger.experiment.add_figure('validation_pairwise_similarity/epoch_'+str(self.current_epoch)+'/image_'+str(im_idx), fig, self.current_epoch)
                plt.close(fig)
            # CAM visualization
            for method in ["gradcam", "gradcam++", "ablationcam", "eigencam", "eigengradcam"]:
                for im_idx, im in enumerate(self.data_module.viz_images):
                    fig, axes = plt.subplots(2, len(im), constrained_layout=True, figsize=(12,6))
                    anchor_path = os.path.join(self.args.data_dir, str(im['anchor']['id']) + '.jpg')
                    for i, key in enumerate(im):
                        if key == 'anchor':
                            axes[1,i].imshow(Image.open(anchor_path))
                            axes[0,i].axis('off')
                            axes[1,i].axis('off')
                        else:
                            im_path = os.path.join(self.args.data_dir, str(im[key]['id']) + '.jpg')
                            with torch.set_grad_enabled(True):
                                cam_img = self.visualizer.get_cam(anchor_path, im_path, method=method)
                            axes[0,i].imshow(Image.open(im_path))
                            axes[1,i].imshow(cam_img)
                            axes[0,i].axis('off')
                            axes[1,i].axis('off')
                    self.logger.experiment.add_figure('validation_'+method+'/epoch_'+str(self.current_epoch)+'/image_'+str(im_idx), fig, self.current_epoch)
                    plt.close(fig)
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        images, labels = batch
        embeddings = self.model(images)
        print(dataloader_idx)
        self.test_similarity_metrics[dataloader_idx].update(embeddings, labels)
    
    def test_epoch_end(self, _):
        seen_acc, seen_data_dict = self.test_similarity_metrics[0].compute(return_data=True)
        unseen_acc, unseen_data_dict = self.test_similarity_metrics[1].compute(return_data=True)
        if os.path.exists('visualization'):
            shutil.rmtree('visualization')
        os.makedirs('visualization')
        for tag, data_dict in zip(['seen', 'unseen'], [seen_data_dict, unseen_data_dict]):
            n_imgs = len(data_dict['ids'])
            n_samples = min(n_imgs, self.args.n_test_viz)
            idxes = np.random.choice(n_imgs, n_samples)
            distances, neighbours, imos, ids = data_dict['dist'], data_dict['neighbours'], data_dict['imos'], data_dict['ids']
            for method in ['pairwise', 'gradcam++', 'ablationcam', 'eigencam']:
                dir_path = os.path.join('visualization', tag, method)
                os.makedirs(dir_path)
                if method == 'pairwise':
                    for idx in idxes:
                        fig, axes = plt.subplots(self.args.n_neighbours + 1, self.args.n_neighbours + 1, constrained_layout=True, figsize=(12,12))
                        for i in range(self.args.n_neighbours + 1):
                            for j in range(self.args.n_neighbours + 1):
                                if i == j:
                                    im_path = os.path.join(self.args.data_dir, f"{ids[neighbours[idx, i]]}.jpg")
                                    axes[i,j].imshow(Image.open(im_path))
                                    if i == 0:
                                        axes[i,j].set_title(f"IMO: {imos[neighbours[idx, i]]}", loc='center')
                                        axes[i,j].axis('off')
                                    else:
                                        axes[i,j].set_title(f"d = {distances[idx, i]:.2f}", loc='center')
                                        color = 'green' if imos[neighbours[idx, 0]] == imos[neighbours[idx, i]] else 'red'
                                        axes[i,j].set_xticks([])
                                        axes[i,j].set_yticks([])
                                        for spine in axes[i,j].spines.values():
                                            spine.set_edgecolor(color)
                                            spine.set_linewidth(5)
                                else:
                                    im1_path = os.path.join(self.args.data_dir, f"{ids[neighbours[idx, i]]}.jpg")
                                    im2_path = os.path.join(self.args.data_dir, f"{ids[neighbours[idx, j]]}.jpg")
                                    map1, _ = self.visualizer.get_sim_maps(im1_path, im2_path)
                                    axes[i,j].imshow(map1)
                                    axes[i,j].axis('off')
                        plt.savefig(os.path.join(dir_path, f"{ids[idx]}.jpg"))
                        plt.close(fig)
                else:
                    for idx in idxes:
                        fig, axes = plt.subplots(2, self.args.n_neighbours + 1, constrained_layout=True, figsize=(12,6))
                        anchor_path = os.path.join(self.args.data_dir, f"{ids[idx]}.jpg")
                        axes[1,0].imshow(Image.open(im_path))
                        axes[1,0].set_title(f"IMO: {imos[idx]}", loc='center')
                        axes[0,0].axis('off')
                        axes[1,0].axis('off')
                        for i in range(1, self.args.n_neighbours + 1):
                            im_path = os.path.join(self.args.data_dir, f"{ids[neighbours[idx, i]]}.jpg")
                            with torch.set_grad_enabled(True):
                                cam_img = self.visualizer.get_cam(anchor_path, im_path, method=method)
                            axes[0,i].imshow(Image.open(im_path))
                            axes[1,i].imshow(cam_img)
                            axes[1,i].set_title(f"d = {distances[idx, i]:.2f}", loc='center')
                            color = 'green' if imos[neighbours[idx, 0]] == imos[neighbours[idx, i]] else 'red'
                            axes[0,i].set_xticks([])
                            axes[0,i].set_yticks([])
                            for spine in axes[0,i].spines.values():
                                spine.set_edgecolor(color)
                                spine.set_linewidth(5)
                            axes[1,i].axis('off')
                        plt.savefig(os.path.join(dir_path, f"{ids[idx]}.jpg"))
                        plt.close(fig)
        print('Accuracy on seen data: ', seen_acc)
        print('Accuracy on unseen data: ', unseen_acc)