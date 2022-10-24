import gc
from typing import Tuple
import faiss
import torch
import numpy as np
import torch.nn.functional as F
from torchmetrics import Metric


class SimilarityMetrics(Metric):
    def __init__(self, top_k: Tuple[int, ...] = (5,)):
        super().__init__()
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")
        self.add_state("objects_labels", default=[], dist_reduce_fx="cat")
        self.add_state("objects_ids", default=[], dist_reduce_fx="cat")
        self.k_for_tops = top_k

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.embeddings.append(F.normalize(preds, p=2, dim=1))
        self.objects_labels.append(target[:, 1])
        self.objects_ids.append(target[:, 2])

    def compute(self, return_data=False):
        embeddings = torch.cat(self.embeddings) if type(self.embeddings) == list else self.embeddings
        objects_labels = torch.cat(self.objects_labels) if type(self.objects_labels) == list else self.objects_labels
        objects_ids = torch.cat(self.objects_ids) if type(self.objects_ids) == list else self.objects_ids
        index = faiss.IndexFlatIP(embeddings.shape[1])
        if embeddings.is_cuda:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(embeddings)
        distances, pred_indices = index.search(embeddings, max(self.k_for_tops) + 1)
        pred_indices_with_objects_labels = torch.empty_like(pred_indices)
        for idx in range(len(objects_labels)):
            pred_indices_with_objects_labels[pred_indices == idx] = objects_labels[idx]
        top_k_accuracies = {}
        for top_k in self.k_for_tops:
            neighbours_indices = pred_indices_with_objects_labels[:, 1:top_k + 1]
            matches = (neighbours_indices == pred_indices_with_objects_labels[:, 0].view(-1, 1)).float()
            matches_top_k = matches.sum(dim=1).clip(max=1)
            top_k_accuracies[top_k] = matches_top_k.mean()
        del index
        gc.collect()
        if return_data:
            data_dict = {
                'dist': distances, 
                'neighbours': pred_indices, 
                'imos': objects_labels, 
                'ids': objects_ids
            }
            return top_k_accuracies, data_dict
        return top_k_accuracies
