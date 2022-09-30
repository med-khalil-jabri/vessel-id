from typing import Tuple
import faiss
import torch
import torch.nn.functional as F
from torchmetrics import Metric



class SimilarityMetrics(Metric):
    def __init__(self, top_k: Tuple[int, ...] = (5,)):
        super().__init__()
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")
        self.add_state("objects_ids", default=[], dist_reduce_fx="cat")
        self.k_for_tops = top_k

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.embeddings.append(preds)
        self.objects_ids.append(target[:, 1])

    def compute(self):
        embeddings = torch.cat(self.embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        objects_ids = torch.cat(self.objects_ids)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        if self.device.type == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(embeddings)
        _, pred_indices = index.search(embeddings, max(self.k_for_tops) + 1)
        pred_indices_with_objects_ids = torch.empty_like(pred_indices)
        for idx in range(len(objects_ids)):
            pred_indices_with_objects_ids[pred_indices == idx] = objects_ids[idx]
        top_k_accuracies = {}
        for top_k in self.k_for_tops:
            neighbours_indices = pred_indices_with_objects_ids[:, 1:top_k + 1]
            matches = (neighbours_indices == pred_indices_with_objects_ids[:, 0].view(-1, 1)).float()
            matches_top_k = matches.sum(dim=1).clip(max=1)
            top_k_accuracies[top_k] = matches_top_k.mean()
        return top_k_accuracies
