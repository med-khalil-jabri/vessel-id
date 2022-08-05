import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from numpy import matlib as mb
from src.utils import norm
from src.dataset_norms import dataset_norms


class Visualizer:
    def __init__(self, model):
        self.model = model

    def get_sim_maps(self, imageA_path, imageB_path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_norms['imagenet21k']['mean'], dataset_norms[('imagenet21k')]['std'])])
        transform_greyscale = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=3)
        ])
        simmapA, simmapB, sim_score = self.generate_sim_maps(imageA_path, imageB_path, transform)
        
        imageA_gs = transform_greyscale(Image.open(imageA_path).convert("RGB"))
        imageA_gs = np.array(imageA_gs) / 255

        imageB_gs = transform_greyscale(Image.open(imageB_path).convert("RGB"))
        imageB_gs = np.array(imageB_gs) / 255

        simmapA = norm(cv2.resize(simmapA, (256, 256), interpolation=cv2.INTER_CUBIC))
        simmapA = self.show_cam_on_image(imageA_gs, simmapA, use_rgb=True)

        simmapB = norm(cv2.resize(simmapB, (256, 256), interpolation=cv2.INTER_CUBIC))
        simmapB = self.show_cam_on_image(imageB_gs, simmapB, use_rgb=True)

        simmapA = Image.fromarray(simmapA)
        simmapB = Image.fromarray(simmapB)

        return simmapA, simmapB
    
    def compute_spatial_similarity(self, prepooledA, pooledA, prepooledB, pooledB):
        """
        Takes in the last convolutional layer from two images, computes the pooled output
        feature, and then generates the spatial similarity map for both images.
        """
        score = np.dot(pooledA/np.linalg.norm(pooledA), pooledB/np.linalg.norm(pooledB))
        out_sz = (int(np.sqrt(prepooledA.shape[0])), int(np.sqrt(prepooledA.shape[0])))
        prepooledA_normed = prepooledA / np.linalg.norm(pooledA) / prepooledA.shape[0]
        prepooledB_normed = prepooledB / np.linalg.norm(pooledB) / prepooledB.shape[0]
        im_similarity = np.zeros((prepooledA_normed.shape[0], prepooledA_normed.shape[0]))
        for zz in range(prepooledA_normed.shape[0]):
            repPx = mb.repmat(prepooledA_normed[zz,:], prepooledA_normed.shape[0],1)
            t = np.multiply(repPx, prepooledB_normed)
            im_similarity[zz, :] = t.sum(axis=1)
        similarityA = np.reshape(np.sum(im_similarity, axis=1), out_sz)
        similarityB = np.reshape(np.sum(im_similarity, axis=0), out_sz)
        return similarityA, similarityB, score

    def compute_rollout(self, attn_weights, start_layer=0, attn_head_agg='mean'):
        attn_weights = torch.tensor(attn_weights)
        if attn_head_agg == 'mean':
            attn_weights = torch.mean(attn_weights, dim=1)  # avg across heads
        elif attn_head_agg == 'min':
            attn_weights = torch.min(attn_weights, dim=1)[0]  # min across heads
        elif attn_head_agg == 'max':
            attn_weights = torch.max(attn_weights, dim=1)[0]  # max across heads
        else:
            raise ValueError('invalid input for "attn_head_agg", must be "mean", "min", or "max"')

        num_tokens = attn_weights[0].shape[1]
        eye = torch.eye(num_tokens).to(attn_weights[0].device)
        attn_weights = [attn_weights[i] + eye for i in range(len(attn_weights))]
        attn_weights = [attn_weights[i] / attn_weights[i].sum(dim=-1, keepdim=True) for i in range(len(attn_weights))]
        rollout_output = attn_weights[start_layer]
        for i in range(start_layer + 1, len(attn_weights)):
            rollout_output = attn_weights[i].matmul(rollout_output)
        return rollout_output

    def generate_sim_maps(self, A_path, B_path, transform):

        inpA = transform(Image.open(A_path).convert('RGB')).unsqueeze(0)
        inpB = transform(Image.open(B_path).convert('RGB')).unsqueeze(0)

        outputsA = list(self.model(inpA, return_tokens_and_weights=True))
        outputsB = list(self.model(inpB, return_tokens_and_weights=True))

        for i in range(len(outputsA)):
            outputsA[i] = outputsA[i].cpu().numpy().squeeze()
        for i in range(len(outputsB)):
            outputsB[i] = outputsB[i].cpu().numpy().squeeze()

        output_featA, prepooled_tokensA, attn_weightsA = outputsA
        output_featB, prepooled_tokensB, attn_weightsB = outputsB

        simmapA, simmapB, score = self.compute_spatial_similarity(prepooled_tokensA, output_featA,
                                                            prepooled_tokensB, output_featB)

        original_shape = (simmapA.shape[0], simmapA.shape[1])

        rolloutA = self.compute_rollout(attn_weightsA).cpu()
        simmapA = torch.matmul(rolloutA, torch.tensor(simmapA.flatten()).float())
        simmapA = simmapA.detach().numpy().reshape(original_shape)

        rolloutB = self.compute_rollout(attn_weightsB).cpu()
        simmapB = torch.matmul(rolloutB, torch.tensor(simmapB.flatten()).float())
        simmapB = simmapB.detach().numpy().reshape(original_shape)

        return simmapA, simmapB, score

    def show_cam_on_image(self, img: np.ndarray, mask: np.ndarray, use_rgb: bool = False,
                          colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.

        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :returns: The default image with the cam overlay.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        if np.max(img) > 1:
            raise Exception("The input image should np.float32 in the range [0, 1]")

        cam = heatmap + img
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
