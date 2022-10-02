import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.reducers import ClassWeightedReducer
from pytorch_metric_learning.utils import common_functions as c_f

class BatchClassWeightedReducer(ClassWeightedReducer):
    def element_reduction_helper(self, losses, indices, labels):
        self.weights = c_f.to_device(self.weights, losses, dtype=losses.dtype)
        weights = self.weights[labels[indices]]
        weights = weights.shape[0] * F.normalize(weights, dim=0)
        return torch.mean(losses * weights)

def norm(a):
    c = a.copy()
    c = c - np.min(c)
    max_val = np.max(c)
    c = c / max_val
    return c

def fig2array(fig):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image functio
    """
    # Draw figure on canvas
    fig.canvas.draw()
    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    img = np.swapaxes(img, 1, 2)
    return img