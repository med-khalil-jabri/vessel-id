import numpy as np


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