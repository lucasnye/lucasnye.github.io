import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

def pad_image(image, ph, pw):
    """Zero-pad image by ph rows and pw columns"""
    H, W = image.shape
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=image.dtype)
    padded[ph:ph + H, pw:pw + W] = image
    return padded

def _pad_for_mode(image, kh, kw, mode):
    """
    full:
    - (kh-1) rows, (kw-1) columns per side
    - output size: (H + kh - 1) x (W + kw - 1)
    same: (kh//2) rows, (kw//2) columns per side
    
    valid: no padding
    """
    if mode == "full":
        return pad_image(image, kh - 1, kw - 1)
    if mode == "same":
        return pad_image(image, kh // 2, kw // 2)
    if mode == "valid":
        return image
    raise ValueError("mode must be 'full, 'same', or 'valid'")

def conv_4loops(image, kernel, mode="same"):
    image = image.astype(np.float64)
    kh, kw = kernel.shape
    kflip = kernel[::-1, ::-1] # convolution flips the kernel
    padded = _pad_for_mode(image, kh, kw, mode)
    out_h = padded.shape[0] - kh + 1
    out_w = padded.shape[1] - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):          # 1. which output row
        for j in range(out_w):      # 2. which output column
            for u in range(kh):     # 3. which row inside the kernel window
                for v in range(kw): # 4. which column inside the kernel window
                    out[i, j] += padded[i + u, j + v] * kflip[u, v]
    
    return out

def conv_2loops(image, kernel, mode="same"):
    image = image.astype(np.float64)
    kh, kw = kernel.shape
    kflip = kernel[::-1, ::-1]
    padded = _pad_for_mode(image, kh, kw, mode)
    out_h = padded.shape[0] - kh + 1
    out_w = padded.shape[1] - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kflip)
    
    return out

box_filter = np.ones((9, 9)) / 81.0
D_x = np.array([[1, 0, -1]], dtype=np.float64)
D_y = np.array([[1], [0], [-1]], dtype=np.float64)

# Compare against scipy
if __name__ == "__main__":
    img = io.imread("selfie.jpeg")
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3]) # keep everything in the first axes, but only the first 3 channels in the last axis (e.g., drop alpha channel)
    else:
        img /= 255.0

    blur = conv_2loops(img, box_filter, "same")
    gx = conv_2loops(img, D_x, "same")
    gy = conv_2loops(img, D_y, "same")

    plt.imsave("./media/blur.png", blur, cmap="gray")
    plt.imsave("./media/dx.png", gx, cmap="gray")
    plt.imsave("./media/dy.png", gy, cmap="gray")