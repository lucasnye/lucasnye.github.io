# part2_2.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io, color, transform
from align_image_code import align_images as align_pair

def load_gray_img(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    else:
        img = img / 255.0
    return img

def gaussian_kernel(sigma):
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    g1d = cv2.getGaussianKernel(ksize, sigma)
    return g1d @ g1d.T

def align_images(im1, im2, pts1, pts2):
    """pts1, pts2: [(x1,y1),(x2,y2)] two corresponding points in each image
    (e.g. two eyes). Crops/scales/translates im2 to align with im1. 2D grayscale in/out."""
    (x1a, y1a), (x1b, y1b) = pts1
    (x2a, y2a), (x2b, y2b) = pts2
    d1 = np.hypot(x1b - x1a, y1b - y1a)
    d2 = np.hypot(x2b - x2a, y2b - y2a)
    scale = d1 / d2
    im2_scaled = transform.rescale(im2, scale)
    x2a, y2a = x2a * scale, y2a * scale
    dy, dx = int(y1a - y2a), int(x1a - x2a)
    canvas = np.zeros(im1.shape)
    h, w = im2_scaled.shape
    y0, x0 = max(0, dy), max(0, dx)
    y1_, x1_ = min(im1.shape[0], dy + h), min(im1.shape[1], dx + w)
    sy0, sx0 = y0 - dy, x0 - dx
    canvas[y0:y1_, x0:x1_] = im2_scaled[sy0:sy0 + (y1_ - y0), sx0:sx0 + (x1_ - x0)]
    return canvas

def hybrid_image(im1, im2, sigma1, sigma2):
    """im1: low freq source, im2: high freq source. Both 2D grayscale, same shape, aligned."""
    G1 = gaussian_kernel(sigma1)
    G2 = gaussian_kernel(sigma2)
    low = convolve2d(im1, G1, mode="same", boundary="symm")
    high = im2 - convolve2d(im2, G2, mode="same", boundary="symm")
    hybrid = low + high
    return np.clip(hybrid, 0, 1), low, high

if __name__ == "__main__":
    im1 = load_gray_img("./media/DerekPicture.jpg")  # low-freq face
    im2 = load_gray_img("./media/nutmeg.jpg")  # high-freq face

    im1, im2 = align_pair(im1, im2)

    # If sizes/alignment already match (same crop), skip align_images.
    # Otherwise: im2 = align_images(im1, im2, [(x1,y1),(x2,y2)], [(x1',y1'),(x2',y2')])

    sigma1, sigma2 = 6, 14
    hybrid, low, high = hybrid_image(im1, im2, sigma1, sigma2)

    plt.imsave("./media/hybrid_d&n.png", hybrid, cmap="gray")
    plt.imsave("./media/hybrid_low_d&n.png", np.clip(low, 0, 1), cmap="gray")
    plt.imsave("./media/hybrid_high_d&n.png", np.clip(high + 0.5, 0, 1), cmap="gray")

    def show_fft(im, name):
        f = np.fft.fftshift(np.fft.fft2(im))
        mag = np.log(np.abs(f) + 1)
        plt.imsave(f"./media/fft_{name}.png", mag, cmap="gray")

    for im, name in [(im1, "im1"), (im2, "im2"), (low, "low"), (high, "high"), (hybrid, "hybrid")]:
        show_fft(im, name)