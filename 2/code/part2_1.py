import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io

def load_color_img(path):
    """Load a color image as float RGB in [0, 1], dropping any alpha channel."""
    img = io.imread(path)
    img = img[..., :3]              # keep R, G, B only
    return img.astype(np.float64) / 255.0

def gaussian_kernel(sigma):
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    g1d = cv2.getGaussianKernel(ksize, sigma)
    return g1d @ g1d.T

def impulse(ksize):
    """The identity filter (convolving with this returns unchanged image)"""
    d = np.zeros((ksize, ksize))
    d[ksize // 2, ksize // 2] = 1.0
    return d

def apply_per_channel(img, kernel):
    """Convolve each of the H x W x 3 channels separately, then restack."""
    out = np.zeros_like(img)
    for c in range(3):
        out[..., c] = convolve2d(img[..., c], kernel, mode="same", boundary="symm")
    return out   

def sharpen_two_steps(img, G, alpha):
    blur = apply_per_channel(img, G)
    detail = img - blur
    return img + alpha * detail

def sharpen_one_kernel(img, G, alpha):
    delta = impulse(G.shape[0])
    kernel = (1 + alpha) * delta - alpha * G
    return apply_per_channel(img, kernel)

def evaluate_sharpen(img, G, alpha):
    """
    Returns original, blurred, and re-sharpened versions for comparison.
    """
    blurred = apply_per_channel(img, G)
    blurred = np.clip(blurred, 0, 1)
    resharpened = sharpen_one_kernel(blurred, G, alpha)
    resharpened = np.clip(resharpened, 0, 1)
    return img, blurred, resharpened

if __name__ == "__main__":
    # img = load_color_img("./media/blur_sunset.jpeg")   # pick a colorful/detailed image for sharpening
    # sigma = 2
    # G = gaussian_kernel(sigma)

    # alpha = 5
    # img = sharpen_one_kernel(img, G, alpha)
    # # for alpha in [0.5, 1, 2, 3]:
    # #     s2 = sharpen_two_steps(img, G, alpha)
    # #     s1 = sharpen_one_kernel(img, G, alpha)
    # #     print(f"alpha={alpha}  match={np.allclose(s1, s2)}")
    # #     plt.imsave(f"./media/sharpen_alpha={alpha}.png", np.clip(s1, 0, 1))

    # plt.imsave("./media/sharpen_sunset.png", np.clip(img, 0, 1))

    # evaluation
    img = load_color_img("./media/sharp_photo.jpeg")   # pick a genuinely sharp/in-focus photo
    sigma = 5
    G = gaussian_kernel(sigma)
    alpha = 5

    original, blurred, resharpened = evaluate_sharpen(img, G, alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, title in zip(axes, [original, blurred, resharpened],
                              ["Original", "Blurred", "Re-sharpened"]):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("./media/sharpen_evaluation.png", dpi=150)

    plt.imsave("./media/eval_original.png", original)
    plt.imsave("./media/eval_blurred.png", blurred)
    plt.imsave("./media/eval_resharpened.png", resharpened)