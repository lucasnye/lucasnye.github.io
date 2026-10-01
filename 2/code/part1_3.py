import cv2, numpy as np
from skimage import io, color
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from part1_2 import load_image, binarize, gradient_magnitude, best_threshold

if __name__ == "__main__":
    """Part 1: Blur then gradient magnitude"""
    sigma = 3 # pick the standard deviation first
    ksize = 2 * int(np.ceil(3 * sigma)) + 1

    gauss_1d = cv2.getGaussianKernel(ksize, sigma)
    gauss_2d = gauss_1d @ gauss_1d.T

    img = load_image("./media/cameraman.png")

    blur = convolve2d(img, gauss_2d, mode="same", boundary="symm")

    # plt.imsave(f"./media/cameraman_blur_sigma={sigma}.png", blur, cmap="gray")

    Dx = np.array([[1, 0, -1]], dtype=np.float64)
    Dy = np.array([[1], [0], [-1]], dtype=np.float64)

    gx, gy, mag = gradient_magnitude(blur, Dx, Dy)

    # plt.imsave("./media/gx_blur.png", gx, cmap="gray")
    # plt.imsave("./media/gy_blur.png", gy, cmap="gray")
    # plt.imsave("./media/mag_blur.png", mag, cmap="gray")

    # thresholds = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075]
    # best_threshold(thresholds, mag)
    best_thresh = 0.055

    # plt.imsave("./media/edges_blur.png", binarize(mag, best_thresh), cmap="gray")

    """Part 2: DoG filters"""
    # Build the DoG filters by convolving the gaussian with Dx / Dy
    # mode = "full" keeps the whole result, so nothing at the kernel border is cut off
    DoG_x = convolve2d(gauss_2d, Dx, mode="full") # shape (ksize, ksize + 2)
    DoG_y = convolve2d(gauss_2d, Dy, mode="full") # shape: (ksize + 2, ksize)

    # Apply each as one filter directly to the og image
    gx_DoG = convolve2d(img, DoG_x, mode="same", boundary="symm")
    gy_DoG = convolve2d(img, DoG_y, mode="same", boundary="symm")
    mag_dog = np.sqrt(gx_DoG**2 + gy_DoG**2)

    plt.imsave("./media/gx_DoG.png", gx_DoG, cmap="gray")
    plt.imsave("./media/gy_DoG.png", gy_DoG, cmap="gray")
    plt.imsave("./media/mag_DoG.png", mag_dog, cmap="gray")

    plt.imsave("./media/edges_DoG.png", binarize(mag_dog, best_thresh), cmap="gray")

    # Save the filters themselves for the writeup (stretched to fit the gray range)
    plt.imsave("./media/DoG_x_filter.png", DoG_x, cmap="gray")
    plt.imsave("./media/DoG_y_filter.png", DoG_y, cmap="gray")

    # Check that both methods agree
    m = ksize  # ignore a border wider than the kernel, where padding differs
    print("gx match:", np.allclose(gx[m:-m, m:-m], gx_DoG[m:-m, m:-m]))
    print("gy match:", np.allclose(gy[m:-m, m:-m], gy_DoG[m:-m, m:-m]))
    print("max abs diff (gx):", np.abs(gx[m:-m, m:-m] - gx_DoG[m:-m, m:-m]).max())