import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io, color

def load_image(image_path):
    img = io.imread(image_path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    else:
        img = img / 255.0
    
    return img

# Edge map: keep pixels whose gradient magnitude is above a threshold
def binarize(m, threshold):
    return (m > threshold).astype(np.float64)

def gradient_magnitude(img, Dx, Dy):
    # Partial derivatives
    # boundary="symm" mirrors the edges, which avoids the fake bright border with zero padding
    gx = convolve2d(img, Dx, mode="same", boundary="symm") 
    gy = convolve2d(img, Dy, mode="same", boundary="symm")

    # Calculate magnitude of gradient (i.e., how strong the edge is at each pixel)
    mag = np.sqrt(gx**2 + gy**2)

    return gx, gy, mag

# 4. Pick a threshold by comparing several
def best_threshold(thresholds: list, mag):
    print("magnitude range:", mag.min(), mag.max())
    # thresholds = [0.1, 0.2, 0.3, 0.4]   # adjust to your image's magnitude range
    fig, axes = plt.subplots(1, len(thresholds), figsize=(4 * len(thresholds), 4))
    for ax, t in zip(axes, thresholds):
        ax.imshow(binarize(mag, t), cmap="gray")
        ax.set_title(f"threshold = {t}")
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    Dx = np.array([[1, 0, -1]], dtype=np.float64)
    Dy = np.array([[1], [0], [-1]], dtype=np.float64)

    img = load_image("./media/cameraman.png")
    gx, gy, mag = gradient_magnitude(img, Dx, Dy)

    threshold = 0.25

    # plt.imsave("./media/gx.png", gx, cmap="gray")
    # plt.imsave("./media/gy.png", gy, cmap="gray")
    # plt.imsave("./media/mag.png", mag, cmap="gray")

    plt.imsave("./media/edges.png", binarize(mag, threshold), cmap="gray")