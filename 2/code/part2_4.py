# part2_4.py
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon, ellipse
from part2_1 import load_color_img
from part2_3 import gaussian_stack, laplacian_stack
from skimage import io, transform

def blend(imA, imB, mask, sigma=2, levels=5):
    """mask: same H,W as imA/imB, values in [0,1], 1 = show imA, 0 = show imB."""
    mask3 = np.repeat(mask[:, :, None], 3, axis=2) if mask.ndim == 2 else mask

    LA = laplacian_stack(gaussian_stack(imA, sigma, levels))
    LB = laplacian_stack(gaussian_stack(imB, sigma, levels))
    GR = gaussian_stack(mask3, sigma, levels)   # blur the mask too -> smooth transition

    blended_stack = [GR[i] * LA[i] + (1 - GR[i]) * LB[i] for i in range(levels)]
    result = sum(blended_stack)
    return np.clip(result, 0, 1)

if __name__ == "__main__":
    # apple = load_color_img("./media/chicken.jpg")
    # orange = load_color_img("./media/trex.jpg")

    # # vertical half-and-half mask for the oraple
    # H, W = apple.shape[:2]
    # mask = np.zeros((H, W))
    # mask[:, :W // 2] = 1.0

    # oraple = blend(apple, orange, mask, sigma=4, levels=5)
    # plt.imsave("./media/chickrex.png", oraple)

    # irregular mask example: load a hand-drawn mask (black/white PNG) or build one with a circle
    imA = load_color_img("./media/trex.jpg")
    imB = load_color_img("./media/landscape.jpeg")
    H, W = imA.shape[:2]
    # yy, xx = np.mgrid[0:H, 0:W]
    # cy, cx, r = H // 2, W // 2, min(H, W) // 4
    # circle_mask = ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2).astype(np.float64)
    # circle_blend = blend(imA, imB, circle_mask, sigma=4, levels=5)
    # plt.imsave("./media/dino_rainbow.png", circle_blend)

    # # your own blend: swap in two of your own photos + a custom mask
    
    mask = np.zeros((H, W))

    # Ellipse example (fast, looks natural for faces/fruit/etc.)
    rr, cc = ellipse(H // 2, W // 2, H // 3, W // 4)  # center_y, center_x, radius_y, radius_x

    mask[rr, cc] = 1.0
    irregular_blend = blend(imA, imB, mask, sigma=5, levels=5)
    plt.imsave("./media/dino_rainbow.png", irregular_blend)
    # # plt.imsave("./media/ellipse_mask.png", mask, cmap="gray")  # save the mask too, spec wants to see it

    imA = load_color_img("./media/luke_skywalker.jpg")
    imB = load_color_img("./media/middle_earth.jpeg")
    imB = transform.resize(imB, imA.shape[:2], anti_aliasing=True)
    H, W = imA.shape[:2]

    mask = io.imread("./media/custom_mask.png")
    if mask.ndim == 3:
        mask = mask[..., 0]        # take one channel if it saved as RGB
    mask = (mask / 255.0 > 0.5).astype(np.float64)   # binarize
    result = blend(imA, imB, mask, sigma=5, levels=5)
    plt.imsave("./media/luke_middle_earth.png", result)