# part2_3.py
import numpy as np
import matplotlib.pyplot as plt
from part2_1 import load_color_img, apply_per_channel
from part2_2 import gaussian_kernel

def gaussian_stack(img, sigma, levels):
    """Same size at every level (no downsampling), sigma grows each level."""
    stack = [img]
    for i in range(1, levels):
        G = gaussian_kernel(sigma * (2 ** (i - 1)))   # widen the blur each level
        stack.append(apply_per_channel(stack[-1], G))
    return stack

def laplacian_stack(gstack):
    lstack = [gstack[i] - gstack[i + 1] for i in range(len(gstack) - 1)]
    lstack.append(gstack[-1])   # last level = final Gaussian level (so it all sums back)
    return lstack

if __name__ == "__main__":
    img = load_color_img("./media/apple.jpeg")
    levels = 5
    gstack = gaussian_stack(img, sigma=2, levels=levels)
    lstack = laplacian_stack(gstack)

    for i, (g, l) in enumerate(zip(gstack, lstack)):
        plt.imsave(f"./media/gstack_{i}.png", np.clip(g, 0, 1))
        if i == len(lstack) - 1:
            plt.imsave(f"./media/lstack_{i}.png", np.clip(l, 0, 1))        # last level: no offset
        else:
            plt.imsave(f"./media/lstack_{i}.png", np.clip(l + 0.5, 0, 1))  # difference levels: offset to visualize

    # Sanity check: stack should sum back to original
    print(np.allclose(lstack[-1], gstack[-1]))
    recon = sum(lstack)
    print("reconstruction max error:", np.abs(recon - img).max())