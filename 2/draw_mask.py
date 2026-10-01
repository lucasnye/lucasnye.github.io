import matplotlib.pyplot as plt
from skimage.draw import polygon
import numpy as np
from part2_1 import load_color_img

imA = load_color_img("./media/luke_skywalker.jpg")
H, W = imA.shape[:2]

plt.imshow(imA)
plt.title("Click points around your region, then close the window")
pts = plt.ginput(n=-1, timeout=0)   # click as many points as you want, press Enter when done
plt.close()

pts = np.array(pts)   # shape (N, 2), columns are (x, y)
rows = pts[:, 1]
cols = pts[:, 0]

rr, cc = polygon(rows, cols, shape=(H, W))
mask = np.zeros((H, W))
mask[rr, cc] = 1.0

plt.imsave("./media/custom_mask.png", mask, cmap="gray")