import numpy as np
import skimage as sk
import skimage.io as skio

def crop_border(img, pct=0.15):
    h, w = img.shape[:2]
    dy, dx = int(h * pct), int(w * pct)
    return img[dy:h-dy, dx:w-dx]

def downsample(img):
    return img[::2, ::2]

def euclidean(img1, img2):
    return np.sqrt(np.sum((img1 - img2) ** 2)) # or np.linalg.norm(img1 - img2)

def ncc(img1, img2):
    img1_norm = (img1-np.mean(img1)) / np.linalg.norm(img1-np.mean(img1))
    img2_norm = (img2-np.mean(img2)) / np.linalg.norm(img2-np.mean(img2))
    return np.sum(img1_norm * img2_norm)

def exhaustive_search(ref, to_align, score_fn, better, window=(-15, 15)):
    best_score = None
    best_shift = (0, 0)
    for dx in range(window[0], window[1] + 1):
        for dy in range(window[0], window[1] + 1):
            candidate = np.roll(np.roll(to_align, dx, axis=1), dy, axis=0)
            score = score_fn(ref, candidate)
            if best_score is None or better(score, best_score):
                best_score = score
                best_shift = (dx, dy)
    
    return best_shift

def pyramid(ref, to_align, score_fn, better, window=(-2, 2), min_size=400, base_window=(-20, 20)):
    h, w = ref.shape[:2]

    if max(h, w) < min_size:
        return exhaustive_search(ref, to_align, score_fn, better, base_window)

    ref_small = downsample(ref)
    to_align_small = downsample(to_align)

    dx_small, dy_small = pyramid(ref_small, to_align_small, score_fn, better, window, min_size, base_window)

    dx_est, dy_est = dx_small * 2, dy_small * 2

    best_score = None
    best_shift = (dx_est, dy_est)
    for ddx in range(window[0], window[1] + 1):
        for ddy in range(window[0], window[1] + 1):
            dx, dy = dx_est + ddx, dy_est + ddy
            candidate = np.roll(np.roll(to_align, dx, axis=1), dy, axis=0)
            score = score_fn(ref, candidate)
            if best_score is None or better(score, best_score):
                best_score = score
                best_shift = (dx, dy)

    return best_shift

def main():
    for name in ['master-pnp-prok-00600-00657a', 'master-pnp-prok-00900-00998a', 'master-pnp-prok-01000-01027a']:
        # name of the input file
        imname = f'data/{name}.tif'

        # read in the image
        im = skio.imread(imname)

        # convert to double (might want to do this later on to save memory)    
        im = sk.img_as_float(im)

        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(int)

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        l2_better = lambda new, best: new < best
        ncc_better = lambda new, best : new > best

        b_c, g_c, r_c = crop_border(b), crop_border(g), crop_border(r)

        dx_g_l2, dy_g_l2 = pyramid(b_c, g_c, euclidean, l2_better)
        dx_r_l2, dy_r_l2 = pyramid(b_c, r_c, euclidean, l2_better)

        dx_g_ncc, dy_g_ncc = pyramid(b_c, g_c, ncc, ncc_better)
        dx_r_ncc, dy_r_ncc = pyramid(b_c, r_c, ncc, ncc_better)

        # l2
        ag_l2 = np.roll(np.roll(g, dx_g_l2, axis=1), dy_g_l2, axis=0)
        ar_l2 = np.roll(np.roll(r, dx_r_l2, axis=1), dy_r_l2, axis=0)

        # ncc
        ag_ncc = np.roll(np.roll(g, dx_g_ncc, axis=1), dy_g_ncc, axis=0)
        ar_ncc = np.roll(np.roll(r, dx_r_ncc, axis=1), dy_r_ncc, axis=0)

        # create a color image (l2)
        im_out_l2 = np.dstack([ar_l2, ag_l2, b])
        im_out_l2 = sk.img_as_ubyte(np.clip(im_out_l2, 0, 1))

        # create a color image (ncc)
        im_out_ncc = np.dstack([ar_ncc, ag_ncc, b])
        im_out_ncc = sk.img_as_ubyte(np.clip(im_out_ncc, 0, 1))

        # save the image
        fname_l2 = f'out_path/{name}_l2.jpg'
        fname_ncc = f'out_path/{name}_ncc.jpg'

        skio.imsave(fname_l2, im_out_l2)
        skio.imsave(fname_ncc, im_out_ncc)

if __name__ == "__main__":
    main()