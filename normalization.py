###########################################################################################
# Example
# python normalization.py 'input directory' 'output file name' 'tile size' (default: 4096)
# python normalization.py /home/svs/ reinhardStats.csv 4096
###########################################################################################

import os
import sys
import openslide
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from color_conversion import lab_mean_std
from simple_mask import simple_mask


def color_normalization(file_path, stride_):
    # read whole slide images; openslide version; runnting time can be improved by DeepZoomGenerator
    wsi = openslide.OpenSlide(file_path)
    (lrWidth, lrHeight) = wsi.level_dimensions[0]
    array_x = np.arange(0, lrWidth + 1, stride_)
    array_y = np.arange(0, lrHeight + 1, stride_)
    mesh_x, mesh_y = np.meshgrid(array_x, array_y)

    # find sample pixels and perform a color normalization
    sample_fraction = 0.01 # default sample fraction 0.01
    sample_pixels = []

    for i in range(mesh_x.shape[0] - 1):
        for j in range(mesh_x.shape[1] - 1):
            tile = wsi.read_region((int(mesh_x[i, j]), int(mesh_y[i, j])), 0, (stride_, stride_))
            tile = np.asarray(tile)
            tile = tile[:, :, :3]
            bn = np.sum(tile[:, :, 0] < 5) + np.sum(np.mean(tile,axis=2) > 250) # set to 250
            if (np.std(tile[:, :, 0]) + np.std(tile[:, :, 1]) + np.std(tile[:, :, 2])) / 3 > 18 \
                    and bn < stride_ * stride_ * 0.1:
                im_fgnd_mask_lres = simple_mask(tile)
                # generate linear indices of sample pixels in fgnd mask
                nz_ind = np.nonzero(im_fgnd_mask_lres.flatten())[0]
                # Handle fractions in the desired sample size by rounding up
                # or down, weighted by the fractional amount.
                float_samples = sample_fraction * nz_ind.size
                num_samples = int(np.floor(float_samples))
                num_samples += np.random.binomial(1, float_samples - num_samples)
                sample_ind = np.random.choice(nz_ind, num_samples)
                # convert rgb tile image to Nx3 array
                tile_pix_rgb = np.reshape(tile, (-1, 3))
                # add rgb triplet of sample pixels
                sample_pixels.append(tile_pix_rgb[sample_ind, :])

    sample_pixel = np.concatenate(sample_pixels, 0)
    # reshape the Nx3 pixel array into a 1 x N x 3 image for lab_mean_std
    sample_pixels_rgb = np.reshape(sample_pixel,
                                   (1, sample_pixel.shape[0], 3))
    # compute mean and stddev of sample pixels in Lab space
    mu, sigma = lab_mean_std(sample_pixels_rgb)
    # build named tuple for output
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    src_mu_lab_out,  src_sigma_lab_out = ReinhardStats(mu, sigma)
    return src_mu_lab_out,  src_sigma_lab_out


def main():
    # read input argument
    if len(sys.argv) != 3:
        print ("Usage: ", sys.argv[0], "<path to the WSI directory> <path to the output file> <tile size>")
        exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    stride = sys.argv[3]

    # create pandas dataframe to be stored in a file
    data = {
      "slidename": [], "mu1": [], "mu2": [], "mu3": [], "sigma1": [], "sigma2": [], "sigma3": []
    }
    df = pd.DataFrame(data)

    # read whole slide image files
    whole_slide_images = sorted(os.listdir(input_dir))
    for img_name in tqdm(whole_slide_images):
        slide_path = input_dir + img_name
        src_mu_lab,  src_sigma_lab = color_normalization(slide_path, int(stride))
        print(img_name, src_mu_lab,  src_sigma_lab)
        df.loc[len(df.index)] = [img_name, src_mu_lab[0], src_mu_lab[1], src_mu_lab[2],
                                 src_sigma_lab[0], src_sigma_lab[1], src_sigma_lab[2]]

    # save pandas dataframe
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
