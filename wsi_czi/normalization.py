#python -m pip install -U imagecodecs
import os
import numpy as np
import collections
import pandas as pd
import czifile as czi
from tqdm import tqdm
from PIL import Image
from color_conversion import lab_mean_std
from simple_mask import simple_mask
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def color_normalization(f_path, t_size, t_point):
    # open image
    czi_obj = czi.CziFile(f_path)
    c_wsi = np.zeros(czi_obj.shape[2:], np.uint8)
    for idx, directory_entry in enumerate(czi_obj.filtered_subblock_directory):
        sub_block = directory_entry.data_segment()
        czi_tile = sub_block.data(resize=False, order=0)
        xs = directory_entry.start[2] - czi_obj.start[2]
        xe = xs + czi_tile.shape[2]
        ys = directory_entry.start[3] - czi_obj.start[3]
        ye = ys + czi_tile.shape[3]
        c_wsi[xs:xe, ys:ye, :] = czi_tile.squeeze()
    height, width, _ = c_wsi.shape
    X = np.arange(0, height + 1, t_size)
    Y = np.arange(0, width + 1, t_size)
    X, Y = np.meshgrid(X, Y)
    #####################################################
    # find sample pixels and perform color normalization
    #####################################################
    sample_fraction = 0.01
    sample_pixels = []
    sample_limit = 20
    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1] - 1):
            tile = c_wsi[int(X[i, j]):int(X[i, j])+t_size, int(Y[i, j]):int(Y[i, j])+t_size]
            bn = np.sum(tile[:, :, 0] < 5) + np.sum(np.mean(tile, axis=2) > 210)
            if (np.std(tile[:, :, 0]) + np.std(tile[:, :, 1]) + np.std(tile[:, :, 2])) / 3 > 18 and bn < t_size * \
                    t_size * t_point:
                sample_limit -= 1
                if sample_limit > 0:
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
    Mu, Sigma = lab_mean_std(sample_pixels_rgb)
    # build named tuple for output
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    czi_src_mu_lab, czi_src_sigma_lab = ReinhardStats(Mu, Sigma)
    return czi_src_mu_lab,  czi_src_sigma_lab


if __name__ == '__main__':
    stride = 1024
    t_point = 0.1
    wsi_ext = '.czi'
    source_dir = [
        '/datasets/stomach_cancer_immunotherapy/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/',
        '/datasets/stomach_cancer_immunotherapy/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']
    dst_path = '/home/leesan/workspace/illumination-preserving-rotations/'
    data = {
      "slidename": [], "mu1": [], "mu2": [], "mu3": [],
       "sigma1": [], "sigma2": [], "sigma3": []
    }
    df = pd.DataFrame(data)
    for p in range(len(source_dir)):
        wsi_path = source_dir[p]
        wsi_paths = sorted(os.listdir(wsi_path))
        for img_name in tqdm(wsi_paths):
            if wsi_ext in img_name:
                print(img_name)
                file_path = wsi_path + img_name
                # if img_name == 'SS1523249.czi':
                #     t_point = 0.3
                # else:
                #     t_point = 0.1
                if img_name == 'SS1262241.czi':
                    src_mu_lab = [7.92273816, -0.34644195, 0.06128472]
                    src_sigma_lab = [0.91207587, 0.23390772, 0.04346374]

                else:
                    src_mu_lab,  src_sigma_lab = color_normalization(file_path, stride, t_point)
                print(img_name, src_mu_lab,  src_sigma_lab)
                df.loc[len(df.index)] = [img_name, src_mu_lab[0], src_mu_lab[1], src_mu_lab[2], src_sigma_lab[0],
                                         src_sigma_lab[1], src_sigma_lab[2]]

    df.to_csv(dst_path+'Stomach_Cancer_Stage4_Immunotherapy_reinhardStats.csv', index=False)
