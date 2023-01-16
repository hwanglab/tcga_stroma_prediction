import os
import openslide
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from color_conversion import lab_mean_std
from simple_mask import simple_mask

def color_normalization(file_path, stride_):
    wsi = openslide.OpenSlide(file_path)
    (lrWidth, lrHeight) = wsi.level_dimensions[0]
    array_x = np.arange(0, lrWidth + 1, stride_)
    array_y = np.arange(0, lrHeight + 1, stride_)
    mesh_x, mesh_y = np.meshgrid(array_x, array_y)
    #####################################################
    # find sample pixels and perform a color nomoralization
    #####################################################
    sample_fraction = 0.01
    sample_pixels = []
    for i in range(mesh_x.shape[0] - 1):
        for j in range(mesh_x.shape[1] - 1):
            tile = wsi.read_region((int(mesh_x[i, j]), int(mesh_y[i, j])), 0, (stride_, stride_))
            tile = np.asarray(tile)
            tile = tile[:, :, :3]
            # bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile, axis=2) > 210)
            bn = np.sum(tile[:, :, 0] < 5) + np.sum(np.mean(tile,axis=2) > 250)
            if (np.std(tile[:, :, 0]) + np.std(tile[:, :, 1]) + np.std(tile[:, :, 2])) / 3 > 18 and bn < stride_ * stride_ * 0.1:
            # if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18:
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


if __name__ == "__main__":
cell_type = 'STAD'  # "GC_SM2_stmary", "STAD", "CLASSIC", "Stomach_Immunotherapy_stmary" "Stomach_Cancer_Stage4_Immunotherapy"
Stomach_Cancer_Stage4_Immunotherapy = False  # mpp = 0.21981612381243093
stride = 4096

if cell_type == 'GC_SM2_stmary': # mpp = 0.2521
    imagePath = ['/datasets/Gastric/stomach_cancer_immunotherapy/GC_SM2_stmary/']
    destPath = ['/datasets/Gastric/stomach_cancer_immunotherapy/GC_SM2_stmary_predict/']
    f_normal_stat = 'GC_SM2_stmary_reinhardStats_3.csv'
    wsi_ext = '.svs'
    # microns_per_tile = 258.2 # 1024x1024
    microns_per_tile = 1032.6016  # 4096x4096
elif cell_type == 'STAD': # mpp = 0.2527
    imagePath = ['/datasets/Gastric/STAD/svs/']
    destPath = ['/datasets/Gastric/STAD/predict/']
    f_normal_stat = 'STAD_reinhardStats.csv'
    wsi_ext = '.svs'
    # microns_per_tile = 258.2 # 1024x1024
    microns_per_tile = 1035.0592  # 4096x4096
elif cell_type == 'Stomach_Immunotherapy_stmary':  # mpp = 0.25
    imagePath = ['/datasets/Gastric/stomach_cancer_immunotherapy/Stomach_Immunotherapy_stmary/']
    destPath = ['/datasets/Gastric/stomach_cancer_immunotherapy/Stomach_Immunotherapy_stmary_predict/']
    df_normal_stat = 'Stomach_Immunotherapy_stmary_reinhardStats.csv'
    wsi_ext = '.tiff'
    # microns_per_tile = 258.2 # 1024x1024
    microns_per_tile = 1024  # 4096x4096
elif cell_type == 'CLASSIC': # mpp = 0.2532
    imagePath = ['/datasets/Gastric/CLASSIC/svs/201/']
    destPath = ['/datasets/Gastric/CLASSIC/predict/']
    f_normal_stat = 'CLASSIC_reinhardStats.csv'
    wsi_ext = '.svs'
    # microns_per_tile = 258.2 # 1024x1024
    microns_per_tile = 900.3668431357171  # 4096x4096
elif cell_type == 'GC': # mpp = 0.2532
    imagePath = ['/datasets/Gastric/']
    destPath = ['/datasets/Gastric/']
    f_normal_stat = 'GC_reinhardStats.csv'
    wsi_ext = '.tiff'

data = {
  "slidename": [], "mu1": [], "mu2": [], "mu3": [],
   "sigma1": [], "sigma2": [], "sigma3": []
}
df = pd.DataFrame(data)
idx = 0
for i in range(len(imagePath)):
    temp_imagePath = imagePath[i]
    dest_imagePath = destPath[i]
    wsis = sorted(os.listdir(temp_imagePath))
    for img_name in tqdm(wsis):
        if wsi_ext in img_name:
            if 'TCGA-CG-4301-01Z-00-DX1' in img_name:
                idx += 1
                print(img_name)
                file = temp_imagePath + img_name
                src_mu_lab,  src_sigma_lab = color_normalization(file, stride)
                print(img_name, src_mu_lab,  src_sigma_lab)
                df.loc[len(df.index)] = [img_name, src_mu_lab[0], src_mu_lab[1], src_mu_lab[2],
                src_sigma_lab[0], src_sigma_lab[1], src_sigma_lab[2]]
            if 'TCGA-CG-4438-01Z-00-DX1' in img_name:
                idx += 1
                print(img_name)
                file = temp_imagePath + img_name
                src_mu_lab,  src_sigma_lab = color_normalization(file, stride)
                print(img_name, src_mu_lab,  src_sigma_lab)
                df.loc[len(df.index)] = [img_name, src_mu_lab[0], src_mu_lab[1], src_mu_lab[2],
                src_sigma_lab[0], src_sigma_lab[1], src_sigma_lab[2]]



# df.to_csv('GC_reinhardStats.csv', index=False)
# df.to_csv('GC_SM2_stmary_reinhardStats_2.csv', index=False)
# df.to_csv('Stomach_Immunotherapy_stmary_reinhardStats.csv', index=False)
# [ 9.26211324 -0.07591264  0.01077722] [0.25048092 0.09867779 0.01163571]
# TCGA-CG-4301-01Z-00-DX1.d3a38b46-9f9c-4ed4-a1c7-fc1e48263a53.svs
# TCGA-CG-4438-01Z-00-DX1.3691c587-cadf-4523-a332-5a18f52c94b0.svs
