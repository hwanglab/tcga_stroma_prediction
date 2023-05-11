#python -m pip install -U imagecodecs
import os
import numpy as np
import czifile as czi
import tensorflow as tf
import pandas as pd
from PIL import Image
from tqdm import tqdm
from reinhard import reinhard

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def slide_prediction(f_path, m_tumor, m_stroma, m_tils, s_mu, s_sigma, d_path, im_name, t_size, b_scale):
    # open image
    czi_obj = czi.CziFile(f_path)
    c_wsi = np.zeros(czi_obj.shape[2:], np.uint8)
    for idx, directory_entry in enumerate(czi_obj.filtered_subblock_directory):
        sub_block = directory_entry.data_segment()
        czi_tile = sub_block.data(resize=True, order=1)
        xs = directory_entry.start[2] - czi_obj.start[2]
        xe = xs + czi_tile.shape[2]
        ys = directory_entry.start[3] - czi_obj.start[3]
        ye = ys + czi_tile.shape[3]
        c_wsi[xs:xe, ys:ye, :] = czi_tile.squeeze()
    height, width, _ = c_wsi.shape
    height_adj = round(height * b_scale)
    width_adj = round(width * b_scale)
    stride_adj = round(t_size * b_scale)
    slide_predict = np.zeros((height_adj, width_adj, 3), 'uint8')
    heights = np.arange(0, height + 1, t_size)
    widths = np.arange(0, width + 1, t_size)
    heights, widths = np.meshgrid(heights, widths)
    for i in range(heights.shape[0] - 1):
        for j in range(heights.shape[1] - 1):
            tile = c_wsi[int(heights[i, j]):int(heights[i, j])+t_size, int(widths[i, j]):int(widths[i, j])+t_size, :]
            bn = np.sum(tile[:, :, 0] < 5) + np.sum(np.mean(tile,axis=2) > 210)
            if (np.std(tile[:, :, 0]) + np.std(tile[:, :, 1]) + np.std(tile[:, :, 2])) / 3 > 18 and \
                    bn < t_size * t_size * 0.3:
                im_predict = np.zeros((t_size, t_size, 3), 'float')
                img_norm = reinhard(tile, reference_mu_lab, reference_std_lab, src_mu=s_mu,
                                    src_sigma=s_sigma)
                r = m_tils.predict(np.expand_dims(img_norm, axis=0), verbose=0)
                g = m_stroma.predict(np.expand_dims(img_norm, axis=0), verbose=0)
                b = m_tumor.predict(np.expand_dims(img_norm, axis=0), verbose=0)
                # tile_preds = (pred1 > 0.5).astype(np.bool)
                im_predict[:, :, 0] = np.squeeze(r)
                im_predict[:, :, 1] = np.squeeze(g)
                im_predict[:, :, 2] = np.squeeze(b)
                im_predict = Image.fromarray((255*im_predict).astype(np.uint8), "RGB")
                resized_im_predict = im_predict.resize((stride_adj, stride_adj))
                i_adj = int(heights[i, j]*b_scale)
                j_adj = int(widths[i, j]*b_scale)
                slide_predict[i_adj:i_adj+stride_adj,j_adj:j_adj+stride_adj] = np.asarray(resized_im_predict)
    img_slide = Image.fromarray(slide_predict)
    img_slide.save(d_path + im_name.split('.')[0] + '.png')


if __name__ == '__main__':
    stride = 256
    base_scale = 1/128.
    wsi_ext='.czi'
    # microns_per_tile = 258.2 # 1024x1024
    microns_per_tile = 900.3668431357171 # 4096x4096

    # source_dir = [
    #     '/datasets/stomach_cancer_immunotherapy/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/',
    #     '/datasets/stomach_cancer_immunotherapy/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']
    source_dir = ['/datasets/stomach_cancer_immunotherapy/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/']
    # source_dir = ['/datasets/stomach_cancer_immunotherapy/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']
    dst_dir = ['/datasets/stomach_cancer_immunotherapy/Stomach_Cancer_Stage4_Immunotherapy_predict_40x/']

    global reference_mu_lab, reference_std_lab
    reference_mu_lab = [8.63234435, -0.11501964, 0.03868433]
    reference_std_lab = [0.57506023, 0.10403329, 0.01364062]
    df = pd.read_csv('/home/leesan/workspace/illumination-preserving-rotations'
                     '/Stomach_Cancer_Stage4_Immunotherapy_reinhardStats.csv')

    # load model
    modelPath_tumor = '/home/leesan/workspace/illumination-preserving-rotations/tumor_256_256_40x.h5'
    modelPath_stroma = '/home/leesan/workspace/illumination-preserving-rotations/stroma_256_256_40x.h5'
    modelPath_tils = '/home/leesan/workspace/illumination-preserving-rotations/lymphocytic_infiltrate_256_256_40x.h5'
    model_tumor = tf.keras.models.load_model(modelPath_tumor)
    model_stroma = tf.keras.models.load_model(modelPath_stroma)
    model_tils = tf.keras.models.load_model(modelPath_tils)

    idx = 0
    for i in range(len(source_dir)):
        wsi_path = source_dir[i]
        dst_path = dst_dir[0]
        wsi_paths = sorted(os.listdir(wsi_path))
        for img_name in tqdm(wsi_paths):
            if wsi_ext in img_name:
                src_df = df.loc[df['slidename'] == img_name].to_numpy()[:, 1:].astype(np.float64)
                if len(src_df) != 0:
                    idx += 1
                    if idx >= 23:
                        print(idx, img_name)
                        src_mu_lab = src_df[0, :3]
                        src_sigma_lab = src_df[0, 3:]
                        file = wsi_path + img_name
                        slide_prediction(file, model_tumor, model_stroma, model_tils, src_mu_lab, src_sigma_lab, dst_path,
                                         img_name, stride, base_scale)
