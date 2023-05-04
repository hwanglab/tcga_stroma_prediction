###################################################################################################
# Example
# python generate_mask_from_WSIs.py 'slide input directory' 'prediction output directory' 'model path' 'norm stats'
# python generate_mask_from_WSIs.py /home/svs/ /home/predict/ model.h5 reinhardStats.csv
###################################################################################################

import os
import sys
import math
#import openslide
import tiffslide as openslide
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
from tqdm import tqdm
from openslide.deepzoom import DeepZoomGenerator
from reinhard import reinhard
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def slide_prediction(file_path, m_model, s_mu_lab, s_sigma_lab, out_path, s_name, t_size, b_scale): #t_size:256, b_scale:0.0078125
    # read slide
    print(file_path)
    slide = openslide.OpenSlide(file_path) 
    
    (width, height) = slide.level_dimensions[0]
    generator = DeepZoomGenerator(slide, tile_size=t_size, overlap=0, limit_bounds=True)
    highest_zoom_level = generator.level_count - 1
    # try:
    if not slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]==None:
        mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    else:
        mag = 10/slide.properties[openslide.PROPERTY_NAME_MPP_X]
    offset = math.floor((mag / 20) / 2)
    level = highest_zoom_level - offset
    # except KeyError:
    #     mag = None
    #     level = -104
    cols, rows = generator.level_tiles[level]
    validate = True
    if mag == 40:
        height_adj = round(height * b_scale)
        width_adj = round(width * b_scale)
        tile_size_adj = round(t_size * b_scale*2)
    elif mag == 20:
        height_adj = round(height * b_scale * 2)
        width_adj = round(width * b_scale * 2)
        tile_size_adj = round(t_size * b_scale * 2)
    else:
        height_adj = round(height * b_scale * 40/mag)
        width_adj = round(width * b_scale * 40/mag)
        tile_size_adj = round(t_size * b_scale * 40/mag)
        #validate = False
    # check validation for magnification
    if validate:
        im_tile_predict = np.zeros((height_adj, width_adj), 'uint8')
        for col in range(cols - 1):
            for row in range(rows - 1):
                tile = np.array(generator.get_tile(level, (col, row)))[:, :, :3]
                bn = np.sum(tile[:, :, 0] < 5) + np.sum(np.mean(tile, axis=2) > 245)
                if (np.std(tile[:, :, 0]) + np.std(tile[:, :, 1]) + np.std(
                        tile[:, :, 2])) / 3 > 18 and bn < t_size * t_size * 0.3:
                    img_norm = reinhard(tile, reference_mu_lab,
                                        reference_std_lab, src_mu=s_mu_lab,
                                        src_sigma=s_sigma_lab)
                    pred_prob = m_model.predict(np.expand_dims(img_norm, axis=0), verbose=0)
                    im_predict = Image.fromarray((255 * np.squeeze(pred_prob)).astype(np.uint8)) #256x256
                    resized_im_predict = im_predict.resize((tile_size_adj, tile_size_adj)) #4x4
                    im_tile_predict[row*tile_size_adj:row*tile_size_adj+tile_size_adj,
                    col*tile_size_adj:col*tile_size_adj+tile_size_adj] = np.asarray(resized_im_predict)
        img_slide = Image.fromarray(im_tile_predict)
        img_slide.save(out_path + s_name.split('.')[0] + '.png')


def main():

    # if len(sys.argv) != 5:
    #     print("Usage: ", sys.argv[0], "<path to the WSI directory> <path to the output directory> <model path> "
    #                                   "<color normalization stats file>")
    #     exit(1)


    input_dir ="/home/ext_jang_inyeop_mayo_edu/project/WSI/Pathology_Slides/Nanostring-GC_CTA"
    output_dir="/home/ext_jang_inyeop_mayo_edu/project/WSI/Pathology_Slides/Nanostring-GC_CTA/immune_subtype"
    model_path="/home/ext_jang_inyeop_mayo_edu/project/WSI/tcga_stroma_prediction/models/tumor.h5"
    norm_stats="/home/ext_jang_inyeop_mayo_edu/project/WSI/Pathology_Slides/Nanostring-GC_CTA/immune_subtype/reinhard_normalization_stat"

    #input_dir = sys.argv[1]
    #output_dir = sys.argv[2]
    #model_path = sys.argv[3]
    model_name=model_path.split('/')[-1].replace('.h5','')
    output_dir=output_dir+'/'+model_name+'/'
    os.makedirs(output_dir, exist_ok=True)
    #norm_stats = sys.argv[4]

    # set base scale and size
    tile_size = 256  # at 20x
    base_scale = 1 / 128.

    # set Reference mu and std for color normalization
    global reference_mu_lab, reference_std_lab
    reference_mu_lab = [8.63234435, -0.11501964, 0.03868433]
    reference_std_lab = [0.57506023, 0.10403329, 0.01364062]

    # read model
    model = tf.keras.models.load_model(model_path)

    # read mu and std stats from slides
    
    whole_slide_images = sorted(os.listdir(input_dir))
    for img_name in tqdm(whole_slide_images):
        if (not '.svs' in img_name): continue
        df = pd.read_csv(norm_stats +'/'+img_name.split('.')[0] + '.csv')
        print('output:', output_dir + img_name.split('.')[0] + '.png')
        #if os.path.exists(output_dir + img_name.split('.')[0] + '.png'): continue
        # find mu and std for each slide
        src_df = df.loc[df['slidename'] == img_name].to_numpy()[:, 1:].astype(np.float64) # TCGA 
        if len(src_df) != 0:
            src_mu_lab = src_df[0, :3]
            src_sigma_lab = src_df[0, 3:]
            slide_path = os.path.join(input_dir, img_name)
            slide_prediction(slide_path, model, src_mu_lab, src_sigma_lab,
                             output_dir, img_name, tile_size, base_scale)


if __name__ == "__main__":
    main()
