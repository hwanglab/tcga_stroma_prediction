###########################################################################################
# Example
# python normalization.py 'input directory' 'output file name' 'tile size' (default: 4096)
# python normalization.py /home/svs/ reinhardStats.csv 4096
###########################################################################################

import os
import sys
import openslide
#import tiffslide as openslide
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from color_conversion import lab_mean_std
from simple_mask import simple_mask
from PIL import Image
import multiprocessing as mp
from glob import glob

global wsi
global mesh_x, mesh_y
global stride

def thread_read(idx):
    i,j = idx
    global wsi
    global mesh_x, mesh_y
    global stride
    
    ppt_x = int(stride)
    ppt_y = int(stride)
    #print('before read')
    tile = wsi.read_region((int(mesh_x[i, j]), int(mesh_y[i, j])), 0, (ppt_x, ppt_y))
    #tile.save('test.png')
    #exit()
    #tile = Image.open('test.png')
    sample_fraction = 0.01 # default sample fraction 0.01
    tile = np.asarray(tile)
    tile = tile[:, :, :3]
    bn = np.sum(tile[:, :, 0] < 5) + np.sum(np.mean(tile,axis=2) > 250) # set to 250
    if (np.std(tile[:, :, 0]) + np.std(tile[:, :, 1]) + np.std(tile[:, :, 2])) / 3 > 18 \
            and bn < ppt_x * ppt_y * 0.1:
        
        try:
            im_fgnd_mask_lres = simple_mask(tile)
        except:
            return -1
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
        return tile_pix_rgb[sample_ind, :]

def color_normalization(file_path, stride_):
    global wsi
    global mesh_x, mesh_y
    global stride 
    stride=stride_
    # read whole slide images; openslide version; runnting time can be improved by DeepZoomGenerator


    wsi = openslide.OpenSlide(file_path)
    #wsi.get_thumbnail(wsi.level_dimensions[-1]).save(file_path.replace(file_path[-5:].split('.')[-1],'png')) #save thumbnail

    (lrWidth, lrHeight) = wsi.level_dimensions[0]

    # ppt_x = int(stride_ / float(wsi.properties[openslide.PROPERTY_NAME_MPP_X]))
    # ppt_y = int(stride_ / float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y]))
    # array_x = np.arange(0, lrWidth + 1, ppt_x)
    # array_y = np.arange(0, lrHeight + 1, ppt_y)
    # ppt_x = int(stride_)
    # ppt_y = int(stride_)
    array_x = np.arange(0, lrWidth + 1, stride_)
    array_y = np.arange(0, lrHeight + 1, stride_)

    mesh_x, mesh_y = np.meshgrid(array_x, array_y)

    # find sample pixels and perform a color normalization
    
    sample_pixels = []
    #print(file_path)
    cnt=0
    indices =[]
    for i in range(mesh_x.shape[0] - 1):
        for j in range(mesh_x.shape[1] - 1):
            indices.append([i,j])


    with mp.pool.Pool() as pool:
        result = pool.map_async(thread_read, indices)
        result.wait()
        pool.close()
        pool.join()
        for i, it in enumerate(result.get()):
            if isinstance(it,np.ndarray):
                # add rgb triplet of sample pixels
                sample_pixels.append(it)    #sample_pixels.append(tile_pix_rgb[sample_ind, :])


                
                
    sample_pixel = np.concatenate(sample_pixels, 0)
    # reshape the Nx3 pixel array into a 1 x N x 3 image for lab_mean_std
    sample_pixels_rgb = np.reshape(sample_pixel,
                                   (1, sample_pixel.shape[0], 3))
    # compute mean and stddev of sample pixels in Lab space
    mu, sigma = lab_mean_std(sample_pixels_rgb)
    # build named tuple for output
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    src_mu_lab_out,  src_sigma_lab_out = ReinhardStats(mu, sigma)
    #print('mu sigma',mu, sigma)
    return src_mu_lab_out,  src_sigma_lab_out
 
def main(slides_list):
    print("Num of cpu:", mp.cpu_count()) # 48
    #exit()
    print(sys.argv)
    # read input argument
    # if len(sys.argv) != 3:
    #     print ("Usage: ", sys.argv[0], "<path to the WSI directory> <path to the output file> <tile size>")
    #     exit(1)


    stride = 256
    # create pandas dataframe to be stored in a file


    # read whole slide image files
    
    for slide_path in tqdm(slides_list):   
        
        outdir = '/'.join(slide_path.split('/')[:-1])+'/immune_subtype/reinhard_normalization_stat'
        img_name = slide_path.split('/')[-1] 
        os.makedirs(outdir, exist_ok=True)
        
        data = {
        "slidename": [], "mu1": [], "mu2": [], "mu3": [], "sigma1": [], "sigma2": [], "sigma3": []
        }
        df = pd.DataFrame(data)
        
        if "ipynb" in slide_path:
            continue
        try:
            src_mu_lab,  src_sigma_lab = color_normalization(slide_path, int(stride))
        except Exception as e:
            print(e)
            print(img_name)
            print("-------------------------")
            continue
        print(img_name, src_mu_lab,  src_sigma_lab)
        df.loc[len(df.index)] = [img_name, src_mu_lab[0], src_mu_lab[1], src_mu_lab[2],
                                 src_sigma_lab[0], src_sigma_lab[1], src_sigma_lab[2]]

        # save pandas dataframe
        
        df.to_csv("{0}/{1}.csv".format(outdir, img_name.replace(('.'+img_name[-5:].split(".")[-1]),'')), mode='a', index=False)

'''
01/20/2023
Wrong Slides List:
MSG: (need at least one array to concatenate)
2_S_34388532_0.svs 
1_S_38766901_0.svs
1_S_38891465_0.svs
1_S_20189813_0.svs
'''
if __name__ == "__main__":
    whole_slide_images = sorted(glob("/home/m264377/Downloads/new/Stomach_Cancer_Stage4_Immunotherapy/*.svs"))

    main(whole_slide_images)