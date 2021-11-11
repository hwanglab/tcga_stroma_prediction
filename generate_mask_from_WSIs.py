###############################################################################
#  Copyright Sanghoon Lee leesan@marshall.edu
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Example
# python generate_mask_from_WSIs.py 'inputDir' 'outputDir' 'modelPath'

###############################################################################

import sys
import math
import large_image
import json
import numpy as np
import glob
import tensorflow as tf
import skimage.io as io

from reinhard import reinhard
from reinhard_stats import reinhard_stats

from compute_tile_foreground_fraction import compute_tile_foreground_fraction
from segment_wsi_foreground_at_low_res import segment_wsi_foreground_at_low_res

from scipy import ndimage
from skimage.transform import rescale


def predict_data(model, img_path, tile_position, it_kwargs,
reference_mu_lab, reference_std_lab, src_mu_lab, src_sigma_lab):
    # get slide tile source
    ts = large_image.getTileSource(img_path)
    # get scale for the tile and adjust centroids points
    ts_metadata = ts.getMetadata()
    # get requested tile information
    tile_info = \
        ts.getSingleTile(tile_position=tile_position,
                         format=large_image.tilesource.TILE_FORMAT_NUMPY,
                         **it_kwargs)
    # get global x and y positions, tile height and width
    im_tile = tile_info['tile'][:, :, :3]
    # perform color normalization
    im_tile = reinhard(im_tile, reference_mu_lab,
                                 reference_std_lab, src_mu=src_mu_lab,
                                 src_sigma=src_sigma_lab)
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3
    rows, cols = im_tile.shape[:2]
    min_var = 0.0015
    idx = 0
    im_predict = np.zeros((tile_info['height'], tile_info['width']), dtype=np.bool)
    for i in range(0, rows, IMG_HEIGHT):
        for j in range(0, cols, IMG_WIDTH):
            if (i+IMG_HEIGHT) > rows:
                continue
            elif (j+IMG_WIDTH) > cols:
                continue
            else:
                img = im_tile[i:i+IMG_HEIGHT, j:j+IMG_WIDTH, :IMG_CHANNELS]
            var = ndimage.variance(img)
            if var < min_var:
                continue
            preds_test = model.predict(np.expand_dims(img, axis=0))
            tile_preds = (preds_test > 0.5).astype(np.bool)
            im_predict[i:i+IMG_HEIGHT, j:j+IMG_WIDTH] = np.squeeze(tile_preds)
            idx += 1
    return im_predict

def main():

    if len(sys.argv) != 4:
    	print ("Usage: ", sys.argv[0], "<path to the WSI directory> <path to the output image directory> \
        <path to the model>")
    	exit(1)

    inputSlideDir = sys.argv[1]
    outputDir = sys.argv[2]
    modelPath = sys.argv[3]

    # set base scale and size
    baseSize = 4096
    baseScale = 1/64.
    initScale = 1/2.

    # find slides
    img_paths = glob.glob(inputSlideDir+'/*.svs')

    # set Reference mu and std for numalization
    reference_mu_lab = [8.63234435, -0.11501964, 0.03868433]
    reference_std_lab = [0.57506023, 0.10403329, 0.01364062]

    # read U-Net model
    model = tf.keras.models.load_model(modelPath)

    # set the range of slides
    for i in range(len(img_paths)):
        slide_name = img_paths[i].split('/')[-1].split('.')[0]
        print(slide_name)
        #
        # Read Input Image
        #
        print('\n>> Reading input image ... \n')
        ts = large_image.getTileSource(img_paths[i])
        ts_metadata = ts.getMetadata()
        # check magnification
        if ts_metadata['magnification']:
            print(json.dumps(ts_metadata, indent=2))
            # if ts_metadata['magnification'] is not None:
            im_fgnd_mask_lres, fgnd_seg_scale = segment_wsi_foreground_at_low_res(ts)
            it_kwargs = {
                'tile_size': {'width': baseSize},
                'scale': {'magnification': 20},
            }
            num_tiles = ts.getSingleTile(**it_kwargs)['iterator_range']['position']
            tile_fgnd_frac_list = compute_tile_foreground_fraction(
                img_paths[i], im_fgnd_mask_lres, fgnd_seg_scale,
                it_kwargs
            )
            min_fgnd_frac = 0.001
            num_fgnd_tiles = np.count_nonzero(tile_fgnd_frac_list >= min_fgnd_frac)
            percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles
            src_mu_lab, src_sigma_lab = reinhard_stats(
                img_paths[i], 0.01, magnification=ts_metadata['magnification'])
            print('\n>> Tile Processing ...\n')
            if ts_metadata['magnification'] == 20.0:
                n_tiles_width = math.ceil(ts_metadata['sizeX']/baseSize)
                n_tiles_height = math.ceil(ts_metadata['sizeY']/baseSize)
            else:
                n_tiles_width = math.ceil(ts_metadata['sizeX']*initScale/baseSize)
                n_tiles_height = math.ceil(ts_metadata['sizeY']*initScale/baseSize)
            im_row_predict = []
            im_predict = []
            is_first = True
            print('Total_tile_number = {}'.format(num_tiles))
            for tile in ts.tileIterator(**it_kwargs):
                tile_position = tile['tile_position']['position']
                tile_level_x = tile['tile_position']['level_x']
                tile_level_y = tile['tile_position']['level_y']
                print(tile_position, tile_level_x, tile_level_y)
                tile_info = \
                    ts.getSingleTile(tile_position=tile_position,
                                     format=large_image.tilesource.TILE_FORMAT_NUMPY,
                                     **it_kwargs)
                # get global x and y positions, tile height and width
                height, width, _ = tile_info['tile'][:, :, :3].shape
                if height < 64 or width < 64:
                    continue
                im_tile_predict = np.zeros((height, width), dtype=np.bool)
                im_tile_predict = rescale(im_tile_predict, baseScale, anti_aliasing=False)
                if tile_fgnd_frac_list[tile_position] > min_fgnd_frac:
                    tile_preds = predict_data(model, img_paths[i],
                    tile_position, it_kwargs, reference_mu_lab, reference_std_lab,
                    src_mu_lab, src_sigma_lab)
                    tile_preds = rescale(tile_preds, baseScale, anti_aliasing=False)
                    im_tile_predict = tile_preds
                if tile_position == 0:
                    im_row_predict = im_tile_predict
                elif tile_position%n_tiles_width == 0:
                    if is_first:
                        im_predict = im_row_predict
                        im_row_predict = im_tile_predict
                        is_first = False
                    else:
                        im_predict = np.vstack((im_predict, im_row_predict))
                        im_row_predict = im_tile_predict
                elif tile_position == (num_tiles-1):
                    im_row_predict = np.hstack((im_row_predict, im_tile_predict))
                    im_predict = np.vstack((im_predict, im_row_predict))
                else:
                    im_row_predict = np.hstack((im_row_predict, im_tile_predict))
            io.imsave(outputDir+'/'+slide_name+'.png', im_predict)


if __name__ == "__main__":
    main()
