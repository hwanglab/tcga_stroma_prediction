import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import dataset
import os, fnmatch
import openslide
from math import ceil, floor
from scipy import ndimage as ndi
import pandas as pd
from pathlib import Path
from torch.autograd import Variable

from skimage import transform, draw
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, opening,closing, square

import multiprocessing as mp
import argparse
import normalizer as norm
import time

import staintools

import gc
import glob

from tqdm import tqdm

BATCH_SIZE = 1800

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
nb_classes = 2
cores = 30


label_name = ["B","BD","F","H","I","M","MF","N","T"]
class_name = {"B":"Blood Cell",
              "BD":"Bile Duct",
              "F":"Fibrosis",
              "H":"Hepatocyte",
              "I":"Inflammation",
              "M":"Mucin",
              "MF":"Macrophage",
              "N":"Necrosis",
              "T":"Tumor",}

# model_base_path = './output_model/1vr_macenko_normalize/'
# model_path = ['B__e11.h5',
#                   'BD__e30.h5',
#                   'F__e97.h5',
#                   'H__e79.h5',
#                   'I__e93.h5',
#                   'M__e96.h5',
#                   'MF__e21.h5',
#                   'N__e86.h5',
#                   'T__e60.h5',
#                  ]

# model_base_path = './output_model/1vr_imagenet_normalize/'
# model_path = ['resnet_1vr_60um_imagenetnormalize_B__e57.h5',
#                   'resnet_1vr_60um_imagenetnormalize_BD__e68.h5', # need to change
#                   'resnet_1vr_60um_imagenetnormalize_F__e63.h5',
#                   'resnet_1vr_60um_imagenetnormalize_H__e97.h5',
#                   'resnet_1vr_60um_imagenetnormalize_I__e58.h5',
#                   'resnet_1vr_60um_imagenetnormalize_M__e93.h5',
#                   'resnet_1vr_60um_imagenetnormalize_MF__e83.h5',
#                   'resnet_1vr_60um_imagenetnormalize_N__e90.h5',
#                   'resnet_1vr_60um_imagenetnormalize_T__e77.h5',
#                  ]

model_base_path = './'
model_path = ['resnet_1vr_60um_imagenetnormalize_T__e77.h5']

# model_path = ['resnet_1vr_60um_imagenetnormalize_MF__e83.h5']

# model_base_path = './output_model/1vr_reinhard_normalize/'
# model_path = ['resnet_1vr_60um_reinhard_B__e49.h5',
#                   'resnet_1vr_60um_reinhard_BD__e50.h5', # need to change
#                   'resnet_1vr_60um_reinhard_F__e49.h5',
#                   'resnet_1vr_60um_reinhard_H__e48.h5',
#                   'resnet_1vr_60um_reinhard_I__e35.h5',
#                   'resnet_1vr_60um_reinhard_M__e39.h5',
#                   'resnet_1vr_60um_reinhard_MF__e49.h5',
#                   'resnet_1vr_60um_reinhard_N__e48.h5',
#                   'resnet_1vr_60um_reinhard_T__e49.h5',
#                  ]

# model_base_path = './output_model/1vr_stainnet_normalize/'
# model_path = ['resnet_1vr_60um_stainnet_B__e100.h5',
#                   'resnet_1vr_60um_stainnet_BD__e42.h5', # need to change
#                   'resnet_1vr_60um_stainnet_F__e36.h5',
#                   'resnet_1vr_60um_stainnet_H__e96.h5',
#                   'resnet_1vr_60um_stainnet_I__e91.h5',
#                   'resnet_1vr_60um_stainnet_M__e67.h5',
#                   'resnet_1vr_60um_stainnet_MF__e80.h5',
#                   'resnet_1vr_60um_stainnet_N__e85.h5',
#                   'resnet_1vr_60um_stainnet_T__e98.h5',
#                  ]



SUPPORTED_WSI_FORMATS = ["svs","ndpi","vms","vmu","scn","mrxs","tiff","svslide","tif","bif"]


def setup_normalizer(normalizer_choice, ref_img_path=None):
    """
    Initialize a WSI normalizer object using the method of choice.

    Input:
        normalizer_choice (str): Valid choice for normalizer method. Use 'None' to return a Null object.
        ref_img_path (str): Path to reference image for the normalizer.

    Output:
        An initialized normalizer object:
    """

    normalizer = None

    # Import target image
    if ref_img_path is None or ref_img_path == "None":
        ref_img = norm.get_target_img()
    else:
        if os.path.exists(ref_img_path):
            ref_img = np.array(Image.open(ref_img_path).convert('RGB'))
        else:
            raise ValueError("Target image does not exist")

    # Initialize normalizer & setup reference image if required
    if normalizer_choice is not None and normalizer_choice != "None":
        
        if normalizer_choice == "macenko":
            normalizer = norm.MacenkoNormalizer.MacenkoNormalizer()

        # Add more options here as "else if" blocks, like: 
        # elif normalizer_choice == "vahadane":
        #     normalizer = norm.VahadaneNormalizer.VahadaneNormalizer()
        elif normalizer_choice == "reinhard" :
            normalizer = staintools.ReinhardColorNormalizer()
            target_image = staintools.read_image("./normalizer/macenko_reference_img.png")
            normalizer.fit(target_image)
        else:
            raise ValueError("Normalizer choice not supported")

        normalizer.fit(ref_img)
    


    return normalizer

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])

loader = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])



class SingleImage(dataset.Dataset):
    def __init__(self, data_path, transform=None, augment=None):
        self.data_path = data_path
        self.transform = transform
        self.augment = augment
        #self.image_list = fnmatch.filter(os.listdir(data_path), '*.png')#list_file_tree(os.path.join(data_path), "png")
        self.image_list = glob.glob(os.path.join(data_path, "*" + "png"), recursive=True)
        #self.image_list = list_file_tree(os.path.join(data_path), "png")
        # assert len(self.image_list) == len(self.cyt_list)
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        #print(self.image_list[item])
        img = Image.open(self.image_list[item])
        img = img.convert('RGB')
        img = loader(img).float()
        # img = Variable(img, requires_grad=True)
        # img = img.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        filepath = self.image_list[item].split('/')[-1]
        
        return filepath, img
    

def predict_batch(model_list, ref_df, offset_x, offset_y, temp_path='./temp/') :
    print(len(ref_df))
    
    #image_datasets = datasets.ImageFolder(temp_path, data_transforms)
    image_datasets = SingleImage(temp_path)

    dataloader = torch.utils.data.DataLoader(image_datasets,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=8
                                    )  
    result_list = []
    
    result_nparray = []
    result_nparray = np.array(result_nparray)
    for filenames, images in tqdm(dataloader):
        with torch.no_grad():  
        #images = images.cuda()
            images = images.to(device)
            
            print('!!!!!!!',images.shape)
            print(filenames)
            exit()
            
            new_pred = np.array(filenames)
            for i in range(len(model_list)) :
                #start_time = time.time()
                outputs = model_list[i](images)
                #print("prediction time",time.time()-start_time)
                #sample_fname, _ = dataloader.dataset.samples[index]
                
                #start_time = time.time()
                pred = []
                pred = np.array(pred)
                for j in range(len(outputs)):  
                    percentage = torch.nn.functional.softmax(outputs[j])
                    pred_index = percentage.argmax()
                    #print(filenames[j], percentage[0])
                    #ref_df.loc[ref_df['filename']==filenames[j], "pred_"+label_name[i]] = float(percentage[0]*100)
                    pred = np.append(pred, round(float(percentage[0]*100), 4))
                new_pred = np.vstack([new_pred, pred])
                                
                #print("dataframe processing time",time.time()-start_time)
            if len(result_nparray) == 0 :
                result_nparray = new_pred
            else :
                result_nparray = np.hstack([result_nparray,new_pred])
    
    result_nparray = np.flipud(np.rot90(result_nparray,k=1))
    return result_nparray#ref_df

def predict(model_list, ref_df, offset_x, offset_y, temp_path='./temp/') :
    print(len(ref_df))
    result_list = []
    for index, row in ref_df.iterrows():
        if row['filename'] == None :
            continue

        if os.path.exists(temp_path+row['filename']) == False :
            continue

        loader = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        try :
            image = Image.open(temp_path+'/'+row['filename'])
            image = image.convert('RGB')
        except :
            print(row['filename'], 'cannot be opened!')
            continue

        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        image = image.cuda()
        
        new_row = {"index_x": row['index_x'],
                    "index_y": row['index_y'],
                    "wsi_x": row['wsi_x'],
                    "wsi_y": row['wsi_y'],
                    "pred_B": 0.,
                    "pred_BD": 0.,
                    "pred_F": 0.,
                    "pred_H": 0.,
                    "pred_I": 0.,
                    "pred_M": 0.,
                    "pred_MF": 0.,
                    "pred_N": 0.,
                    "pred_T": 0.}

        for i in range(len(model_list)) :
            output = model_list[i](image)
            percentage = torch.nn.functional.softmax(output, dim=1)[0] 
            pred_index = percentage.argmax()
            
            value = round(float(percentage[0]*100), 4)
            new_row["pred_"+label_name[i]] = value
                
        result_list.append(new_row)

    colnames = ["tile_id", "index_x", "index_y", "wsi_x", "wsi_y", "mask_x", "mask_y", "filename", "tissue_ratio","pred_B","pred_BD","pred_F","pred_H","pred_I","pred_M","pred_MF","pred_N","pred_T"]

    result_df = pd.DataFrame(data=result_list, columns=colnames)
    
    return result_df

def prepare_tiles(wsi, file_name, normalizer, mpt=128, get_chunk_id=False):
    offset_x = 0 
    offset_y = 0
    
    if 'openslide.bounds-x' in wsi_image.properties.keys() :
        offset_x = int(wsi.properties['openslide.bounds-x'])
    if 'openslide.bounds-y' in wsi_image.properties.keys() :
        offset_y = int(wsi.properties['openslide.bounds-y'])

    # Calculate desired tile dimensions (pixels per tile)
    ppt_x = int(mpt / float(wsi.properties['openslide.mpp-x']))
    ppt_y = int(mpt / float(wsi.properties['openslide.mpp-y']))

    # Get thumbnail for tissue mask
    thumbnail_og = wsi.get_thumbnail(size=(wsi.level_dimensions[-1][0], wsi.level_dimensions[-1][1]))
    thumbnail = np.array(thumbnail_og)
    thumbnail = (rgb2gray(thumbnail) * 255).astype(np.uint8)

    # calculate mask parameters
    thumbnail_ratio = wsi.dimensions[0] / thumbnail.shape[1]  # wsi dimensions: (x,y); thumbnail dimensions: (rows,cols)
    thumbnail_mpp = float(wsi.properties['openslide.mpp-x']) * thumbnail_ratio
    noise_size_pix = round(256 / thumbnail_mpp)
    noise_size = round(noise_size_pix / thumbnail_ratio)
    thumbnail_ppt_x = ceil(ppt_x / thumbnail_ratio)
    thumbnail_ppt_y = ceil(ppt_y / thumbnail_ratio)
    tile_area = thumbnail_ppt_x*thumbnail_ppt_y

    # Create and clean tissue mask
    tissue_mask = (thumbnail[:, :] < threshold_otsu(thumbnail))
    tissue_mask = closing(tissue_mask, square(5))
    tissue_mask = opening(tissue_mask, square(5))
    tissue_mask = remove_small_objects(tissue_mask, noise_size)
    tissue_mask = ndi.binary_fill_holes(tissue_mask)

    if get_chunk_id:
        # Get labels for all chunks
        chunk_mask = ndi.label(tissue_mask)[0]

        # Filter out chunks smaller than tile size
        (chunk_label, chunk_size) = np.unique(chunk_mask,return_counts=True)
        filtered_chunks = chunk_label[ chunk_size < tile_area ]
        for l in filtered_chunks:
            chunk_mask[chunk_mask == l] = 0

    # Calculate margin according to ppt sizes
    wsi_x_tile_excess = wsi.dimensions[0] % ppt_x
    wsi_y_tile_excess = wsi.dimensions[1] % ppt_y

    # Determine WSI tile coordinates
    wsi_tiles_x = list(range(ceil(wsi_x_tile_excess / 2), wsi.dimensions[0] - floor(wsi_x_tile_excess / 2), ppt_x))
    wsi_tiles_y = list(range(ceil(wsi_y_tile_excess / 2), wsi.dimensions[1] - floor(wsi_y_tile_excess / 2), ppt_y))

    # Approximate mask tile coordinates
    mask_tiles_x = [floor(i / thumbnail_ratio) for i in wsi_tiles_x]
    mask_tiles_y = [floor(i / thumbnail_ratio) for i in wsi_tiles_y]

    # Populate tile reference table
    rowlist = []
    # pool = mp.Pool(48)
    for x in range(len(wsi_tiles_x)):
        for y in range(len(wsi_tiles_y)):
            aTile = tissue_mask[mask_tiles_y[y]:mask_tiles_y[y] + thumbnail_ppt_y,
                    mask_tiles_x[x]:mask_tiles_x[x] + thumbnail_ppt_x]
            
            # Determine chunk id by most prevalent ID
            if get_chunk_id:
                chunk_tile = chunk_mask[mask_tiles_y[y]:mask_tiles_y[y] + thumbnail_ppt_y,
                    mask_tiles_x[x]:mask_tiles_x[x] + thumbnail_ppt_x]
                chunk_id = np.bincount(chunk_tile.flatten()).argmax()
                

            # Calculate tissue ratio for tile
            tissue_ratio = np.sum(aTile) / aTile.size

            slide_id = len(rowlist) + 1
            
            new_row = {"tile_id": slide_id,
                       "index_x": x,
                       "index_y": y,
                       "wsi_x": wsi_tiles_x[x],
                       "wsi_y": wsi_tiles_y[y],
                       "mask_x": mask_tiles_x[x],
                       "mask_y": mask_tiles_y[y],
                       "filename": "%s_%d_%d_%d_%d.png" % (file_name.split('.')[0], x, y, wsi_tiles_x[x], wsi_tiles_y[y]),
                       "tissue_ratio": tissue_ratio,
                       }

            if get_chunk_id:
                new_row['chunk_id'] = chunk_id

            rowlist.append(new_row)
            
            
    # pool.close()
    # pool.join()
    # pool.terminate()

    # Create reference dataframe
    colnames = ["tile_id", "index_x", "index_y", "wsi_x", "wsi_y", "mask_x", "mask_y", "filename", "tissue_ratio"]
    if get_chunk_id:
                colnames.append('chunk_id')
    # print(str(len(rowlist)))
    ref_df = pd.DataFrame(data=rowlist, columns=colnames)
    
    # Remove filenames for empty tiles
    ref_df.loc[ref_df['tissue_ratio'] == 0, "filename"] = None

    return (ref_df, ppt_x, ppt_y, offset_x, offset_y)

# +
color = [[0,0,1], [0,1,0], [1,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [0,0,0], [0.3,0.3,0.4], [0.2,0.5,0.3], [0.5,0.3,0.2]] 

def export_tiles(wsi, tile_data, tile_dims, normalizer="macenko", final_tile_size=0, temp_path='./temp/'):
    labeld_list = []
    
    offset_x = 0 
    offset_y = 0
    try :
        wsi_image = openslide.open_slide(str(wsi))
    except :
        print('openslide error!')
        return
    
    if 'openslide.bounds-x' in wsi_image.properties.keys() :
        offset_x = int(wsi_image.properties['openslide.bounds-x'])
    if 'openslide.bounds-y' in wsi_image.properties.keys() :
        offset_x = int(wsi_image.properties['openslide.bounds-y'])

    for index, aTile in tile_data.iterrows():
        if aTile['filename'] == None :
            continue
        
        png_file_path = temp_path + '/' + aTile['filename']
        
        if os.path.exists(png_file_path) :
            continue
            
        aTile_img = wsi_image.read_region((aTile["wsi_x"], aTile["wsi_y"]), level=0,
                                size=(tile_dims['x'], tile_dims['y']))

        #Convert to RGB array
        aTile_img = np.array( aTile_img.convert('RGB') )

        # Normalize if required
        if normalizer is not None:
            try :
                aTile_img = normalizer.transform(aTile_img)
            except :
                print('normalizer error!')
                aTile['filename'] = None
                continue


        # Resize tile to final size
        if final_tile_size != 0:
            aTile_img = transform.resize(aTile_img, (final_tile_size,final_tile_size,3), order=1)  # 0:nearest neighbor

        
        # print(png_file_path)
        plt.imsave(png_file_path, aTile_img)            
    
    wsi_image.close()    
    

def annotation_json_consolidated(label_list, output_path, max_x, max_y) :
    import json
    
    label_color = [[255,0,0],[0,128,0],[128,0,0],[128,128,0],[0,255,0],[0,0,255],[0,255,255],[255,0,0],[255,255,0],[128,128,128]]
    

    label_array = np.zeros((int(max_x*3), int(max_y*3)), dtype=int)

    for label in label_list :
        label_array[label[6]*3][label[7]*3] = 1
        label_array[label[6]*3][label[7]*3+1] = 1
        label_array[label[6]*3][label[7]*3+2] = 1
        label_array[label[6]*3+1][label[7]*3] = 1
        label_array[label[6]*3+1][label[7]*3+1] = 1
        label_array[label[6]*3+1][label[7]*3+2] = 1
        label_array[label[6]*3+2][label[7]*3] = 1
        label_array[label[6]*3+2][label[7]*3+1] = 1
        label_array[label[6]*3+2][label[7]*3+2] = 1

    for i in range(1,max_x*3-1) :
        for j in range(1,max_y*3-1) :
            if label_array[i][j] == 1 or label_array[i][j] == -1 :
                if label_array[i-1][j] == 1 or label_array[i-1][j] == -1 :
                    if label_array[i][j-1] == 1 or label_array[i][j-1] == -1 :
                        if label_array[i+1][j] == 1 or label_array[i+1][j] == -1 :
                            if label_array[i][j+1] == 1 or label_array[i][j+1] == -1 :
                                if label_array[i-1][j-1] == 1 or label_array[i-1][j-1] == -1 :
                                    if label_array[i+1][j-1] == 1 or label_array[i+1][j-1] == -1 :
                                        if label_array[i-1][j+1] == 1 or label_array[i-1][j+1] == -1 :
                                            if label_array[i+1][j+1] == 1 or label_array[i+1][j+1] == -1 :
                                                label_array[i][j] = -1

    for i in range(0,max_x*3) :
        for j in range(0,max_y*3) :
            if label_array[i][j] == -1 :
                label_array[i][j] = 0


    annotataion_json = {}
    annotataion_json["type"]="FeatureCollection"
    annotataion_json["features"]=[]

    for label in label_list :
        # print(label)
        x1,y1,x2,y2=label[0],label[1],label[2],label[3]

        left = False
        right = False
        top = False
        bottom = False
        is_label = False

        if label_array[label[6]*3][label[7]*3] == 1 and label_array[label[6]*3+1][label[7]*3] == 1 and label_array[label[6]*3+2][label[7]*3] == 1 :
            top = True

        if label_array[label[6]*3][label[7]*3] == 1 and label_array[label[6]*3][label[7]*3+1] == 1 and label_array[label[6]*3][label[7]*3+2] == 1 :
            left = True

        if label_array[label[6]*3+2][label[7]*3] == 1 and label_array[label[6]*3+2][label[7]*3+1] == 1 and label_array[label[6]*3+2][label[7]*3+2] == 1 :
            right = True

        if label_array[label[6]*3][label[7]*3+2] == 1 and label_array[label[6]*3+1][label[7]*3+2] == 1 and label_array[label[6]*3+2][label[7]*3+2] == 1 :
            bottom = True

        if top and left and not right and not bottom :
            is_label = True
        elif top and not left and right and not bottom :
            is_label = True
        elif not top and left and not right and bottom :
            is_label = True
        elif not top and not left and right and bottom :
            is_label = True

        if top :
            if is_label :
                feature = {"type":"Feature","geometry":{"type":"LineString","coordinates":[[x1,y1],[x2,y1]]},"properties":{"object_type":"annotation","name":label[4],"classification":{"name":class_name[label_name[label[5]]],"color":label_color[label[5]],"isLocked":"true"}}}            
            else :
                feature = {"type":"Feature","geometry":{"type":"LineString","coordinates":[[x1,y1],[x2,y1]]},"properties":{"object_type":"annotation","classification":{"name":class_name[label_name[label[5]]],"color":label_color[label[5]],"isLocked":"true"}}}
            annotataion_json["features"].append(feature)        

        if left :
            if is_label :
                feature = {"type":"Feature","geometry":{"type":"LineString","coordinates":[[x1,y1],[x1,y2]]},"properties":{"object_type":"annotation","name":label[4],"classification":{"name":class_name[label_name[label[5]]],"color":label_color[label[5]],"isLocked":"true"}}}                
            else :
                feature = {"type":"Feature","geometry":{"type":"LineString","coordinates":[[x1,y1],[x1,y2]]},"properties":{"object_type":"annotation","classification":{"name":class_name[label_name[label[5]]],"color":label_color[label[5]],"isLocked":"true"}}}
            annotataion_json["features"].append(feature)

        if right :
            if is_label :
                feature = {"type":"Feature","geometry":{"type":"LineString","coordinates":[[x2,y1],[x2,y2]]},"properties":{"object_type":"annotation","name":label[4],"classification":{"name":class_name[label_name[label[5]]],"color":label_color[label[5]],"isLocked":"true"}}}                
            else :
                feature = {"type":"Feature","geometry":{"type":"LineString","coordinates":[[x2,y1],[x2,y2]]},"properties":{"object_type":"annotation","classification":{"name":class_name[label_name[label[5]]],"color":label_color[label[5]],"isLocked":"true"}}}
            annotataion_json["features"].append(feature)

        if bottom :
            if is_label :
                feature = {"type":"Feature","geometry":{"type":"LineString","coordinates":[[x1,y2],[x2,y2]]},"properties":{"object_type":"annotation","name":label[4],"classification":{"name":class_name[label_name[label[5]]],"color":label_color[label[5]],"isLocked":"true"}}}
            else :
                feature = {"type":"Feature","geometry":{"type":"LineString","coordinates":[[x1,y2],[x2,y2]]},"properties":{"object_type":"annotation","classification":{"name":class_name[label_name[label[5]]],"color":label_color[label[5]],"isLocked":"true"}}}
            annotataion_json["features"].append(feature)



    with open(output_path, 'w') as outfile :
        json.dump(annotataion_json, outfile)
        
def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()

    parser.add_argument(
          '-i','--input',
          required=True,
          type=str,          
          help='path of input wsi data')
    parser.add_argument(
          '-o','--output',
          required=True,
          type=str,          
          help='path for output')
    parser.add_argument(
          '-g','--gpu',          
          nargs="?",
          type=int,
          default=0,
          const='-1',
          help='Input GPU Number you want to use.')
    parser.add_argument(
          '-s','--size',          
          nargs="?",
          type=int,
          default=256,
          const='-1',
          help='tile size (um)')
    parser.add_argument(
          '-r','--reverse',          
          nargs="?",
          type=int,
          default=0,
          const='-1',
          help='reverse file list, 0: False, 1: True, Default: 0')
    parser.add_argument(
          '-t','--temp',
          default='./temp/',
          type=str,          
          help='path for temporal tile save path')
                           
    args = parser.parse_args()
    return args

# +
if __name__ == '__main__':    
    args = get_args()    
    
    if args.gpu > -1 :
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # The GPU id to use, usually either "0" or "1";        
    
    GPU = "cuda:0"#"cuda:"+str(args.gpu)
    device = torch.device(GPU if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    
    model_list = []
    
    for i in range(len(model_path)) :               
        model = models.resnet50(pretrained=False).to(device)
        model.fc = nn.Sequential(
                       nn.Linear(2048, 128),
                       nn.ReLU(inplace=True),
                       nn.Linear(128, nb_classes)).to(device)
        model.load_state_dict(torch.load(model_base_path+model_path[i],map_location=GPU))
        model.eval()
        model_list.append(model)

  
    paths = (args.input).split(',')
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.temp, exist_ok=True)    
        
    # normalizer = setup_normalizer('macenko', ref_img_path='normalizer/macenko_reference_img.png')
    normalizer = None
    
    for path in paths :                
        if os.path.isdir(Path(path)):
            file_list = os.listdir(path)
            if args.reverse == 1 :
                file_list.reverse()
                
            for each_file in file_list[:] :              
                print(each_file)
                for wsi_format in SUPPORTED_WSI_FORMATS :
                    if wsi_format == each_file.split('.')[-1] :
                        wsi = path + '/' + each_file                        
                        json_path = args.output + '/' + each_file.split('.')[0] 
                        print(json_path)
                        if os.path.exists(json_path+'_result.csv') :
                            print('This slide is already done!!')
                            continue                            
                        try :
                            wsi_image = openslide.open_slide(str(wsi))
                        except :
                            print('openslide error!')
                            continue

                        # tile_lists.clear()
                        start_time = time.time()
                        (ref_df, ppt_x, ppt_y, offset_x, offset_y) = prepare_tiles(wsi_image, each_file, normalizer, mpt=args.size)
                        wsi_image.close()
                        print('prepare_tiles time =', time.time()-start_time)
                        # print(ref_df)
                        max_x = ref_df['index_x'].max() 
                        max_y = ref_df['index_y'].max()
                        
                        tile_data_lists = np.array_split(ref_df.loc[ref_df['tissue_ratio'] > 0 ],cores)
                        tile_dims = {'x':ppt_x,'y':ppt_y}
                        
                        start_time = time.time()
                        pool = mp.Pool(cores)
                        for a_tile_list in tile_data_lists:
                            #export_tiles(tile_data=a_tile_list, wsi_image=wsi_image, tile_dims=tile_dims, normalizer=normalizer, final_tile_size=224)
                            pool.apply_async(func=export_tiles,kwds={
                                'wsi':wsi, 
                                'tile_data':a_tile_list,                              
                                'tile_dims':tile_dims, 
                                'normalizer':normalizer, 
                                'final_tile_size':224,
                                'temp_path':args.temp,
                                })
                        pool.close()
                        pool.join()
                        pool.terminate()
                        #export_tiles(tile_data_lists=tile_data_lists, output_path=each_file, wsi_image=wsi_image, tile_dims=tile_dims, normalizer=normalizer, final_tile_size=224)
                        
                        print('tiling time =', time.time()-start_time)
                        
                        start_time = time.time()
                        # for i in range(len(model_list)) :
                        #labeld_list = []
                        
                        # result_df = predict(model_list, ref_df, offset_x, offset_y, args.temp)
                        # result_df.to_csv(json_path + '_result.csv', sep=",", line_terminator="\n", index=False)
                        
                        result_nparray = predict_batch(model_list, ref_df, offset_x, offset_y, args.temp)
                        np.savetxt(json_path + '_result.csv', result_nparray, delimiter=",",fmt='%s',header="filename,pred_B,pred_BD,pred_F,pred_H,pred_I,pred_M,pred_MF,pred_N,pred_T")
                        
                        
                        # df = pd.DataFrame(result_df)
                        # df.to_csv(json_path + '_result.csv', sep=",", line_terminator="\n", index=False)


                        
                        #labeld_list = predict(model_list[i], tile_data_lists, label_name[i], i, offset_x, offset_y)
                        #annotation_json_consolidated(labeld_list, json_path + '_' + label_name[i] + '.json', max_x, max_y)

                        print('prediction time =', time.time()-start_time)
                        continue
                files = glob.glob(args.temp + '/*')
                for f in files:
                    os.remove(f)
                gc.collect()
    
            
        else:
            if args.input.endswith(tuple(SUPPORTED_WSI_FORMATS)):
                all_wsi_paths.append(Path(path))

       
    
    
 
