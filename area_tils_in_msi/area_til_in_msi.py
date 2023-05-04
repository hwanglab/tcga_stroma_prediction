import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from myhtml import *
import pandas as pd

dir = '/home/ext_jang_inyeop_mayo_edu/hwangt-share/datasets/Pathology_Slides/CLASSIC_Stomach_Cancer_Image/immune_subtype'
msi_list = glob(f'{dir}/MSI_prediction/pred_heatmaps/*.png')
web_dir ='area_til_stroma_tumor_msi'
os.makedirs(f'{dir}/{web_dir}/images', exist_ok=True)

df = pd.DataFrame()
page = HTML(f'{dir}/{web_dir}/','test_html')
page.add_header(f"<portion of til, stroma in MSI>-{dir.split('/')[-2]}")

for i, msi_file in enumerate(msi_list):


    msi = cv2.imread(msi_file, cv2.IMREAD_GRAYSCALE)
    s_id = msi_file.split(os.sep)[-1].split('_pred_')[0]
    
    thumbnail = cv2.imread(f'{dir}/MSI_prediction/wsi_thumbnails/{s_id}.png', cv2.IMREAD_COLOR)
    til = cv2.imread(f'{dir}/overlay/{s_id}__til.png', cv2.IMREAD_GRAYSCALE)
    stroma = cv2.imread(f'{dir}/overlay/{s_id}__stroma.png', cv2.IMREAD_GRAYSCALE)
    tumor = cv2.imread(f'{dir}/overlay/{s_id}__tumor.png', cv2.IMREAD_GRAYSCALE)

    msi=cv2.resize(msi, (til.shape[1], til.shape[0]))

    
    msi_mask = ((msi>128)*255).astype('uint8')

    til_in_msi = cv2.bitwise_and(msi_mask, til)
    stroma_in_msi = cv2.bitwise_and(msi_mask, stroma)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((10, 10), np.uint8)
    dilated_msi = cv2.dilate(msi_mask, kernel, iterations=3)
    boudary_msi = cv2.subtract(dilated_msi, msi_mask)

    til_in_boundary_msi = cv2.bitwise_and(boudary_msi, til)
    stroma_in_boundary_msi = cv2.bitwise_and(boudary_msi, stroma)
    
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}.png',thumbnail)
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}_msi.png',msi_mask)
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}_tumor.png', tumor)
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}_stroma.png', stroma)
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}_til.png', til)
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}_til_in_msi.png', til_in_msi)
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}_msi_boundary.png', boudary_msi)
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}_til_in_boundary_msi.png', til_in_boundary_msi)
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}_stroma_in_msi.png', stroma_in_msi)    
    cv2.imwrite(f'{dir}/{web_dir}/images/{s_id}_stroma_in_boundary_msi.png', stroma_in_boundary_msi)

    tumor_area = (tumor>128).sum()
    msi_area = (msi>128).sum()
    stroma_area = (stroma>128).sum()
    til_area = (til>128).sum()
    til_in_msi_area = (til_in_msi>128).sum()
    til_in_boundary_msi_area = (til_in_boundary_msi>128).sum()
    stroma_in_msi_area = (stroma_in_msi>128).sum()
    stroma_in_boundary_msi_area = (stroma_in_boundary_msi>128).sum()

    ims,links,txts =[],[],[]
    ims.append(f'{s_id}.png')
    links.append(f'{s_id}.png')
    txts.append(f'{s_id}')

    ims.append(f'{s_id}_tumor.png')
    links.append(f'{s_id}_tumor.png')
    txts.append(f'{s_id}_tumor, tumor_area:{tumor_area}')

    ims.append(f'{s_id}_stroma.png')
    links.append(f'{s_id}_stroma.png')
    txts.append(f'{s_id}_stroma, stroma_area:{stroma_area}, stroma_area/tumor_area:{stroma_area/tumor_area}')

    ims.append(f'{s_id}_til.png')
    links.append(f'{s_id}_til.png')
    txts.append(f'{s_id}_til, til_area:{til_area}, til_area/tumor_area:{til_area/tumor_area}')


    ims.append(f'{s_id}_msi.png')
    links.append(f'{s_id}_msi.png')
    txts.append(f'{s_id}_msi, msi_area:{msi_area}, msi_area/tumor_area:{msi_area/tumor_area}')

    

    ims.append(f'{s_id}_msi_boundary.png')
    links.append(f'{s_id}_msi_boundary.png')
    txts.append(f'{s_id}_msi_bounday')

    ims.append(f'{s_id}_til_in_msi.png')
    links.append(f'{s_id}_til_in_msi.png')
    txts.append(f'{s_id}_til_in_msi, til_in_msi_area:{til_in_msi_area}, area/tumor_area:{til_in_msi_area/tumor_area}')

    ims.append(f'{s_id}_til_in_boundary_msi.png')
    links.append(f'{s_id}_til_in_boundary_msi.png')
    txts.append(f'{s_id}_til_in_msi_boundary, til_in_boundary_msi_area:{til_in_boundary_msi_area}, area/tumor_area:{til_in_boundary_msi_area/tumor_area}')

    ims.append(f'{s_id}_stroma_in_msi.png')
    links.append(f'{s_id}_stroma_in_msi.png')
    txts.append(f'{s_id}_stroma_in_msi, stroma_in_msi_area:{stroma_in_msi_area}, area/tumor_area:{stroma_in_msi_area/tumor_area}')

    ims.append(f'{s_id}_stroma_in_boundary_msi.png')
    links.append(f'{s_id}_stroma_in_boundary_msi.png')
    txts.append(f'{s_id}_stroma_in_msi_boundary, stroma_in_boundary_msi_area:{stroma_in_boundary_msi_area}, area/tumor_area:{stroma_in_boundary_msi_area/tumor_area}')

    page.add_images(ims, txts, links)

    df.loc[i, 'slide_id'] =  {s_id}
    df.loc[i, 'tumor_area'] =  tumor_area
    df.loc[i, 'stroma_area/tumor_area'] =  stroma_area/tumor_area
    df.loc[i, 'til_area/tumor_area'] =  til_area/tumor_area
    df.loc[i, 'msi_area/tumor_area'] =  msi_area/tumor_area
    df.loc[i, 'til_in_msi_area/tumor_area'] =  til_in_msi_area/tumor_area
    df.loc[i, 'til_in_boundary_msi_area/tumor_area'] =  til_in_boundary_msi_area/tumor_area
    df.loc[i, 'stroma_in_msi_area/tumor_area'] =  stroma_in_msi_area/tumor_area
    df.loc[i, 'stroma_in_boundary_msi_area/tumor_area'] =  stroma_in_boundary_msi_area/tumor_area

    # plt.figure(figsize=(16,16))
    # ax = plt.subplot(1,2,1)
    # ax.imshow((msi>128).astype('uint8'))
    # ax = plt.subplot(1,2,2)
    # ax.imshow(til)
    # break
    if (i%20==0) and (i>1):
        page.save(f'{i}.html')
        page = HTML(f'{dir}/{web_dir}/','test_html')
        page.add_header('tils in MSI---------------')
df['response']=''
df.to_csv(f'{dir}/{web_dir}/comparison.csv')
