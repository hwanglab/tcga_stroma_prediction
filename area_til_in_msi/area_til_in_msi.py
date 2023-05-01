import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from myhtml import *
msi_list = glob('./MSI_prediction/pred_heatmaps/*.png')
web_dir ='Yonsei_Classic'

page = HTML(f'{web_dir}/','test_html')
page.add_header('tils in MSI---------------')
for i, msi_file in enumerate(msi_list):
    msi = cv2.imread(msi_file, cv2.IMREAD_GRAYSCALE)
    s_id = msi_file.split(os.sep)[-1].split('_pred_')[0]
    til = cv2.imread(f'./overlay/{s_id}__til.png', cv2.IMREAD_GRAYSCALE)

    msi=cv2.resize(msi, (til.shape[1], til.shape[0]))

    
    msi_mask = ((msi>128)*255).astype('uint8')

    til_in_msi = cv2.bitwise_and(msi_mask, til)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((10, 10), np.uint8)
    dilated_msi = cv2.dilate(msi_mask, kernel, iterations=3)
    boudary_msi = cv2.subtract(dilated_msi, msi_mask)

    til_in_boundary_msi = cv2.bitwise_and(boudary_msi, til)

    cv2.imwrite(f'./{web_dir}/images/{s_id}_msi.png',msi_mask)
    cv2.imwrite(f'./{web_dir}/images/{s_id}_til.png', til)
    cv2.imwrite(f'./{web_dir}/images/{s_id}_til_in_msi.png', til_in_msi)
    cv2.imwrite(f'./{web_dir}/images/{s_id}_msi_boundary.png', boudary_msi)
    cv2.imwrite(f'./{web_dir}/images/{s_id}_til_in_boundary_msi.png', til_in_boundary_msi)

    ims, txts, links = [], [], []
    ims.append(f'{s_id}_msi.png')
    links.append(f'{s_id}_msi.png')
    txts.append(f'{s_id}_msi, area:{(msi>128).sum()}')

    ims.append(f'{s_id}_til.png')
    links.append(f'{s_id}_til.png')
    txts.append(f'{s_id}_til')

    ims.append(f'{s_id}_msi_boundary.png')
    links.append(f'{s_id}_msi_boundary.png')
    txts.append(f'{s_id}_msi_bounday')

    ims.append(f'{s_id}_til_in_msi.png')
    links.append(f'{s_id}_til_in_msi.png')
    txts.append(f'{s_id}_til_in_msi, area:{(til_in_msi>128).sum()}')

    ims.append(f'{s_id}_til_in_boundary_msi.png')
    links.append(f'{s_id}_til_in_boundary_msi.png')
    txts.append(f'{s_id}_til_in_msi_boundary, area:{(til_in_boundary_msi>128).sum()}')

    page.add_images(ims, txts, links)


    # plt.figure(figsize=(16,16))
    # ax = plt.subplot(1,2,1)
    # ax.imshow((msi>128).astype('uint8'))
    # ax = plt.subplot(1,2,2)
    # ax.imshow(til)
    # break
    if (i%20==0) and (i>1):
        page.save(f'{i}.html')
        page = HTML(f'{web_dir}/','test_html')
        page.add_header('tils in MSI---------------')
