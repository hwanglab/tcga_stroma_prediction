"""
The code is from the HistomicsTK project.
For more information, go to https://github.com/DigitalSlideArchive/HistomicsTK
"""

import numpy as np
from simple_mask import simple_mask
import large_image

def segment_wsi_foreground_at_low_res(ts, lres_size=2048):

    ts_metadata = ts.getMetadata()

    # get image at low-res
    maxSize = max(ts_metadata['sizeX'], ts_metadata['sizeY'])
    maxSize = float(max(maxSize, lres_size))

    downsample_factor = 2.0 ** np.floor(np.log2(maxSize / lres_size))

    fgnd_seg_mag = ts_metadata['magnification'] / downsample_factor

    fgnd_seg_scale = {'magnification': fgnd_seg_mag}

    im_lres, _ = ts.getRegion(
        scale=fgnd_seg_scale,
        format=large_image.tilesource.TILE_FORMAT_NUMPY
    )

    im_lres = im_lres[:, :, :3]

    # compute foreground mask at low-res
    im_fgnd_mask_lres = simple_mask(im_lres)

    return im_fgnd_mask_lres, fgnd_seg_scale
