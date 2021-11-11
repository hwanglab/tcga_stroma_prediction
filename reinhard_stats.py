"""
The code is from the HistomicsTK project.
For more information, go to https://github.com/DigitalSlideArchive/HistomicsTK
"""

import collections
import numpy as np
from color_conversion import lab_mean_std
from simple_mask import simple_mask
import dask
import dask.distributed
import large_image
import PIL.Image
import numpy as np


def estimate_variance(x, y, peak):
    """Estimates variance of a peak in a histogram using the FWHM of an
    approximate normal distribution.
    Starting from a user-supplied peak and histogram, this method traces down
    each side of the peak to estimate the full-width-half-maximum (FWHM) and
    variance of the peak. If tracing fails on either side, the FWHM is
    estimated as twice the HWHM.
    Parameters
    ----------
    x : array_like
        vector of x-histogram locations.
    y : array_like
        vector of y-histogram locations.
    peak : double
        index of peak in y to estimate variance of
    Returns
    -------
    scale : double
        Standard deviation of normal distribution approximating peak. Value is
        -1 if fitting process fails.
    See Also
    --------
    SimpleMask

    """

    # analyze peak to estimate variance parameter via FWHM
    Left = peak
    while y[Left] > y[peak] / 2 and Left >= 0:
        Left -= 1
        if Left == -1:
            break
    Right = peak
    while y[Right] > y[peak] / 2 and Right < y.size:
        Right += 1
        if Right == y.size:
            break
    if Left != -1 and Right != y.size:
        LeftSlope = y[Left + 1] - y[Left] / (x[Left + 1] - x[Left])
        Left = (y[peak] / 2 - y[Left]) / LeftSlope + x[Left]
        RightSlope = y[Right] - y[Right - 1] / (x[Right] - x[Right - 1])
        Right = (y[peak] / 2 - y[Right]) / RightSlope + x[Right]
        scale = (Right - Left) / 2.355
    if Left == -1:
        if Right == y.size:
            scale = -1
        else:
            RightSlope = y[Right] - y[Right - 1] / (x[Right] - x[Right - 1])
            Right = (y[peak] / 2 - y[Right]) / RightSlope + x[Right]
            scale = 2 * (Right - x[peak]) / 2.355
    if Right == y.size:
        if Left == -1:
            scale = -1
        else:
            LeftSlope = y[Left + 1] - y[Left] / (x[Left + 1] - x[Left])
            Left = (y[peak] / 2 - y[Left]) / LeftSlope + x[Left]
            scale = 2 * (x[peak] - Left) / 2.355

    return scale


def sample_pixels(slide_path, sample_fraction=None, magnification=None,
                  tissue_seg_mag=1.25, min_coverage=0.1, background=False,
                  sample_approximate_total=None, tile_grouping=256):
    """Generates a sampling of pixels from a whole-slide image.

    Useful for generating statistics or Reinhard color-normalization or
    adaptive deconvolution. Uses mixture modeling approach to focus
    sampling in tissue regions.

    Parameters
    ----------
    slide_path : str
        path and filename of slide.
    sample_fraction : double
        Fraction of pixels to sample. Must be in the range [0, 1].
    magnification : double
        Desired magnification for sampling.
        Default value : None (for native scan magnification).
    tissue_seg_mag: double, optional
        low resolution magnification at which foreground will be segmented.
        Default value = 1.25.
    min_coverage: double, optional
        minimum fraction of tile covered by tissue for it to be included
        in sampling. Ranges between [0,1). Default value = 0.1.
    background: bool, optional
        sample the background instead of the foreground if True. min_coverage
        then refers to the amount of background. Default value = False
    sample_approximate_total: int, optional
        use instead of sample_fraction to specify roughly how many pixels to
        sample. The fewer tiles are excluded, the more accurate this will be.
    tile_grouping: int, optional
        Number of tiles to process as part of a single task.

    Returns
    -------
    pixels : array_like
        A Nx3 matrix of RGB pixel values sampled from the whole-slide.

    Notes
    -----
    If Dask is configured, it is used to distribute the computation.

    See Also
    --------
    histomicstk.preprocessing.color_normalization.reinhard

    """

    if (sample_fraction is None) == (sample_approximate_total is None):
        raise ValueError('Exactly one of sample_fraction and ' +
                         'sample_approximate_total must have a value.')

    ts = large_image.getTileSource(slide_path)

    if magnification is None:
        magnification = ts.getMetadata()['magnification']

    # get entire whole-slide image at low resolution
    scale_lres = {'magnification': tissue_seg_mag}
    im_lres, _ = ts.getRegion(
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        scale=scale_lres
    )
    im_lres = im_lres[:, :, :3]

    # compute foreground mask of whole-slide image at low-res.
    # it will actually be a background mask if background is set.
    im_fgnd_mask_lres = bool(background) ^ simple_mask(im_lres)

    if sample_approximate_total is not None:
        scale_ratio = float(magnification) / tissue_seg_mag
        total_fgnd_pixels = np.count_nonzero(im_fgnd_mask_lres) * scale_ratio ** 2
        sample_fraction = sample_approximate_total / total_fgnd_pixels

    # broadcasting fgnd mask to all dask workers
    try:
        c = dask.distributed.get_client()

        [im_fgnd_mask_lres] = c.scatter([im_fgnd_mask_lres],
                                        broadcast=True)
    except ValueError:
        pass

    # generate sample pixels
    sample_pixels = []

    iter_args = dict(scale=dict(magnification=magnification),
                     format=large_image.tilesource.TILE_FORMAT_NUMPY)

    total_tiles = ts.getSingleTile(**iter_args)['iterator_range']['position']

    for position in range(0, total_tiles, tile_grouping):

        sample_pixels.append(dask.delayed(_sample_pixels_tile)(
            slide_path, iter_args,
            (position, min(tile_grouping, total_tiles - position)),
            sample_fraction, tissue_seg_mag, min_coverage,
            im_fgnd_mask_lres))

    # concatenate pixel values in list
    if sample_pixels:
        sample_pixels = (dask.delayed(np.concatenate)(sample_pixels, 0)
                         .compute())
    else:
        print("Sampling could not identify any foreground regions.")

    return sample_pixels


def _sample_pixels_tile(slide_path, iter_args, positions, sample_fraction,
                        tissue_seg_mag, min_coverage, im_fgnd_mask_lres):
    start_position, position_count = positions
    sample_pixels = [np.empty((0, 3))]
    ts = large_image.getTileSource(slide_path)
    for position in range(start_position, start_position + position_count):
        tile = ts.getSingleTile(tile_position=position, **iter_args)
        # get current region in base_pixels
        rgn_hres = {'left': tile['gx'], 'top': tile['gy'],
                    'right': tile['gx'] + tile['gwidth'],
                    'bottom': tile['gy'] + tile['gheight'],
                    'units': 'base_pixels'}

        # get foreground mask for current tile at low resolution
        rgn_lres = ts.convertRegionScale(rgn_hres,
                                         targetScale={'magnification':
                                                      tissue_seg_mag},
                                         targetUnits='mag_pixels')

        top = np.int(rgn_lres['top'])
        bottom = np.int(rgn_lres['bottom'])
        left = np.int(rgn_lres['left'])
        right = np.int(rgn_lres['right'])

        tile_fgnd_mask_lres = im_fgnd_mask_lres[top:bottom, left:right]

        # skip tile if there is not enough foreground in the slide
        cur_fgnd_frac = tile_fgnd_mask_lres.mean()

        if np.isnan(cur_fgnd_frac) or cur_fgnd_frac <= min_coverage:
            continue

        # get current tile image
        im_tile = tile['tile'][:, :, :3]

        # get tile foreground mask at resolution of current tile
        tile_fgnd_mask = np.array(PIL.Image.fromarray(tile_fgnd_mask_lres).resize(
            im_tile.shape[:2],
            resample=PIL.Image.NEAREST
        ))

        # generate linear indices of sample pixels in fgnd mask
        nz_ind = np.nonzero(tile_fgnd_mask.flatten())[0]

        # Handle fractions in the desired sample size by rounding up
        # or down, weighted by the fractional amount.
        float_samples = sample_fraction * nz_ind.size
        num_samples = int(np.floor(float_samples))
        num_samples += np.random.binomial(1, float_samples - num_samples)

        sample_ind = np.random.choice(nz_ind, num_samples)

        # convert rgb tile image to Nx3 array
        tile_pix_rgb = np.reshape(im_tile, (-1, 3))

        # add rgb triplet of sample pixels
        sample_pixels.append(tile_pix_rgb[sample_ind, :])

    return np.concatenate(sample_pixels, 0)


def reinhard_stats(slide_path, sample_fraction, magnification=None,
                   tissue_seg_mag=1.25):
    """Samples a whole-slide-image to determine colorspace statistics (mean,
    variance) needed to perform global Reinhard color normalization.

    Normalizing individual tiles independently creates a significant bias
    in the results of segmentation and feature extraction, as the color
    statistics of each tile in a whole-slide image can vary significantly.
    To remedy this, we sample a subset of pixels from the entire whole-slide
    image in order to estimate the global mean and standard deviation of
    each channel in the Lab color space that are needed for reinhard color
    normalization.

    Parameters
    ----------
    slide_path : str
        path and filename of slide.
    sample_fraction : double
       Fraction of pixels to sample (range (0, 1]).
    magnification : scalar
        Desired magnification for sampling. Defaults to native scan
        magnification.
    tissue_seg_mag: double, optional
        low resolution magnification at which foreground will be segmented.
        Default value = 1.25.

    Returns
    -------
    Mu : array_like
        A 3-element array containing the means of the target image channels
        in sample_pix_lab color space.
    Sigma : array_like
        A 3-element list containing the standard deviations of the target image
        channels in sample_pix_lab color space.

    Notes
    -----
    Returns a namedtuple.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.lab_mean_std
    histomicstk.preprocessing.color_normalization.reinhard

    """

    # generate a sampling of sample_pixels_rgb pixels from whole-slide image
    sample_pixels_rgb = sample_pixels(
        slide_path,
        sample_fraction=sample_fraction,
        magnification=magnification,
        tissue_seg_mag=tissue_seg_mag
    )

    # reshape the Nx3 pixel array into a 1 x N x 3 image for lab_mean_std
    sample_pixels_rgb = np.reshape(sample_pixels_rgb,
                                   (1, sample_pixels_rgb.shape[0], 3))

    # compute mean and stddev of sample pixels in Lab space
    Mu, Sigma = lab_mean_std(sample_pixels_rgb)

    # build named tuple for output
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    stats = ReinhardStats(Mu, Sigma)

    return stats

def reinhard_stats_png(slide_path):
    """Samples an image to determine colorspace statistics (mean,
    variance) needed to perform global Reinhard color normalization.

    """
    # generate a sampling of sample_pixels_rgb pixels from whole-slide image
    sample_pixels_rgb = np.array(PIL.Image.open(slide_path).convert('RGB'))

    # reshape the Nx3 pixel array into a 1 x N x 3 image for lab_mean_std
    sample_pixels_rgb = np.reshape(sample_pixels_rgb,
                                   (1, sample_pixels_rgb.shape[0]*sample_pixels_rgb.shape[1], 3))

    # compute mean and stddev of sample pixels in Lab space
    Mu, Sigma = lab_mean_std(sample_pixels_rgb)

    # build named tuple for output
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    stats = ReinhardStats(Mu, Sigma)

    return stats
