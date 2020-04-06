import numpy as np
from astropy.nddata.utils import overlap_slices

from photometry import setup_CSP


def insert_fake_source(image, source, x0, y0, flux, slicesout=False):
    '''Insert source with some scaling into image at position x0, y0

    ``image`` will be modified in place. Pass in a copy if the original
    should be preserved.
    '''
    slice_large, slice_small = overlap_slices(image.shape[:2],
                                              source.shape, (y0, x0), 'trim')
    image[slice_large] = image[slice_large] + flux / source.sum() * source[slice_small]
    if slicesout:
        return image, slice_large, slice_small
    else:
        return image


def insert_any_fake_source(imagelist, srclist, x0, y0, flux, **kwargs):
    '''Insert fake source, selecting options from list

    Randomly select images and source from a list of inputs, then insert
    fake source into image.

    Returns
    -------
    image : np.array
        Image with inserted source
    select_image : int
        Index of the image where the source was inserted. Use this to remove
        the image with the new source from the PSF base list.
    '''
    select_image = np.random.randint(imagelist.shape[-1])
    image = imagelist[:, :, select_image].copy()
    select_src = np.random.randint(len(srclist))
    src = srclist[select_src].data
    return insert_fake_source(image, src, x0, y0, flux, **kwargs), select_image


def insert_fake_fit(x0, y0, flux, stars, arr, base, normperim, medianim,
                    i_inserted=None, r_opt_mask=5):
    if i_inserted is None:
        fake, i_inserted = insert_any_fake_source(np.atleast_3d(arr),
                                                  stars, x0, y0, flux)
    else:
        fake, temp = insert_any_fake_source(arr[:, :, [i_inserted]].copy(),
                                            stars, x0, y0, flux)

    median_norm_inserted = fake / normperim[i_inserted] / medianim
    insertedfitter = setup_CSP(base, i_inserted, median_norm_inserted)

    x, y = np.indices(insertedfitter.image_dim)
    mask_region = (x - y0)**2 + (y - x0)**2 < r_opt_mask**2
    insertedfitter.manual_optmask[mask_region] = True
    outim = insertedfitter.remove_psf()

    reduced_inserted = outim  * normperim[i_inserted] * medianim
    return i_inserted, fake, median_norm_inserted, reduced_inserted
