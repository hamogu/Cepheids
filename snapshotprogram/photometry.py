import numpy as np
from astropy.nddata.utils import overlap_slices
from astropy.table import Table
import sherpa
import sherpa.astro.models

from psfsubtraction.fitpsf.fitters import CepheidSnapshotpaper as CSP
from psfsubtraction.fake import remove_img_from_psfbase


def setup_CSP(psfs, i, images):
    fitter = CSP(remove_img_from_psfbase(psfs, i), images)
    fitter.min_number_of_bases = 30
    fitter.region_min_size = 1
    fitter.manual_optmask = psfs.mask[:, :, i]
    return fitter


def initial_fitter_list(psfs, images):
    '''Make a list of fitters

    Here, the psfs are assumed to contain the image data. For the fit to
    image[..., i], the psf[..., i] will be removed from the psf base.

    Parameters
    ----------
    psfs : np.array
        data to be used as psf templates (images stacked along the third
        dimension).  This can, but does not have to be, different from
        ``images``. For example, the ``psfs`` and the ``images`` could use the
        same image data, but the ``psfs`` could have more pixels masked.

    images : np.array
        data to be fit (images stacked along the third dimension)

    '''
    fitterlist = [setup_CSP(psfs, i, images[..., i]) for i in range(images.shape[-1])]
    return fitterlist


# Vegamag zero points are here http://www.stsci.edu/hst/wfc3/phot_zp_lbn
zeropoints = {'F621M': 24.4539, 'F845M': 23.2809}

# Units in image are electrons/s
# Images are drz -> drizzeled -> pixels are all same size.


def flux2mag(flux, filtername):
    return -2.5 * np.log10(flux) + zeropoints[filtername]


def mag2flux(mag, filtername):
    return 10**((zeropoints[filtername] - mag) / 2.5)


def betamodel2mag(beta, filtername):
    '''Calculate photometric mag from parameters of fitted beta model'''
    flux = beta.ampl.val / ((beta.alpha.val - 1) / (np.pi * beta.r0.val**2))
    return flux2mag(flux, filtername)


def phot_sherpa(reduced_images621, reduced_images845, indexname,
                 row, psf_621, psf_845, slice_size=11, debug=False, maxdxdy=1, **kwargs):
    '''Perform 2 band photometry

    This function performs two band photometry on a single source. Photometry
    is done by fitting a PSF model simultaneously to data in both bands. Only a
    single object is fit (so this does not take into account blended sources),
    but this function is still providing a benefit over simple aperture
    photometry:

    - Because the functional form of the PSF is fit, it does work
    in the presence of some masked pixels (e.g. if the center of the source is
    saturated).

    - This function couples the source position in both
    bands. Source positions are allowed to vary within a small range around the
    input source position. This allows for small errors in the WCS of the two
    images or for differences that arise because we extract the
    Cepheid-centered images to full-pixels, so that sub-pixel differences in
    the position can arise.
    '''
    i = indexname(row['TARGNAME'])

    # Set up data #
    # Note how x and y are reversed to match order of array
    slice_large, slice_small = overlap_slices(reduced_images621[:, :, 0].shape,
                                              (slice_size, slice_size),
                                              (row['ycentroid'], row['xcentroid']),
                                              'trim')
    x0axis, x1axis = np.mgrid[slice_large]
    im621 = reduced_images621[:, :, i][slice_large]
    im845 = reduced_images845[:, :, i][slice_large]
    data621 = sherpa.data.Data2D('img621', x0axis.ravel(), x1axis.ravel(),
                                 im621.ravel(), shape=(slice_size, slice_size))
    data621.mask = ~im621.mask.ravel()
    data845 = sherpa.data.Data2D('img845', x0axis.ravel(), x1axis.ravel(),
                                 im845.ravel(), shape=(slice_size, slice_size))
    data845.mask = ~im845.mask.ravel()

    # Set up model #
    colnames = ['ycentroid', 'xcentroid']

    for psf in [psf_621, psf_845]:
        for j, par in enumerate([psf.xpos, psf.ypos]):
            # Set min/max to large numbers to make sure assignment works
            par.min = -1e10
            par.max = 1e10
            # set value
            par.val = row[colnames[j]]
            # Then restrict min/max to the range we want
            par.max = par.val + maxdxdy
            par.min = par.val - maxdxdy

    psf_621.ampl = data621.y.max()
    psf_845.ampl = data845.y.max()

    # Prepare and perform combined fit #
    # Code for combined fitting - but that turned out not to be necessary
    # databoth = sherpa.data.DataSimulFit('bothdata', (data621, data845))
    # modelboth = sherpa.models.SimulFitModel('bothmodel', (psf_621, psf_845))
    # fitboth = sherpa.fit.Fit(databoth, modelboth, **kwargs)
    # fitres = fitboth.fit()
    fit621 = sherpa.fit.Fit(data621, psf_621, **kwargs)
    fit845 = sherpa.fit.Fit(data845, psf_845, **kwargs)
    fit621.fit()
    fit845.fit()

    # Format output #
    # Get full images so that we can construct residual images of full size
    im621 = reduced_images621[:, :, i]
    x0axis, x1axis = np.mgrid[0:im621.shape[0], 0:im621.shape[1]]
    im845 = reduced_images845[:, :, i]
    resim621 = im621 - psf_621(x0axis.ravel(), x1axis.ravel()).reshape(im621.shape)
    resim845 = im845 - psf_845(x0axis.ravel(), x1axis.ravel()).reshape(im845.shape)

    outtab = Table({'TARGNAME': [row['TARGNAME']],
                    'y_621': [psf_621.xpos.val],
                    'x_621': [psf_621.ypos.val],
                    'y_845': [psf_845.xpos.val],
                    'x_845': [psf_845.ypos.val],
                    'mag_621': [betamodel2mag(psf_621, 'F621M')],
                    'mag_845': [betamodel2mag(psf_845, 'F845M')],
                    'x_0': [row['xcentroid']],
                    'y_0': [row['ycentroid']],
                    'residual_image_621': resim621.reshape(1, resim621.shape[0], resim621.shape[1]),
                    'residual_image_845': resim845.reshape(1, resim845.shape[0], resim845.shape[1]),
                   })
    if debug:
        return data621, data845, fit621, fit845, outtab
    else:
        return outtab
