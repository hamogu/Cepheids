'''Detect possible companions for Nancy Evans Cepheid observations

This code is custom code. It's not meant to be general for other projects
at this point because

 - it relies on the development version of photutils where the API might
   still change significantly,
 - some properties of the observations are hardcoded (e.g. the angle of the
   diffraction spikes and the read-out streak),

Most interesting for generalization is probably the PSF fit using the template
library, but this would require more general testing, more general parameters
and ideally also the  implementation of other (e.g. LOCI) algorithms.
Yet, I could see this routine go into photutils.
'''


from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import astropy
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from astropy.stats import median_absolute_deviation as mad
from astropy.modeling.models import Gaussian2D
from astropy.convolution.utils import discretize_model
from astropy.nddata.utils import overlap_slices
from astropy.nddata import Cutout2D

import photutils

# ### IMAGES AND SCALING #####

def apply_normmask(fimages, normperim=None, medianim=None, mastermask=None):
    '''Normalizes the flattened images

    fimages : np.ma.ndarray
        2 d ndarray with the image data.

    normperim : ndarray
        One normalization number per image.
        If ``None`` this will be calculated, if given the given number
        will be applied to ``fimages``, e.g. to apply the right number
        to a simulated image.
    medianim : ndarray
        Median image (median is taken after normalisation of each image).
        If ``None`` this will be calculated, if given the given number
        will be applied to ``fimages``, e.g. to apply the right number
        to a simulated image.
    mastermask : ndarray
        ``True`` at any position where one of more of the input images are
        masked.
        If ``None`` this will be calculated, if given the given number
        will be applied to ``fimages``, e.g. to apply the right number
        to a simulated image.
    '''
    if mastermask is None:
        mastermask = np.max(fimages.mask, axis=1)
    if normperim is None:
        normperim = np.ma.median(fimages[~mastermask, :], axis=0)
    if medianim is None:
        medianim = np.ma.median(fimages, axis=1)
        medianim = medianim / np.ma.median(medianim)

    fimages = fimages / normperim[None, :]
    fimages = fimages / medianim[:, None]
    return fimages[~mastermask], normperim, medianim, mastermask


def remove_normmask(data, normperim, medianim, mastermask):
    ''' reverse operate of apply_normmask'''
    out = np.ma.zeros((len(medianim), len(normperim)))
    out[~mastermask, :] = data
    out = out * medianim[:, None]
    out = out * normperim[None, :]
    # out[mastermask, :] = np.ma.masked
    return out


def prepare_images(images):
    '''flatten, mask and normalize images'''
    halfwidth = images.shape[0] // 2
    fimages = deepcopy(images)
    fimages = np.ma.array(fimages)

    fimages = fimages.reshape((-1, fimages.shape[2]))

    maxperim = np.max(fimages, axis=0)
    for i in range(fimages.shape[1]):
        fimages[fimages[:, i] > 0.6 * maxperim[i], i] = np.ma.masked

    fimages = fimages.reshape((2 * halfwidth + 1, 2 * halfwidth + 1, -1))
    fimages[:, halfwidth - 2: halfwidth + 3 + 1, :] = np.ma.masked
    fimages = fimages.reshape((-1, fimages.shape[2]))
    mfimages, normperim, medianim, mastermask = apply_normmask(fimages)
    return fimages, mfimages, normperim, medianim, mastermask


# ####### PSF AND FITTING ######

def fit_sources(image1d, psfbase, shape, normperim, medianim, mastermask,
                threshold=12, **kwargs):
    '''find and fit sources in the image

    perform PSF subtraction and then find and fit sources
    see comments in code for details

    Parameters
    ----------
    image1d : ndarray
        flattened, normalized image
    psfbase : ndarray
       2d array of psf templates (PSF library)
    threshold : float
        Detection threshold. Higher numbers find only the stronger sources.
        Experiment to find the right value.
    kwargs : dict or names arguments
        arguments for daofind (fwmh, min and max roundness, etc.)

    Returns
    -------
    fluxes_gaussian : astropy.table.Table
    imag :
        PSF subtracted image
    scaled_im :
        PSF subtracted image in daofind scaling

    '''
    psf_coeff = psf_from_projection(image1d, psfbase)
    im = image1d - np.dot(psfbase, psf_coeff)
    bkg_sigma = 1.48 * mad(im)

    # Do source detection on 2d, scaled image
    scaled_im = remove_normmask(im.reshape((-1, 1)), np.ones(1), np.ones_like(medianim), mastermask).reshape(shape)
    imag = remove_normmask(im.reshape((-1, 1)), normperim, medianim, mastermask).reshape(shape)
    sources = photutils.daofind(scaled_im, threshold=threshold * bkg_sigma, **kwargs)

    if len(sources) == 0:
        return None, imag, scaled_im
    else:
        # insert extra step here to find the brightest source, subtract it and
        # redo the PSF fit or add a PSF model to psfbase to improve the PSF fit
        # I think 1 level of that is enough, no infinite recursion.
        # Idea 1: mask out a region around the source, so that this does not
        #         influence the PSF fit.
        newmask = deepcopy(mastermask).reshape(shape)
        for source in sources:
            sl, temp = overlap_slices(shape, [9,9], [source['xcentroid'], source['ycentroid']])
            newmask[sl[0], sl[1]] = True
        newmask = newmask.flatten()

        psf_coeff = psf_from_projection(image1d[~(newmask[~mastermask])],
                                        psfbase[~(newmask[~mastermask]), :])
        im = image1d - np.dot(psfbase, psf_coeff)
        scaled_im = remove_normmask(im.reshape((-1, 1)), np.ones(1), np.ones_like(medianim), mastermask).reshape(shape)

        imag = remove_normmask(im.reshape((-1, 1)), normperim, medianim, mastermask).reshape(shape)
        # cosmics in the image lead to high points, which means that the
        # average area will be overcorrected
        imag = imag - np.ma.median(imag)
        # do photometry on image in real space

        psf_gaussian = photutils.psf.GaussianPSF(1.8)  # width measured by hand
        # default in photutils is to freeze this stuff, but I disagree
        # psf_gaussian.fixed['sigma'] = False
        # psf_gaussian.fixed['x_0'] = False
        # psf_gaussian.fixed['y_0'] = False
        fluxes_gaussian = photutils.psf.psf_photometry(imag, sources['xcentroid', 'ycentroid'], psf_gaussian)

        '''Estimate flux of Gaussian PSF from A and sigma.

        Should be part of photutils in a more clever (analytic) implementation.
        As long as it's missing there, but in this crutch here.
        '''
        x, y = np.mgrid[-3:3, -4:4]
        amp2flux = np.sum(psf_gaussian.evaluate(x, y, 1, 1, 0, 1.8))  # 1.8 hard-coded above
        fluxes_gaussian.add_column(MaskedColumn(name='flux_fit', data=amp2flux * fluxes_gaussian['amplitude_fit']))

        return fluxes_gaussian, imag, scaled_im


def photometryloop(images, targets, **kwargs):
    '''master photometry loop that loops fit_sources for each image

    Also assembles the right columns for the output of an astropy.table.Table
    '''
    all_fluxes = []
    imout = np.zeros_like(images)
    scaledim_out = np.zeros_like(images)

    fimages, mfimages, normperim, medianim, mastermask = prepare_images(images)
    for i in range(fimages.shape[1]):
        target = targets[i]
        otherpsfs = np.delete(np.arange(fimages.shape[1]), i)
        psfbase = mfimages[:, otherpsfs]
        res, image, scaled_im = fit_sources(mfimages[:, i], psfbase,
                                            images.shape[:2], normperim[[i]],
                                            medianim, mastermask, **kwargs)
        imout[:, :, i] = image
        scaledim_out[:, :, i] = scaled_im
        if res is not None:
            coord_cep = astropy.coordinates.SkyCoord(*target.all_pix2world(target.halfwidth, target.halfwidth), unit=astropy.units.degree)
            coord_res = astropy.coordinates.SkyCoord(*target.all_pix2world(res['x_0_fit'], res['y_0_fit']), unit='deg')
            res.add_column(Column(name='r', data=coord_cep.separation(coord_res).arcsec))
            res.add_column(Column(name='PA', data=coord_cep.position_angle(coord_res).degree))

            res.add_column(Column(name='id', data=[i] * len(res)))
            res.add_column(Column(name='name', data=[target.targname] * len(res),
                                  dtype='S20'))
            ra, dec = target.all_pix2world(res['x_0_fit'], res['y_0_fit'])
            res.add_column(Column(name='ra', data=ra))
            res.add_column(Column(name='dec', data=dec))
            all_fluxes.append(res)
    all_fluxes = astropy.table.vstack(all_fluxes)
    return all_fluxes, imout, scaledim_out


# ######### SIMULATIONS ########

def simulate_image(images):
    '''pick image, pick source parameters, add gaussian source'''
    n = np.random.choice(images.shape[2])
    halfwidth = images.shape[0] // 2
    theta = np.random.uniform() * 2. * np.pi
    r = np.random.uniform() * halfwidth  # pix
    x_in = halfwidth + r * np.sin(theta)  # pix
    y_in = halfwidth + r * np.cos(theta)  # pix
    flux_in = 10 ** (np.random.uniform() * 7 + 1)
    sigma_in = np.random.uniform() * 0.5 + 1.

    val_in = {'id': [n], 'theta': [theta], 'r': [r], 'x': [x_in], 'y': [y_in],
              'flux': [flux_in], 'sigma': [sigma_in]}

    # Create test psf
    psf_model = Gaussian2D(flux_in / (2 * np.pi * sigma_in ** 2), x_in,
                           y_in, sigma_in, sigma_in)
    test_image = images[:, :, n] + discretize_model(psf_model,
                                                    (0, images.shape[0]),
                                                    (0, images.shape[1]),
                                                    mode='oversample')
    return Table(val_in), test_image


def fit_sim_image(test_image, halfwidth, normperim, medianim,
                  mastermask, mfimages, n, x_in, y_in, **kwargs):
    '''run fit_source as for real sources and return sim and fit params'''
    # now do same stuff as with data
    test_image[:, halfwidth - 2: halfwidth + 3 + 1] = np.ma.masked
    ftest_image = test_image.reshape((-1, 1))
    mftest_image, temp_1, temp_2, temp_3 = apply_normmask(ftest_image,
                                                          normperim[[n]],
                                                          medianim, mastermask)

    otherpsfs = np.delete(np.arange(mfimages.shape[1]), n)
    psfbase = mfimages[:, otherpsfs]

    fit_res, temp1, temp2 = fit_sources(mftest_image, psfbase, test_image.shape,
                                        normperim[[n]], medianim, mastermask, **kwargs)
    # fit_res could contain several sources. We look for the one closest to the
    # simulated source
    if fit_res is None:
        return None
    else:
        i = np.argmin((fit_res['x_0_fit'] - x_in) ** 2 + (fit_res['y_0_fit'] - y_in) ** 2)
        return fit_res[i]


def run_sim(images, n=1000, **kwargs):
    '''loop over fit_sim_image'''
    halfwidth = images.shape[0] // 2
    fimages, mfimages, normperim, medianim, mastermask = prepare_images(images)
    out = []
    for i in range(n):
        val_in, sim_image = simulate_image(images)
        fit_res = fit_sim_image(sim_image, halfwidth, normperim, medianim,
                                mastermask, mfimages, val_in['id'][0],
                                val_in['x'][0], val_in['y'][0], **kwargs)
        if fit_res is None:
            out.append(val_in)
        else:
            out.append(astropy.table.hstack([val_in, fit_res]))
    return astropy.table.vstack(out)


# #### PLOTTING #####

def targname(n):
    n = n.replace('-', ' ')
    n = n.replace('BETA', r'$\beta$')
    n = n.replace('ZETA', r'$\zeta$')
    n = n.replace('ETA', r'$\eta$')
    n = n.replace('DEL', r'$\delta$')
    if ' ' not in n:
        n = n[:-3] + ' ' + n[-3:]
    n = n[:-2] + n[-2:].lower()
    return n


def coosys(target, coord, length=1.2, color='g'):
    ra, dec = target.all_pix2world(coord[0], coord[1])
    dec0 = dec - length / 3600.
    dec1 = dec + length / 3600.
    x0, y0 = target.all_world2pix(ra, dec0)
    x1, y1 = target.all_world2pix(ra, dec1)
    plt.annotate("N", xy=(x0, y0), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="<-", lw=2., color=color,
                                 shrinkA=0, shrinkB=0),
                 color=color, weight='bold', ha='center', va='center')


def plot_gallery(title, images, targets, n_col=5, n_row=7,
                 sources={'name': np.array([None])}, arrowlength=0.6, color='g',
                 figsize=(7, 10)):
    plt.figure(figsize=figsize)
    plt.suptitle(title, size=16)
    for i in range(images.shape[2]):
        comp = images[:, :, i]
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp, cmap=plt.cm.gray, interpolation='nearest',
                   vmin=np.percentile(comp, 1), vmax=np.percentile(comp, 99),
                   origin='lower')
        plt.xticks(())
        plt.yticks(())
        plt.title(targname(targets[i].targname))
        coosys(targets[i], (targets[i].halfwidth * 1.6, 20),
               length=arrowlength, color=color)
        ind = (sources['name'] == targets[i].targname)
        if ind.sum()> 0:
            # currently apertures cannot be a table, but that will hopefully improve
            apertures = photutils.CircularAperture(zip(sources['x_0_fit'][ind], sources['y_0_fit'][ind]), r=5.)
            apertures.plot(color='red', lw=3.)
    plt.subplots_adjust(0.01, 0.01, 0.99, 0.92, 0.02, 0.25)
