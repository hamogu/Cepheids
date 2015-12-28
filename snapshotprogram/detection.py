'''Detect possible companions for Nancy Evans Cepheid observations

This code is custom code. It's not meant to be general for other projects
at this point because

 - it relies on the development version of photutils where the API might
   still change significantly,
 - some properties of the observations are hardcoded (e.g. the angle of the
   diffraction spikes and the read-out streak),
 - there are some workaround for shortcomings of current imutils (e.g.
   class Wrap_all_pix2world is needed because there is no simple way to
   extract a subimage and adjust the WCS accordingly) that might be fixed
   soon.

Most interesting for generalization is probably the PSF fit using the template
library, but this would require more general testing, more general parameters
and ideally also the  implementation of other (e.g. LOCI) algorithms.
Yet, I could see this routine go into photutils.
'''


from copy import deepcopy

import numpy as np
from scipy import ndimage, stats
import matplotlib.pyplot as plt
from skimage import filters as skfilter
from skimage.morphology import convex_hull_image

import astropy
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from astropy.stats import median_absolute_deviation as mad
from astropy.modeling.models import Gaussian2D
from astropy.convolution.utils import discretize_model
from astropy.nddata.utils import overlap_slices

import photutils

# ##### MASKS and SPIKES ######


def center_from_spikes(image, **kwargs):
    '''Fit both diffraction spikes and return the intersection point'''
    m1, b1 = fit_diffraction_spike(image, 1., **kwargs)
    m2, b2 = fit_diffraction_spike(image, -1., **kwargs)

    xmnew = (b1 - b2) / (m2 - m1)
    ymnew = m1 * xmnew + b1

    return xmnew, ymnew


def fit_diffraction_spike(image, fac=1, r_inner=50, r_outer=250, width=25):
    '''fit a diffraction spike with a line

    The fit is done on the left side of the image from ``-r_outer`` to
    ``-r_inner`` and on the right side from ``+r_inner`` to ``+r_outer``
    (all coordinates in pixels, measured from the center of the image).
    ``fac`` is an initial guess for the parameter m in the equation y = m*x+b.
    For each column in the image, a ``width`` strip of pixels centered on the
    initial guess is extracted and the position of the maximum is recorded.
    Regions with low signal, i.e. regions where the maximum is not well
    determined (standard deviation < 40% percentile indicates that all pixels
    in that strip are very similar -> there is not signal).
    '''
    xm, ym = ndimage.center_of_mass(image)
    s = np.hstack([np.arange(-r_outer, -r_inner), np.arange(r_inner, r_outer)])
    x = xm + s
    y = ym + fac * s

    ytest = np.zeros((len(x), 2 * width + 1))
    ymax = np.zeros((len(x)))

    for i in range(len(x)):
        ytest[i, :] = image[np.int(x[i]), np.int(y[i] - width): np.int(y[i] + width + 1)]
    ymax = np.argmax(ytest, axis=1) - width

    # identify where there is actually a signal
    st = np.std(ytest, axis=1)
    ind = st > np.percentile(st, 40.)
    m, b, r_value, p_value, std_err = stats.linregress(x[ind], y[ind] + ymax[ind])
    if np.abs(r_value) < 0.99:
        raise Exception("No good fit to spike found")
    return m, b


def mask_spike(image, m, xm, ym, width=3):
    # make normalized vector normal to spike in (x,y)
    n = np.array([1, -1. / m])
    n = n / np.sqrt(np.sum(n**2))
    # Distance separately for x and y, because I did not find a matrix form
    # to write this dot product for each x in closed form
    x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    dx = (x - xm) * n[0]
    dy = (y - ym) * n[1]
    r = np.abs(dx + dy)
    return r < width


def mask_spikes(image, m1, m2, maskwidth=3, **kwargs):
    xmnew, ymnew = center_from_spikes(image, **kwargs)
    mask1 = mask_spike(image, m1, xmnew, ymnew, width=maskwidth)
    mask2 = mask_spike(image, m2, xmnew, ymnew, width=maskwidth)
    return mask1 | mask2, xmnew, ymnew


def mask_readoutstreaks(image):
    # logarithmic image to edge detect faint features
    logimage = np.log10(np.clip(image, 1, 1e5)) / 5
    # Mask overexposed area + sobel edge detect
    mask = (skfilter.sobel(logimage) > 0.1) | (image > 0.6 * np.max(image))
    # pick out the feature that contain the center
    # I hope that this always is bit enough
    mask, lnum = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))

    i = mask[ndimage.center_of_mass(image)]
    mask = (mask == i)
    # fill any holes in that region
    # return mask
    return convex_hull_image(mask)


# ### IMAGES AND SCALING #####


class Wrap_all_pix2world(object):
    '''Work-around required when cutting fits images

    So far, astropy has not implemented a proper way to select a subset
    of a fits image and keep the WCS intact. This is in the works.
    Until that time, we use this simple wrapper the keeps a copy of the
    WCS and just shifts (x,y) coordinates so that we can call the
    all_pix2world on the original WCS.

    Note that we use this wrapper only to keep the relevant header, but not
    the image data itself for performance reasons.
    '''
    def __init__(self, filename, xm, ym, halfwidth):
        self.filename = filename
        self.header = fits.getheader(filename, 1)
        self.header0 = fits.getheader(filename, 0)
        self.wcs = astropy.wcs.WCS(self.header)
        self.xm = xm
        self.ym = ym
        self.halfwidth = halfwidth
        self.targname = self.header0['TARGNAME']

    def all_pix2world(self, x, y):
        return self.wcs.all_pix2world(x + self.ym - self.halfwidth,
                                      y + self.xm - self.halfwidth, 0)

    def all_world2pix(self, ra, dec):
        x, y =  self.wcs.all_world2pix(ra, dec, 0)
        return x - self.xm + self.halfwidth, y - self.ym + self.halfwidth

    def reinsert_image(self, smallimage):
        '''for plotting purposes, return image with zeors on eahc side
        so that WCS works for AplPy.
        '''
        image = np.zeros((self.header['NAXIS1'], self.header['NAXIS2']))
        image[self.xm - self.halfwidth : self.xm + self.halfwidth + 1,
              self.ym - self.halfwidth : self.ym + self.halfwidth + 1] = smallimage
        self.data = image


def read_images(filelist, halfwidth):
    '''Read files, make header wrappers for WCS'''
    images = np.zeros((2 * halfwidth + 1, 2 * halfwidth + 1, len(filelist)))
    targets = []

    for i, f in enumerate(filelist):
        image = fits.getdata(f)
        xm, ym = center_from_spikes(image)
        targets.append(Wrap_all_pix2world(f, xm, ym, halfwidth))
        xm = np.int(xm + 0.5)
        ym = np.int(ym + 0.5)
        images[:, :, i] = image[xm - halfwidth : xm+halfwidth + 1,
                                ym - halfwidth : ym+halfwidth + 1]
    return images, targets


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
        normperim = np.median(fimages[~mastermask, :], axis=0)
    if medianim is None:
        medianim = np.ma.median(fimages, axis=1)

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


def psf_from_projection(image1d, psfbase):
    '''solve a linear algebra system for the best PSF

    Parameters
    ----------
    image1d : array in 1 dim
    psfbase : array in [M,N]
        M = number of pixels in flattened image
        N = number of images that form the space of potential PSFs
    '''
    a = np.dot(psfbase.T, psfbase)
    b = np.dot(psfbase.T, image1d)
    x = np.linalg.solve(a, b)
    return x


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
                 sources={'id': np.array([None])}, arrowlength=0.6, color='g',
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
