import numpy as np
from astropy.table import vstack
from photutils import DAOStarFinder
from astropy.nddata import NDData
from photutils.psf import extract_stars
from astropy.io import fits
from astropy.table import Table

datadir = '/melkor/d1/guenther/downdata/HST/CepMASTfull/'

# I blinked through the images in ds9 to find single, isolated well-exposed
# stars not too far from the center but outside of the Cepheid PSF and not on
# any of the diffraction spikes.

prflist = [
    ['ibg405010_drz.fits', 340,  38],
    ['ibg416010_drz.fits', 443, 215],
    ['ibg418010_drz.fits', 112, 945],
    ['ibg418010_drz.fits', 112, 945],
    ['ibg422010_drz.fits', 895, 319],
    ['ibg426010_drz.fits', 385,  93],
    ['ibg436010_drz.fits', 342, 877],
    ['ibg438010_drz.fits', 416, 401],
    ['ibg440010_drz.fits', 211, 337],
    ['ibg443010_drz.fits', 359, 288],
    ['ibg444010_drz.fits', 328, 345],
    ['ibg444010_drz.fits', 725, 723],
    ['ibg446010_drz.fits', 276, 500],
    ['ibg453010_drz.fits', 812, 845],
    ['ibg453010_drz.fits', 333, 188],
    ['ibg455010_drz.fits', 263, 444],
    ['ibg456010_drz.fits', 529, 696],
    ['ibg458010_drz.fits', 161, 806],
    ['ibg459010_drz.fits', 374, 166],
    ['ibg465010_drz.fits', 588, 723],
    ['ibg468010_drz.fits', 150, 508],
    ['ibg471010_drz.fits', 600, 685],
    ['ibg471010_drz.fits', 892, 511],
    ]

#prflist = [['ibg402010_drz.fits', 612, 209],
#           ['ibg402010_drz.fits', 1007, 951],
#           ['ibg402010_drz.fits', 488, 705], # GAIA bad
#           ['ibg403010_drz.fits', 597, 385],
#           ['ibg405010_drz.fits', 570, 701], # GAIA bad
#           ['ibg455010_drz.fits', 263, 444],
#           ['ibg456010_drz.fits', 530, 696],
#           ['ibg456010_drz.fits', 549, 462], # GAIA bad
#           ['ibg456010_drz.fits', 860, 408],
#           ['ibg456010_drz.fits', 911, 115],
#           ['ibg465010_drz.fits', 588, 723],
#           ['ibg471010_drz.fits', 600, 685],
#           ['ibg471010_drz.fits', 892, 511],
#]

# -1 because the above positions are measured in ds9, which counts from (1,1)
# while the python code counts from (0,0)
stars621 = extract_stars([NDData(fits.open(datadir + row[0])[1].data) for row in prflist],
                         [Table({'x': [row[1] - 1], 'y': [row[2] - 1]}) for row in prflist],
                         size=25)
stars845 = extract_stars([NDData(fits.open(datadir + row[0].replace('10_', '20_'))[1].data) for row in prflist],
                         [Table({'x': [row[1] - 1], 'y': [row[2] - 1]}) for row in prflist],
                         size=25)


def check_matching_source_exists(l1, l2, d,
                                 xname='xcentroid', yname='ycentroid'):
    '''Check for each source in l1, if one or more sources in l2 are close

    This is not the most efficient way to do things, but very quick to code and
    runtime is not a concern for this.

    Parameters
    ----------
    l1, l2: two source lists
    d : float
         maximal distance in pix, `None` means that all input sources are returned

    Returns
    -------
    ind1 : array
        Array of indices for l1. All elements listed in this index have at least one
        source in l2 within the given distance ``d``.
    '''
    ind1 = []
    for i, s in enumerate(l1):
        dsquared = (s[xname] - l2[xname])**2 + (s[yname] - l2[yname])**2
        if (d is None) or (np.min(dsquared) < d**2):
            ind1.append(i)
    return ind1


def combine_source_tables(list621, list845, names, dmax=10, **kwargs):
    '''Combine source tables. Input are two lists of tables in different bands.

    This function:
    - Only keeps sources if there is a source in the second band within ``dmax`` pixels.
    - Adds a table column with the target name (from input ``names``)
    - stackes everything in one big table.
    '''
    finallist = []
    for i in range(len(list621)):
        l1 = list621[i]
        l2 = list845[i]
        if len(l1) > 0:
            l1['filter'] = 'F621M'
            l1['TARGNAME'] = names[i]
        if len(l2) > 0:
            l2['filter'] = 'F845M'
            l2['TARGNAME'] = names[i]
        if (dmax is not None) and len(l1) > 0 and len(l2) > 0:
            l1short = l1[check_matching_source_exists(l1, l2, dmax, **kwargs)]
            l2short = l2[check_matching_source_exists(l2, l1, dmax, **kwargs)]
            l1 = l1short
            l2 = l2short
        finallist.append(vstack([l1, l2]))
    return vstack(finallist)


class DAOStarAutoThresholdFinder(DAOStarFinder):
    '''An extended DAOStarFinder class.


    '''
    def __init__(self, threshold_scale=5, **kwargs):
        self.threshold_in = threshold_scale
        # Need to set threshold in super__init__ but value will be overwritten below anyway
        super().__init__(threshold=1, **kwargs)

    def __call__(self, data, *args, **kwargs):
        self.threshold = self.threshold_in * np.std(data)
        self.threshold_eff = self.threshold * self.kernel.relerr
        return super().__call__(data, *args, **kwargs)

initial_finder = DAOStarAutoThresholdFinder(fwhm=2.5, threshold_scale=5.,
                                            sharplo=0.55, sharphi=.75,
                                            roundlo=-0.6, roundhi=0.6)
