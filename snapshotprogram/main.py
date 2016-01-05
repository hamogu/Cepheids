from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Column, MaskedColumn, vstack

import detection

outdir = '/melkor/d1/guenther/projects/Cepheids/HSTsnapshot/'
fitslist = glob('/melkor/d1/guenther/processing/Cepheids/PSFsubtr/?????????_drz.fits')

F621Mfiles = np.array([f for f in fitslist if (fits.getval(f, 'FILTER') == 'F621M')])
F621Mnames = [fits.getval(f, 'TARGNAME') for f in F621Mfiles]
F621Mfiles = F621Mfiles[np.argsort(F621Mnames)]


F845Mfiles = np.array([f for f in fitslist if (fits.getval(f, 'FILTER') == 'F845M')])
F845Mnames = [fits.getval(f, 'TARGNAME') for f in F845Mfiles]
F845Mfiles = F845Mfiles[np.argsort(F845Mnames)]

# Vegamag zero points are here http://www.stsci.edu/hst/wfc3/phot_zp_lbn

# Units in image are electrons/s
# Images are drz -> drizzeled -> pixels are all same size.


def flux2magF621M(x):
    return -2.5 * np.log10(x) + 24.4539


def flux2magF845M(x):
    return -2.5 * np.log10(x) + 23.2809

halfwidth = 50
daofindkwargs = {'fwhm': 1.5, 'threshold': 7, 'roundlo': -0.8, 'roundhi': 0.8}

images, targets = detection.read_images(F621Mfiles, halfwidth)
fluxes, imout, scaledout = detection.photometryloop(images, targets, **daofindkwargs)
fluxes.add_column(MaskedColumn(flux2magF621M(fluxes['flux_fit']), 'mag_fit'))
fluxes.add_column(MaskedColumn(['F621M'] * len(fluxes), 'filter'))
#detection.plot_gallery('PSF subtr. - linear scale', imout, 10, 7, sources=fluxes)
#detection.plot_gallery('PSF subtr. - funny scale', scaledout, 10, 7, sources=fluxes)
# Get rid of negative fluxes. They must be fit artifacts
# Investigate later where they come from.
fluxes = fluxes[~fluxes['mag_fit'].mask]


images845, targets845 = detection.read_images(F845Mfiles, halfwidth)
fluxes845, imout845, scaledout845 = detection.photometryloop(images845, targets845, **daofindkwargs)
fluxes845.add_column(MaskedColumn(flux2magF845M(fluxes845['flux_fit']), 'mag_fit'))
fluxes845.add_column(Column(['F845M'] * len(fluxes845), 'filter'))
#detection.plot_gallery('PSF subtr. - linear scale', imouts845, 10, 7, sources=fluxes845)
#detection.plot_gallery('PSF subtr. - funny scale', scaledout845, 10, 7, sources=fluxes845)
fluxes845 = fluxes845[~fluxes845['mag_fit'].mask]



fl = vstack([fluxes, fluxes845])
fl.sort(['name', 'x_0_fit'])
# Throw out all objects that appear only once ->
# They are clearly not detected in both bands
# Still need to check manually for objects with detections in both filters
# to see if those describe the same object.
# fl = fl[np.isfinite(fl['mag_fit'])]
bothbands = np.array([((i in (fluxes['id'])) and (i in (fluxes845['id']))) for i in fl['id']])


for c in ['x_0_fit', 'y_0_fit', 'r', 'PA', 'mag_fit']:
    fl[c].format = '%5.1f'

print fl['id', 'name', 'filter', 'x_0_fit', 'y_0_fit', 'r', 'PA','mag_fit'][bothbands]
'''
 id   name   filter x_0_fit y_0_fit   r     PA  mag_fit
--- -------- ------ ------- ------- ----- ----- -------
  5   AX-CIR  F845M    45.5    65.2   0.6 333.7    13.4
  5   AX-CIR  F621M    45.9    60.5   0.4 338.7    11.2
 14  ETA-AQL  F845M    44.2    33.6   0.7  92.8     9.1
 14  ETA-AQL  F621M    44.9    32.6   0.7  96.1     9.8
 21    R-CRU  F845M    45.4    97.4   1.9 344.4    14.5
 21    R-CRU  F621M    45.8    95.5   1.8 344.1    15.5
 30    S-NOR  F845M    27.2    47.3   0.9 260.3    11.6
 30    S-NOR  F621M    27.7    47.0   0.9 261.2    11.7
 42    U-AQL  F845M    55.4    11.0   1.6 224.1    11.3
 42    U-AQL  F621M    55.9     9.9   1.6 224.5    11.7
 45    U-VUL  F845M    28.1    80.9   1.5 321.1    15.1
 45    U-VUL  F621M    28.5    81.9   1.5 319.8    16.3
 58 V659-CEN  F845M    60.0    61.4   0.6 248.5    11.7
 58 V659-CEN  F845M    62.8    58.4   0.6 233.0    11.6
 58 V659-CEN  F621M    68.4    58.2   0.8 223.8    14.8

It's easy to see that the V659-CEN companion is real, but unfortunately
there is some detector artifact (cosmic?) that distorts it's shape and thus
the source finding algorithm has an offset position for F621M. Visually, it is
obvious that only a fraction of the flux is in the aperture and thus the flux
must be underestimated. In F845M the source finding algorithm finds two separate
peaks and treats this source as a binary. The data is insufficient to decide if this
is just an artifact of the fitting procedure or if there really are two sources.

'''

# In this it's easy to cross-match by eye
# but we might want a proper function here
# Or I just let Nancy do it by hand...

detection.plot_gallery('PSF subtr. - F621M', imout[:,:,35:], targets[35:], 5,7, sources=fluxes[35:],color='r', arrowlength=0.5)
plt.savefig(outdir + 'F621M_part2.png')
detection.plot_gallery('PSF subtr. - F621M', imout[:,:,:35], targets[:35], 5,7, sources=fluxes[:35],color='r', arrowlength=0.5)
plt.savefig(outdir + 'F621M_part1.png')

detection.plot_gallery('PSF subtr. - F845M', imout845[:,:,:35], targets845[:35], 5,7, sources=fluxes845[:35],color='r', arrowlength=0.5)
plt.savefig(outdir + 'F845M_part1.png')
detection.plot_gallery('PSF subtr. - F845M', imout845[:,:,35:], targets845[35:], 5,7, sources=fluxes845[35:],color='r', arrowlength=0.5)
plt.savefig(outdir + 'F845M_part2.png')


# ########### SIMULATIONS ###################

# ### Runs a > 1 hour ###
sim = detection.run_sim(images, n=10000, **daofindkwargs)
sim845 = detection.run_sim(images845, n=10000, **daofindkwargs)

posOK = (np.abs(sim['x'] - sim['x_0_fit']) < 2.) & (np.abs(sim['y'] - sim['y_0_fit']) < 2.)


sim.add_column(Column(flux2magF621M(sim['flux']), 'mag'))
sim.add_column(MaskedColumn(flux2magF621M(sim['flux_fit']), 'mag_fit'))

rbins = np.array([5, 12, 20, 30, 60])
magbins = np.array(np.arange(6, 20))

h2, xedges, yedges = np.histogram2d(sim['r'], sim['mag'], bins=[rbins, magbins])

# independent of how bad the flux is
h2f, xedges, yedges = np.histogram2d(sim['r'][posOK], sim['mag'][posOK], bins=[rbins, magbins])

plt.imshow(h2, interpolation='nearest')
plt.colorbar()

# A better way to plot that might be this:
magmids = 0.5 * (magbins[1:] + magbins[:-1])

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

# will need different line styles etc.
for i, c, s in zip(range(h2.shape[0]), 'kbrg', ['-', '--', ':', '-.']):
    ax.plot(magmids, h2f[i,:]/h2[i,:], label='{0:3.1f}"-{1:3.1f}"'.format(rbins[i]*0.04, rbins[i+1]*0.04), lw=2, c=c, ls=s)
    # could even plot uncertainties here... or run MC longer...

ax.legend(loc='lower left')
ax.set_ylabel('Detection probability')
ax.set_xlabel('Source magnitude')
ax.set_xlim([np.min(magmids), np.max(magmids)])
fig.subplots_adjust(left=0.2, right=0.90, top=0.97)
fig.savefig('simdetect.png')


'''
Subtraction might work better on undrizzeled images; it could be that the
diffraction spikes that don't quite fit are images with a different drizzle
correction.

On the other hand, photometry is easier on drizzeled images
(but can also be done on undrizzeled images).

I don't think I want to open that can of worms for me at this point.
'''
# How good is the flux (or mag?)
from scipy.stats import binned_statistic_2d
meanhist = binned_statistic_2d(sim['r'][posOK], sim['mag'][posOK], (sim['mag']-sim['mag_fit'])[posOK], bins=[rbins, magbins], statistic='mean')
stdhist = binned_statistic_2d(sim['r'][posOK], sim['mag'][posOK], (sim['mag']-sim['mag_fit'])[posOK], bins=[rbins, magbins], statistic=np.std)


fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

# will need different line styles etc.
for i, c, s in zip(range(stdhist[0].shape[0]), 'kbrg', ['-', '--', ':', '-.']):
    ax.plot(magmids, stdhist[0][i,:], label='{0:3.1f}"-{1:3.1f}"'.format(rbins[i]*0.04, rbins[i+1]*0.04), lw=2, c=c, ls=s)
    # could even plot uncertainties here... or run MC longer...

ax.legend(loc='upper left')
ax.set_ylabel('Uncertainty [mag]')
ax.set_xlabel('Source magnitude')
ax.set_xlim([np.min(magmids), np.max(magmids)])
ax.set_ylim([0, 1])
fig.subplots_adjust(left=0.2, right=0.90, top=0.97)
fig.savefig('simerror.png')


# look at individual distributions
plt.hist((sim['mag'] - sim['mag_fit'])[posOK][stdhist[3] == 42])
'''Some have wide (1.5 mag) outliers, but in general a shifted Gaussian
seem not too bad as a description, so meanhist and stdhist might be good descriptors.
All lines are systematically shifted, but that shift is smaller than the
new scatter introduced, so I can probably ignore that for now.
(but keep an eye on it when we change the other parameters.)

The new scatter is also so big, that I am confident that it is much larger than
the uncertainty of the normal photometry, so we can just use that as error.
'''



'''
Next plans:
- Looks into some sources by hand: #5 has bad centering, try to improve in order to improve the subtraction
 -> The centroiding is at most 0.5 pixel off (probably less), that is as good as it gets for integer pixel values. We will have to live with this.

- Looking at the images we are pretty close to the noise limit, but we could go crazy and implement a LOCI algorithm. http://iopscience.iop.org/0004-637X/660/1/770/fulltext/  - np.linalg.solve
'''




# #### for some manual tests
''' aplpy plots are not perfect (I run in to problems trying to make subplot
with aplpy, but at least the "where is north" question should have been
debugged there well enough.
So, this little routine helped me debug the orientation of my images and
my north vectors.
'''

import aplpy


def plot_aplpy(title, image, target, sources={'id': np.array([None])}):
        target.reinsert_image(image)
        hdu = fits.PrimaryHDU(target.data)
        hdu.header = target.header

        af = aplpy.FITSFigure(hdu)
        af.show_colorscale(vmin=1., vmax=np.max(target.data), smooth=None, stretch='log')

        af.add_label(0.1, 0.9, detection.targname(target.targname), relative=True,
                     color='r', weight='bold')

        ind = (sources['id'] == i)
        if ind.sum() > 0:
            af.show_circles(sources[ind]['ra'], sources[ind]['dec'], 0.0001, color='r')
        #af.canvas.draw()
        af.recenter(target.header['CRVAL1'], target.header['CRVAL2'], 0.0007)
        af.add_grid()
        return af



AXCIR = targets[5]
AXCIR.reinsert_image(imout[:, :, 5])
hdu = fits.PrimaryHDU(AXCIR.data)
hdu.header = AXCIR.header
af = aplpy.FITSFigure(hdu, figsize=(6,5))
af.show_colorscale(vmid=-40,vmin=-30, vmax=np.max(AXCIR.data), smooth=None, stretch='log')
#af.show_colorscale(vmin=1, vmax=np.percentile(imout[:,:,5],99), smooth=None, stretch='log')
af.set_title(detection.targname(AXCIR.targname))

ind = (fluxes['name'] == 'AX-CIR')
if ind.sum() > 0:
    af.show_circles(fluxes[ind]['ra'], fluxes[ind]['dec'], 0.00007, color='r', linewidth=4)

raceph, decceph = AXCIR.all_pix2world(50,50)
af.show_markers([raceph], [decceph], color='b', linewidth=30, marker='+')
af.set_theme("publication")
af.recenter(AXCIR.header['CRVAL1'], AXCIR.header['CRVAL2'], 0.0005)
af.add_grid()
af.grid.set_color('k')
af.grid.set_linewidth(2)
from astropy import units as u
af.add_scalebar(1 * u.arcsecond)
af.scalebar.set_label('1 arcsec')
af.scalebar.set_color('k')
af.scalebar.set_linewidth(3)


# Experimenting with different PSF fitting...

images, targets = detection.read_images(F621Mfiles, halfwidth)
masked_images = np.ma.array(images)

for i in range(70):
    threshold = 0.6 * np.max(images[:,:,i])
    masked_images[masked_images[:,:,i] > threshold, i] = np.ma.masked

fimages, normperim, medianim, mastermask = detection.apply_normmask(masked_images.reshape((101*101,-1)), mastermask=np.zeros(101*101, dtype=bool))
imout = np.zeros_like(fimages)

for i in range(fimages.shape[1]):
        print "working on image {0}".format(i)
        otherpsfs = np.delete(np.arange(fimages.shape[1]), i)
        psfbase = fimages[:, otherpsfs]
        psfsubobject = psfsubclass(fimages[:,i].reshape((101,101)), psfbase.reshape((101,101,-1)))
        psf = psfsubobject.fit_psf()
        imout[:, i] = fimages[:,i] - psf

imag = detection.remove_normmask(imout, normperim, medianim, mastermask).reshape((101,101,70))



detection.plot_gallery('PSF subtr. - F621M', imag[:,:,35:], targets[35:], 5,7, color='r', arrowlength=0.5)
plt.savefig(outdir + 'F621M_part2.png')
