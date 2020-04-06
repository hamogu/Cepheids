import os
import pickle

import numpy as np
from astropy.table import vstack
import sherpa
import sherpa.astro.models

from fake import insert_fake_fit
from photometry import phot_sherpa, mag2flux
from detection import combine_source_tables as cst
from detection import initial_finder, stars621, stars845

datadir = '/melkor/d1/guenther/downdata/HST/CepMASTfull/'
fakedir = '/melkor/d1/guenther/processing/Cepheids/fake/'

default_mags = [(mag2flux( 8, 'F621M'), mag2flux( 9, 'F845M')),
                (mag2flux(10, 'F621M'), mag2flux(11, 'F845M')),
                (mag2flux(12, 'F621M'), mag2flux(13, 'F845M')),
                (mag2flux(14, 'F621M'), mag2flux(15, 'F845M')),
                (mag2flux(16, 'F621M'), mag2flux(17, 'F845M')),
                (mag2flux(18, 'F621M'), mag2flux(19, 'F845M')),
                (mag2flux(20, 'F621M'), mag2flux(21, 'F845M')),
            ]


psf_621 = sherpa.astro.models.Beta2D(name='psf_621')
psf_621.alpha = 2.2
psf_621.r0 = 1.7
psf_621.alpha.frozen = True
psf_621.r0.frozen = True

psf_845 = sherpa.astro.models.Beta2D(name='psf_845')
psf_845.alpha = 2
psf_845.r0 = 1.6
psf_845.alpha.frozen = True
psf_845.r0.frozen = True


with open(os.path.join(fakedir, 'Cephsubtracted.pickle'), 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f)

specs621 = (data['F621Marr'], data['baseF621'], data['normperimF621'],
            data['medianimF621'])
specs845 = (data['F845Marr'], data['baseF845'], data['normperimF845'],
            data['medianimF845'])


def fake_loop(xin, yin, fluxin, n, outname, i_inserted=None):
    src = []
    fakeout = []
    for j in range(n):
        print('working on fake {}/{}'.format(j, n))
        x0 = xin if np.isscalar(xin) else xin[j]
        y0 = yin if np.isscalar(yin) else yin[j]
        flux = (fluxin[0] if np.isscalar(fluxin[0]) else fluxin[0][j],
                fluxin[1] if np.isscalar(fluxin[1]) else fluxin[1][j])
        i, fake621, mednorm_ins621, red_ins621 = insert_fake_fit(x0, y0,
                                                                 flux[0],
                                                                 stars621,
                                                                 *specs621,
                                                                 i_inserted=i_inserted)
        i, fake845, mednorm_ins845, red_ins845 = insert_fake_fit(x0, y0,
                                                                 flux[1],
                                                                 stars845,
                                                                 *specs845,
                                                                 i_inserted=i)
        src_recovered = cst([initial_finder(mednorm_ins621, mask=mednorm_ins621.mask)],
                            [initial_finder(mednorm_ins845, mask=mednorm_ins845.mask)],
                            ['FAKE_{}'.format(j)], dmax=2.5,
                            xname='xcentroid', yname='ycentroid')
        if len(src_recovered) >= 1:
            src_recovered['xin'] = x0
            src_recovered['yin'] = y0
            src_recovered['f621in'] = flux[0]
            src_recovered['f845in'] = flux[1]
            src.append(src_recovered)

        fakeindex = ['FAKE_{}'.format(j)]

        for row in src_recovered:
            sherpaout = phot_sherpa(np.atleast_3d(red_ins621),
                                    np.atleast_3d(red_ins845),
                                    fakeindex.index, row,
                                    psf_621, psf_845, maxdxdy=2)
            sherpaout['xin'] = x0
            sherpaout['yin'] = y0
            sherpaout['f621in'] = flux[0]
            sherpaout['f845in'] = flux[1]
            fakeout.append(sherpaout)
    fakeoutt = vstack(fakeout)
    srct = vstack(src)
    for tab in [fakeoutt, srct]:
        tab.meta['n_runs'] = n
    fakeoutt.write(os.path.join(fakedir, outname + '_phot.fits'), overwrite=True)
    srct.write(os.path.join(fakedir, outname + '_detect.fits'), overwrite=True)
