import sys
sys.path.append('..')
from run_fakes import fake_loop
from photometry import mag2flux

import numpy as np

phi = 2 * np. pi  * np.random.rand(200)


def index(phi):
    '''Close to diffraction spike or bleed column?'''
    return((np.mod(phi, np.pi) < 0.1) |
           (np.mod(phi, np.pi) > 3) |
           (np.abs(phi - np.pi * 3 / 2) < 0.1) |
           (np.abs(np.mod(phi, np.pi / 2) - np.pi / 4) < 0.1))

ind = index(phi)
while ind.sum() > 0:
    phi[ind] = 2 * np. pi  * np.random.rand(ind.sum())
    ind = index(phi)


x = 60 + 23 * np.sin(phi)
y = 60 + 23 * np.cos(phi)

fake_loop(x, y,
          (mag2flux(11.4, 'F621M'), mag2flux(11.2, 'F845M')),
          200, 'S-NOR', i_inserted=30)
