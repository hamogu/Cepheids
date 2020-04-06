import sys
sys.path.append('..')
from run_fakes import fake_loop
from photometry import mag2flux

import numpy as np
x = 10 + np.random.rand(200) * 20
y = x - 1 + 2 * np.random.rand(200)
#y[50:] = 120 - x


fake_loop(x, y,
          (mag2flux(15.1, 'F621M'), mag2flux(14.7, 'F845M')),
          200, 'BG-CRU', i_inserted=9)
