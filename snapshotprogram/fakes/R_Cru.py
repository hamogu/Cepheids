import sys
sys.path.append('..')
from run_fakes import fake_loop
from photometry import mag2flux

import numpy as np
x = np.ones(200)
y = np.ones(200)
x[:100] = 56 - np.random.rand(100) * 5
x[100:] = 64 + np.random.rand(100) * 5
y = 13 - 3 + np.random.rand(200) * 6


fake_loop(x, y,
          (mag2flux(15.36, 'F621M'), mag2flux(14.33, 'F845M')),
          200, 'R-CRU', i_inserted=21)
