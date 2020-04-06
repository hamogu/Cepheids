import sys
sys.path.append('..')
from run_fakes import fake_loop
from photometry import mag2flux

import numpy as np
x = np.ones(200)
y = np.ones(200)
x[:100] = 56 - np.random.rand(100) * 5
x[100:] = 64 + np.random.rand(100) * 5
y = 100 - 3 + np.random.rand(200) * 6


fake_loop(x, y,
          (mag2flux(11.87, 'F621M'), mag2flux(11., 'F845M')),
          200, 'U-AQL', i_inserted=42)
