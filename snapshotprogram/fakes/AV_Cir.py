import sys
sys.path.append('..')
from run_fakes import fake_loop
from photometry import mag2flux

import numpy as np
x = 5 + np.random.rand(2000) * 5
y = 40 + np.random.rand(2000) * 40
x[:1000] = 120 - x[:1000]

fake_loop(x, y,
          (mag2flux(20., 'F621M'), mag2flux(19.1, 'F845M')),
          2000, 'AV-CIR', i_inserted=3)
