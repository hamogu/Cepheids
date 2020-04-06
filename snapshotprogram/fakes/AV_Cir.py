import sys
sys.path.append('..')
from run_fakes import fake_loop
from photometry import mag2flux

import numpy as np
x = 20 + np.random.rand(200) * 10
y = 40 + np.random.rand(200) * 40

fake_loop(x, y,
          (mag2flux(20., 'F621M'), mag2flux(19.1, 'F845M')),
          200, 'AV-CIR', i_inserted=3)
