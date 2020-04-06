import sys
sys.path.append('..')
from run_fakes import fake_loop
from photometry import mag2flux

import numpy as np
x = 5 + np.random.rand(200)
y = 20 + np.random.rand(200) * 80

fake_loop(x, y,
          (mag2flux(18.5, 'F621M'), mag2flux(17.6, 'F845M')),
          200, 'CO-AUR', i_inserted=11)
