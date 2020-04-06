import sys
sys.path.append('..')
from run_fakes import fake_loop

#fake_loop(10, 30, (1e5, 1e5), 100, 'outside_0')
#fake_loop(10, 30, (1e4, 1e4), 100, 'outside_1')
#fake_loop(10, 30, (1e3, 1e3), 100, 'outside_2')
#fake_loop(10, 30, (2e2, 2e2), 100, 'outside_3')

import numpy as np
x = np.random.rand(100) * 10 + 10
y = np.random.rand(100) * 80 + 20
fake_loop(x, y, (2e4, 2e4), 100, 'outside_i7_0', i_inserted=7)
