import sys
import numpy as np

sys.path.append('..')
from run_fakes import fake_loop, default_mags

ang = 2 * np.pi * np.random.uniform(size=500)
d = float(sys.argv[1])
print(d)
x = d * np.sin(ang) + 61
y = d * np.cos(ang) + 61

for i, mags in enumerate(default_mags):
    fake_loop(x, y, mags, 500, 'r{}_{}'.format(d,i))
