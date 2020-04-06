import sys
sys.path.append('..')
from run_fakes import fake_loop

fake_loop(30, 30, (3e4, 3e4), 100, 'diag_1')
fake_loop(30, 30, (1e4, 1e4), 100, 'diag_2')
fake_loop(30, 30, (3e3, 3e3), 100, 'diag_3')
