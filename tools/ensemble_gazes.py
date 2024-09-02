import numpy as np
import os
import sys

num=15
test_type = 'eth'
exp_dir = sys.argv[1]
file_name = 'within_eva_results.txt'

if len(sys.argv) == 2:
    exp_dir = sys.argv[1]


if 'mpii' in exp_dir:
    sub_dirs = [f'{i:02}' for i in range(15)]
elif 'diap' in exp_dir:
    sub_dirs = ['00_01_02_03', '04_05_06_07', '08_09_10_11', '12_13_14_15']

results = []

for sub_dir in sub_dirs:
    path = os.path.join(exp_dir, sub_dir, test_type, file_name)
    print(path)
    res = np.loadtxt(path, delimiter=',')
    results.append(res)

res_mat = np.array(results)
res_mat = np.mean(res_mat, axis=0)

out_dir = os.path.join(exp_dir, test_type)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
res_path = os.path.join(out_dir, 'within_eva_results.txt')
np.savetxt(res_path, res_mat, delimiter=',')
print('Save results in %s' % (res_path))