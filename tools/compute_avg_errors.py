import os
import sys
import numpy as np
import pandas as pd

err_path = os.path.join(sys.argv[1])

err_tab = pd.read_csv(err_path, ',', header=None, index_col=0)

if len(sys.argv) > 2:
    num = int(sys.argv[2])
    if num > 0:
        err_tab = err_tab.head(num)
    elif num < 0:
        err_tab = err_tab.tail(-num)

print(err_tab)

print(f"\nMean errors: ")
print(err_tab.mean(axis=0))
