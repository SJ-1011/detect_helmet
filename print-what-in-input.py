import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk('./kaggle/input/helmet-detection'):
    for filename in filenames:
        print(os.path.join(dirname, filename))