import os

import numpy as np
import pandas as pd
import seaborn as sns
from data_processing import one_hot_encode
for dirname, _, filenames in os.walk('train-jpg/'):
    for filename in filenames:
        print(os.path.join(filename[0:-4]))

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))
data = pd.read_csv("train_v2.csv")
print(data)