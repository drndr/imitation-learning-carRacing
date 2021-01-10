import gzip
import os
import pickle
from collections import Mapping
from copy import deepcopy
import pandas as pd

temp1 = None
temp2 = None



with gzip.open('./data1.pkl.gzip','rb') as f:
		temp1 = pickle.load(f)
f.close()
print(len(temp1['state']))
with gzip.open('./data2.pkl.gzip','rb') as g:
		temp2 = pickle.load(g)

g.close()

res = {key: temp1[key] + temp2[key] for key in temp1}
print(len(res['state']))

data_file = os.path.join('data.pkl.gzip')
h = gzip.open(data_file,'wb')
pickle.dump(res, h)
h.close()
