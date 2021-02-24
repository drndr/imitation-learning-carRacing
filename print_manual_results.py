import json
import pickle
import gzip

f = open('sample_results.json')
result = json.load(f)
print(result)
with gzip.open('./data/data.pkl.gzip','rb') as g:
    data = pickle.load(g)
sample_size = 0
for i in range(len(data['state'])):
    sample_size += len(data['state'][i])
print("sample size: ", sample_size)
print(len(data['state'][0]))
g.close
f.close
