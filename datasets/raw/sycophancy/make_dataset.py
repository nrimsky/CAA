import glob 
import json

data = []
for file in glob.glob('sycophancy*.jsonl'):
    with open(file) as f:
        data += [json.loads(line) for line in f]

with open('dataset.json', 'w') as f:
    json.dump(data, f, indent=4)