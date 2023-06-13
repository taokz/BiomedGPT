import sys
from tqdm import tqdm
import json
import numpy as np
import random

'''
For fair comprison, we adopt the data spilt used by BioBART
The following split and convert codes are modified based on
https://github.com/GanjinZero/BioBART/blob/main/downstream_src/dataprepare/icliniq.py
'''

data_dir='.../../../datasets/finetuning/icliniq/'

with open(sys.argv[1], 'r') as f:
    raw_docs = f.readlines()

raw_samples = ('\n' + ''.join(raw_docs)).split('\nid=')
for idx, item in enumerate(raw_samples):
    if not item:
        continue
    if '\n\nDescription\n' not in item or '\n\nDialogue\n' not in item:
        raise AttributeError(f'Bad Samples: {item}{idx}')
    tmp = item.split('\n\nDescription\n')
    tmp = tmp[:-1] + tmp[-1].split('\n\nDialogue\n')
    raw_samples[idx] = tmp
print(len(raw_samples))

cleaned_samples = []
for idx, item in tqdm(enumerate(raw_samples)):
    if not item:
        continue
    sample = {'id':None, 'tgt':None, 'src':None}
    sample['id'] = idx
    sample['tgt'] = item[1].replace('\n', ' ').strip('Q. ')
    sample['src'] = item[2].replace('\n', ' ').strip(' ')
    cleaned_samples.append(sample)



indices = np.arange(len(cleaned_samples))
random.shuffle(indices)
train_data, dev_data, test_data = [], [], []
for i, idx in enumerate(indices):
    if i < 24851:
        train_data.append(cleaned_samples[idx])
    elif i >= 24851 and i < 24851 + 3105:
        dev_data.append(cleaned_samples[idx])
    else:
        test_data.append(cleaned_samples[idx])

with open(data_dir + '/train.json', 'w') as f:
    json.dump({'data':train_data}, f, indent=2)

with open(data_dir + '/dev.json', 'w') as f:
    json.dump({'data':dev_data}, f, indent=2)

with open(data_dir + '/test.json', 'w') as f:
    json.dump({'data':test_data}, f, indent=2)

print(f'sample numbers, overall {len(indices)}, train {len(train_data)}, dev {len(dev_data)}, test {len(test_data)}.')

'''
json to tsv format for BiomedGPT
'''
for file_name in ['train','dev','test']:
    data_path = os.path.join(data_dir, file_name+'.json')
    f = open(data_path)
    data_dict = json.load(f)['data']
    if file_name == 'dev':
        name = 'val'
    else:
        name = file_name
    outfile_name = os.path.join(data_dir, name+'.tsv')
    with open(outfile_name, 'w') as outfile:
        for data in data_dict:
            target_text = str(data['tgt']).replace('\n', ' ').replace('\t', ' ').strip()
            source_text = str(data['src']).replace('\n', ' ').replace('\t', ' ').strip()
            outfile.write(source_text + '\t' + target_text + '\n')