import json
import pandas as pd
import csv

data_dir='.../../../datasets/finetuning/mednli'

def read_nli_data(filename, set_genre='clinical', limit=None):
    """
    Read NLI data and return a DataFrame
    Optionally, set the genre column to `set_genre`
    """

    if limit is None:
        limit = float('inf')  # we could use math.inf in Python 3.5 :'(

    all_rows = []
    with open(str(filename)) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row['gold_label'] != '-':
                all_rows.append(row)

            if i > limit:
                break

    nli_data = pd.DataFrame(all_rows)

    if set_genre is not None:
        nli_data['genre'] = set_genre

    return nli_data

# train
data_train = read_nli_data(data_dir + '/mli_train_v1.jsonl')
with open(data_dir + "/mednli_train.tsv", 'w') as out_file:
    for index, row in data_train.iterrows():
        if row['gold_label'] == 'entailment':
            label = str(1)
        elif row['gold_label'] == 'contradiction':
            label = str(2)
        else:
            label = str(0)
        out_file.write(row['sentence1'].replace("\t", " ").strip() + '\t' + row['sentence2'].replace("\t", " ").strip() + '\t' + label + '\t' + row['genre'] + ':' + \
                       row['sentence1'].replace("\t", " ").strip() + '\n')


# valid
data_dev = read_nli_data(data_dir + '/mli_dev_v1.jsonl')
with open(data_dir + "/mednli_val.tsv", 'w') as out_file:
    for index, row in data_dev.iterrows():
        if row['gold_label'] == 'entailment':
            label = str(1)
        elif row['gold_label'] == 'contradiction':
            label = str(2)
        else:
            label = str(0)
        out_file.write(row['sentence1'].replace("\t", " ").strip() + '\t' + row['sentence2'].replace("\t", " ").strip() + '\t' + label + '\t' + row['genre'] + ':' + \
                       row['sentence1'].replace("\t", " ").strip() + '\n')


# test
data_test = read_nli_data(data_dir + '/mli_test_v1.jsonl')
with open(data_dir + "/mednli_test.tsv", 'w') as out_file:
    for index, row in data_test.iterrows():
        if row['gold_label'] == 'entailment':
            label = str(1)
        elif row['gold_label'] == 'contradiction':
            label = str(2)
        else:
            label = str(0)
        out_file.write(row['sentence1'].replace("\t", " ").strip() + '\t' + row['sentence2'].replace("\t", " ").strip() + '\t' + label + '\t' + row['genre'] + ':' + \
                       row['sentence1'].replace("\t", " ").strip() + '\n')