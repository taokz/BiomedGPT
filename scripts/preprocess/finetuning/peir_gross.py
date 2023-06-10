import os
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import base64

path_of_the_directory = '../../../datasets/peir_gross/peir_gross_images'
ext = ('.jpg')

header_list = ["File_name", "Caption"]
data_train = pd.read_csv("../../../datasets/peir_gross/train_images.tsv", delimiter='\t', names=header_list)
data_test = pd.read_csv("../../../datasets/peir_gross/test_images.tsv", delimiter='\t', names=header_list)

with open(".../../../datasets/finetuning/peir_gross/peir_gross_train.tsv", 'w') as out_train, \
    open("../../../datasets/finetuning/peir_gross/peir_gross_val.tsv", 'w') as out_val, \
    open("../../../datasets/finetuning/peir_gross/peir_gross_test.tsv", 'w') as out_test:

    index = 0
    index_train = 0
    train_val_split = len(data_train) * 0.8
    for path, dirc, files in os.walk(path_of_the_directory):
        for name in files:
            if name.endswith(ext):
                file_directory = os.path.join(path, name)
                
                # image string64base 
                img = Image.open(file_directory)
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data)
                base64_str = base64_str.decode("utf-8")
                
                # captions of an image (train and val)
                mask = (data_train['File_name'] == name)
                pos = np.flatnonzero(mask)
                if len(pos)!= 0:
                    temp = data_train.iloc[pos]
                    for ind,item in temp.T.iteritems():
                        caption = item['Caption']
                        caption = caption.replace('\t', ' ')
                        if index_train < train_val_split:
                            # uniq_id, image_id, caption, predicted object labels (set to empty string), image (base64 string)
                            out_train.write('3_' + str(index+1) + '\t' + name.replace(ext, '') + '\t' + \
                                            caption + '\t' + str('') + '\t' + base64_str + '\n')
                            index += 1
                            index_train += 1
                            if index_train % 100 == 0:
                                print("finish training instance", index_train)
                        else:
                            out_val.write('3_' + str(index+1) + '\t' + name.replace(ext, '') + '\t' + \
                                            caption + '\t' + str('') + '\t' + base64_str + '\n')
                            index += 1
                            index_train += 1
                            if index_train % 100 == 0:
                                print("finish valid instance", index_train)

                # captions of an image (test)
                mask = (data_test['File_name'] == name)
                pos = np.flatnonzero(mask)
                if len(pos)!= 0:
                    temp = data_test.iloc[pos]
                    for ind,item in temp.T.iteritems():
                        caption = item['Caption']
                        caption = caption.replace('\t', ' ')
                        out_test.write('3_' + str(index+1) + '\t' + name.replace(ext, '') + '\t' + \
                                        caption + '\t' + str('') + '\t' + base64_str + '\n')
                        index += 1
                        if index % 100 == 0:
                            print("finish test instance", index)

    print("compeleted!# of total instances:", index)
