import os
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import base64

path_of_the_directory = '../../../datasets/iu_xray/iu_xray_images'
ext = ('.png')

header_list = ["File_name", "Caption"]
data_train = pd.read_csv("../../../datasets/iu_xray/train_images.tsv", delimiter='\t', names=header_list)
data_test = pd.read_csv("../../../datasets/iu_xray/test_images.tsv", delimiter='\t', names=header_list)

with open(".../../../datasets/finetuning/iu_xray/iu_xray_train_val.tsv", 'w') as out_train, \
    open(".../../../datasets/finetuning/iu_xray/iu_xray_test.tsv", 'w') as out_test:

    index = 0
    index_train = 0
    index_test = 0

    for index, item in data_train.iterrows():
        file_directory = os.path.join(path_of_the_directory, item['File_name'])
        # image string64base 
        img = Image.open(file_directory)
        img_buffer = BytesIO()
        img.save(img_buffer, format=img.format)
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)
        base64_str = base64_str.decode("utf-8")

        caption = item['Caption']
        caption = caption.replace('\t', ' ')

        # uniq_id, image_id, caption, predicted object labels (set to empty string), image (base64 string)
        out_train.write('200_' + str(index+1) + '\t' + item['File_name'].replace(ext, '') + '\t' + \
                        caption + '\t' + str('') + '\t' + base64_str + '\n')
        index += 1
        index_train += 1
        if index_train % 100 == 0:
            print("finish training instance", index_train)

    
    for index, item in data_test.iterrows():
        file_directory = os.path.join(path_of_the_directory, item['File_name'])
        # image string64base 
        img = Image.open(file_directory)
        img_buffer = BytesIO()
        img.save(img_buffer, format=img.format)
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)
        base64_str = base64_str.decode("utf-8")

        caption = item['Caption']
        caption = caption.replace('\t', ' ')

        # uniq_id, image_id, caption, predicted object labels (set to empty string), image (base64 string)
        out_test.write('200_' + str(index+1) + '\t' + item['File_name'].replace(ext, '') + '\t' + \
                        caption + '\t' + str('') + '\t' + base64_str + '\n')
        index += 1
        index_test += 1
        if index_test % 100 == 0:
            print("finish test instance", index_test)
    

    print("compeleted!# of total instances:", index)
