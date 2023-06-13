import os
import pandas as pd
from PIL import Image
from io import BytesIO
import base64

path_of_the_directory = '../../../datasets/roco-dataset/data'
ext = ('.jpg')

header_list = ["File_name", "Caption"]

with open(".../../../datasets/finetuning/roco/roco_train.tsv", 'w') as out_train, \
    open(".../../../datasets/finetuning/roco/roco_test.tsv", 'w') as out_test, \
    open(".../../../datasets/finetuning/roco/roco_val.tsv", 'w') as out_val:

    index = 0
    index_train = 0
    index_val = 0
    index_test = 0
    for path, dirc, files in os.walk(path_of_the_directory):
        for name in files:
            if name == 'captions.txt':
                caption_directory = os.path.join(path, name)
                data = pd.read_csv(caption_directory, delimiter='\t', names=header_list)
                
                for idx, row in data.iterrows():
                    file_name = row['File_name']
                    caption = row['Caption']
                    file_directory = os.path.join(path, 'images', file_name+ext)

                    try:
                        # image string64base 
                        img = Image.open(file_directory)
                        img_buffer = BytesIO()
                        img.save(img_buffer, format=img.format)
                        byte_data = img_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data)
                        base64_str = base64_str.decode("utf-8")

                        if 'train' in caption_directory:
                            # uniq_id, image_id, caption, predicted object labels (set to empty string), image (base64 string)
                            out_train.write('1500_' + str(index+1) + '\t' + file_name + '\t' + \
                                            caption + '\t' + str('') + '\t' + base64_str + '\n')
                            index += 1
                            index_train += 1
                            if index_train % 100 == 0:
                                print("finish training instance", index_train)
                        
                        elif 'validation' in caption_directory:
                            # uniq_id, image_id, caption, predicted object labels (set to empty string), image (base64 string)
                            out_val.write('1500_' + str(index+1) + '\t' + file_name + '\t' + \
                                            caption + '\t' + str('') + '\t' + base64_str + '\n')
                            index += 1
                            index_val += 1
                            if index_val % 100 == 0:
                                print("finish validation instance", index_val)          

                        elif 'test' in caption_directory:
                            # uniq_id, image_id, caption, predicted object labels (set to empty string), image (base64 string)
                            out_test.write('1500_' + str(index+1) + '\t' + file_name + '\t' + \
                                            caption + '\t' + str('') + '\t' + base64_str + '\n')
                            index += 1
                            index_test += 1
                            if index_test % 100 == 0:
                                print("finish test instance", index_test)
                    except:
                        continue      

    print("compeleted!# of total instances:", index)
