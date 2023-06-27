lesion_dic = {1: 'bone', 2: 'abdomen', 3: 'mediastinum', 4: 'liver', 
              5: 'lung', 6: 'kidney', 7: 'soft tissue', 8: 'pelvis', -1: 'others'}

import os
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import base64

path_of_the_directory = '.../../../datasets/pretraining/deep_lesion/Key_slices/'
ext = ('.png')

data = pd.read_csv(".../../../datasets/pretraining/deep_lesion/DL_info.csv")

with open(".../../../datasets/pretraining/deep_lesion.tsv", 'w') as out_file:
    index = 0
    for path, dirc, files in os.walk(path_of_the_directory):
        for name in files:
            if name.endswith(ext):
                index += 1
                file_directory = os.path.join(path, name)
                
                # image string64base 
                img = Image.open(file_directory)
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data)
                base64_str = base64_str.decode("utf-8")
                
                # description of objects in an image
                mask = (data['File_name'] == name)
                pos = np.flatnonzero(mask)
                if len(pos)!= 0:
                    total = '' # total description of all objects in an image
                    temp = data.iloc[pos]
                    count = 0
                    for ind,item in temp.T.iteritems():
                        count += 1
                        bounding = item['Bounding_boxes']
                        bounding = bounding.replace(' ', '') #string, strip() to remove whitespaces
                        object_id = item['Coarse_lesion_type'] # int
                        if object_id != -1:
                            object_name = lesion_dic[object_id] # string
                            if count != len(temp):
                                des = bounding + ',' + str(object_id) + ',' + object_name + '&&' #description of an object
                            else:
                                des = bounding + ',' + str(object_id) + ',' + object_name
                            total += des
                    if len(total) != 0:
                        out_file.write('4_' + str(index) + '\t' + base64_str + '\t' + total + '\n')
                    if index % 1000 == 0:
                        print("finish instance", index)
    print("compeleted! # of total instances:", index)