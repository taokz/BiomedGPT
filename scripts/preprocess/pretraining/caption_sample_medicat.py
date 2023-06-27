import os
import pandas as pd
from PIL import Image
from io import BytesIO
import base64

output_dir = '.../../../datasets/pretraining'
ref_dir = '.../../../datasets/pretraining/medicat/s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl'
img_dir = '.../../../datasets/pretraining/medicat/figures'

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

output_file_name = os.path.join(output_dir, 'medicat.tsv')

with open(output_file_name, 'w') as out_file:
    index = 1
    jsonObj = pd.read_json(path_or_buf=ref_dir, lines=True)
    for ind,item in jsonObj.T.iteritems():
        try:
            img_id = item['pdf_hash'] + '_' + item['fig_uri']
            # image string64base 
            img_path = os.path.join(img_dir, img_id)
            img = Image.open(img_path)
            img_buffer = BytesIO()
            img.save(img_buffer, format=img.format)
            byte_data = img_buffer.getvalue()
            base64_str = base64.b64encode(byte_data)
            base64_str = base64_str.decode("utf-8")

            '''
            uniq-id, image (base64 string), caption, question, answer, 
            ground-truth objects (objects appearing in the caption or question), 
            dataset name (source of the data),
            and task type (caption, qa or visual gronunding)
            '''
            caption = '.'.join(item['s2_caption'].split('.')[1:]).strip() # remove unneccessary caption such as 'figure 4. ---'
            
            out_file.write('7_' + str(index) + '\t' + base64_str + '\t' + caption + '\t' + \
                          str('') + '\t' + str('') + '\t' + str('') + '\t' + \
                          'medicat' + '\t' + 'caption' + '\n')
            index += 1
            if index% 10000 == 0:
                print("finish '{}' instance {}".format('caption', index))
        except:
            continue

    print("Completed! totally {} '{}' instances".format(index, 'caption'))