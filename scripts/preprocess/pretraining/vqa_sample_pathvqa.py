import os
from PIL import Image
from io import BytesIO
import base64
import pickle

qa_dir = '.../../../datasets/pretraining/PathVQA/qas'
img_dir = '.../../../datasets/pretraining/PathVQA/images'
output_dir = '.../../../datasets/pretraining'


if not os.path.exists(output_dir):
	os.mkdir(output_dir)

def extraction(mode):
	if mode == 'train':
		path = os.path.join(qa_dir, 'train_vqa.pkl')
	else:
		raise Exception("Only training data can be preprocessed in the Pretraining.")

	output_file_name = os.path.join(output_dir, 'pathvqa.tsv')
	index = 0

	with open(output_file_name, 'w') as out:
		with open(path, "rb") as input_file:
			data = pickle.load(input_file)
		
		for item in data:
			img_id = item['img_id']
			question_id = str(item['question_id'])
			question = item['sent']
			confident_ans = '1|!+' + list(item['label'].keys())[0]

			# image string64base 
			img_path = os.path.join(img_dir, mode, img_id + '.jpg')
			img = Image.open(img_path)
			img_buffer = BytesIO()
			img.save(img_buffer, format=img.format)
			byte_data = img_buffer.getvalue()
			base64_str = base64.b64encode(byte_data)
			base64_str = base64_str.decode("utf-8")
			
			# unique_id, image (base64 string), caption (empty for vqa task), question, answer, ground-truth objects (objects appearing in the caption or question, empty here), dataset name, task type
			out.write(img_id + '\t' + base64_str + '\t' + str('') + '\t' \
			 			+ question + '\t' + confident_ans + '\t' + str('') + '\t' + 'pathvqa' \
						+ '\t' + 'qa' + '\n')
			index += 1
			if index% 1000 == 0:
				print("finish '{}' instance {}".format(mode, index))
		
	print("Completed! totally {} '{}' instances".format(index, mode))
	#total += index
	return index

if __name__ == '__main__':
	total = 0
	for mode in ['train']:
		num = extraction(mode)
		total += num
	print("Completed! totally {} instances".format(total))



