import os
from PIL import Image
from io import BytesIO
import base64
import json
import pickle

qa_dir = '../../../datasets/Slake1.0'
img_dir = '../../../datasets/Slake1.0/imgs'
output_dir = '../../../datasets/finetuning/Slake'

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

def extraction(mode):
	if mode == 'train':
		path = os.path.join(qa_dir, 'train.json')
	elif mode == 'val':
		path = os.path.join(qa_dir, 'validate.json')
	elif mode == 'test':
		path = os.path.join(qa_dir, 'test.json')

	output_file_name = os.path.join(output_dir, mode+'.tsv')
	index = 0

	with open(output_file_name, 'w') as out:
		with open(path, "rb") as input_file:
			# data = pickle.load(input_file)
			data = json.load(input_file)
		
		for item in data:
			if item['q_lang'] == 'en':
				img_id = str(item['img_id'])
				question_id = str(item['qid'])
				question = item['question']
				confident_ans = '1|!+' + item['answer']
				object = item['location']

				# image string64base 
				img_name = item['img_name']
				img_path = os.path.join(img_dir, img_name)
				img = Image.open(img_path)
				img_buffer = BytesIO()
				img.save(img_buffer, format=img.format)
				byte_data = img_buffer.getvalue()
				base64_str = base64.b64encode(byte_data)
				base64_str = base64_str.decode("utf-8")
			
				# question_id, image_id, question, answer (with confidence), predicted object labels, image (base64 string)
				out.write(question_id + '\t' + img_id + '\t' + question + '\t' + confident_ans + '\t' + object + '\t' + base64_str + '\n')
				index += 1
				if index% 1000 == 0:
					print("finish '{}' instance {}".format(mode, index))
		
	print("Completed! totally {} '{}' instances".format(index, mode))
	return index

def ans2label():
	path_train = os.path.join(qa_dir, 'train.json')
	path_val = os.path.join(qa_dir, 'validate.json')

	output_file_name = os.path.join(output_dir, 'trainval_ans2label.pkl')
	index = 0

	with open(output_file_name, 'w') as out:
		ans2label = {}

		with open(path_train, "rb") as input_file:
			# data = pickle.load(input_file)
			data = json.load(input_file)
		
		for item in data:
			if item['q_lang'] == 'en':
				confident_ans = item['answer']
				if confident_ans not in ans2label.keys():
					ans2label[confident_ans] = index
					index += 1
					if index% 100 == 0:
						print("finish labeling {} answers".format(index))
		
		with open(path_val, "rb") as input_file:
			# data = pickle.load(input_file)
			data = json.load(input_file)
		
		for item in data:
			if item['q_lang'] == 'en':
				confident_ans = item['answer']
				if confident_ans not in ans2label.keys():
					ans2label[confident_ans] = index
					index += 1
					if index% 100 == 0:
						print("finish labeling {} answers".format(index))	
		
		with open(output_file_name, 'wb') as f:
			pickle.dump(ans2label, f)

	return index


if __name__ == '__main__':
	total = 0
	for mode in ['train', 'val', 'test']:
		num = extraction(mode)
		total += num
	print("Completed! totally {} instances".format(total))
	
	num = ans2label()
	print("Completed! totally {} answers".format(num))


