import os
from PIL import Image
from io import BytesIO
import base64
import pickle

qa_dir = '../../../datasets/PathVQA/qas'
img_dir = '../../../datasets/PathVQA/images'
output_dir = '../../../datasets/finetuning/PathVQA'


if not os.path.exists(output_dir):
	os.mkdir(output_dir)

def extraction(mode):
	if mode == 'train':
		path = os.path.join(qa_dir, 'train_vqa.pkl')
	elif mode == 'val':
		path = os.path.join(qa_dir, 'val_vqa.pkl')
	elif mode == 'test':
		path = os.path.join(qa_dir, 'test_vqa.pkl')

	output_file_name = os.path.join(output_dir, mode+'.tsv')
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
			
			# question_id, image_id, question, answer (with confidence), predicted object labels (set to empty string), image (base64 string)
			out.write(question_id + '\t' + img_id + '\t' + question + '\t' + confident_ans + '\t' + str('') + '\t' + base64_str + '\n')
			index += 1
			if index% 1000 == 0:
				print("finish '{}' instance {}".format(mode, index))
		
	print("Completed! totally {} '{}' instances".format(index, mode))
	return index

if __name__ == '__main__':
	total = 0
	for mode in ['train', 'val', 'test']:
		num = extraction(mode)
		total += num
	print("Completed! totally {} instances".format(total))



