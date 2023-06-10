import os
from PIL import Image
from io import BytesIO
import base64
import pickle
import json

qa_dir = '../../../datasets/data_RAD'
img_dir = '../../../datasets/data_RAD/images'
output_dir = '../../../datasets/finetuning/VQA-RAD'


if not os.path.exists(output_dir):
	os.mkdir(output_dir)

def extraction(mode=None):
	ans2label = {}
	label_idx = 0
	total = 0
	ans2label_file = os.path.join(output_dir, 'trainval_ans2label_pubmedclip.pkl')
	for mode in ['train_val', 'test']:
		if mode == 'train_val':
			path = os.path.join(qa_dir, 'trainset.json')
			# ans2label_file = os.path.join(output_dir, 'trainval_ans2label.pkl')
		elif mode == 'test':
			path = os.path.join(qa_dir, 'testset.json')

		output_file_name = os.path.join(output_dir, mode+'.tsv')
		index = 0

		with open(output_file_name, 'w') as out:
			with open(path, "rb") as input_file:
				data = json.load(input_file)
			
			for item in data:
				img_id = item['image_name'][:-4]
				question_id = str(item['qid'])
				question = item['question'].replace('\t', '').lower().strip()
				ans = str(item['answer']).replace('\t', '').lower().strip()
				confident_ans = '1|!+' + ans
				object_name = item['image_organ'].replace('\t', '').lower()

				# image string64base 
				try:
					img_path = os.path.join(img_dir, img_id + '.jpg')
					# print(img_path)
					img = Image.open(img_path)
					img_buffer = BytesIO()
					img.save(img_buffer, format=img.format)
					byte_data = img_buffer.getvalue()
					base64_str = base64.b64encode(byte_data)
					base64_str = base64_str.decode("utf-8")
					# base64_str = base64.urlsafe_b64encode(byte_data).decode("utf-8") # bytes 
					
					# question_id, image_id, question, answer (with confidence), predicted object labels (from data), image (base64 string)
					out.write(question_id + '\t' + img_id + '\t' + question + '\t' + confident_ans + '\t' + object_name + '\t' + base64_str + '\n')
					index += 1
					
					# if mode == 'train_val':
					if ans not in ans2label.keys():
						print("add answer '{}' into the ans2label dic. Now we have {} answers.".format(ans, len(ans2label)))
						ans2label[ans] = int(label_idx)
						label_idx += 1
				except Exception as e:
					continue
				
				if index% 100 == 0:
					print("finish '{}' instance {}".format(mode, index))
			
		print("Completed! totally {} '{}' instances".format(index, mode))
		total += index

		if mode == 'test': # follow pubmedclip
			with open(ans2label_file, 'wb') as f:
				pickle.dump(ans2label, f)

	return total

if __name__ == '__main__':
	# total = 0
	# for mode in ['train_val', 'test']:
	#for mode in ['test']:
		# total += extraction(mode)
	mode = None
	total = extraction(mode)
	print("Completed! totally {} instances".format(total))



