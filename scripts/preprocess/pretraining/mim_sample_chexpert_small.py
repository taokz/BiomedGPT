import csv
from PIL import Image
from io import BytesIO
import base64
import os
import argparse
import cv2

ext = ('.jpg')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0, help='0: resize to (256*256), or 1: capture middle part (128*128) string, or 2: base64 string, or 3: combine image-code via vq-gan')
args = parser.parse_args()


if args.mode == 0:
    count = 0
    path_of_the_directory = '.../../../datasets/pretraining/CheXpert-v1.0-small/train'
    save_path_of_the_directory = '.../../../datasets/pretraining/CheXpert-v1.0-small/train_resize'
    for path, dirc, files in os.walk(path_of_the_directory):
        for name in files:
            if name.endswith(ext):
                count += 1

                file_directory = os.path.join(path, name)
                try:

                    # using OpenCV
                    img = cv2.imread(file_directory)
                    # remove trivial background
                    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
                    bbox = cv2.boundingRect(thresholded)
                    x, y, w, h = bbox
                    img = img[y:y+h, x:x+w]
                    # resize to be (256, 256)
                    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                    # save images
                    name = name[:-4] + '_' + str(count) + '.jpg'
                    save_name = os.path.join(save_path_of_the_directory, name)
                    cv2.imwrite(save_name , img)

                except Exception as e:
                    pass 
                if count % 1000 == 0:
                    print("processed", count, "images!")
                continue
    print("compeleted! # of total instances:", count)


if args.mode == 1:
    # used to be fed into VQ-GAN
    path_of_the_directory = '.../../../datasets/pretraining/CheXpert-v1.0-small/train_resize'
    count = 0
    with open(".../../../datasets/pretraining/chexpert_mid_image_string.tsv", 'w') as out_file:
        for path, dirc, files in os.walk(path_of_the_directory):
            for name in files:
                if name.endswith(ext):
                    count += 1
                    file_directory = os.path.join(path, name)
                    # try:
                    img = Image.open(file_directory)
                    
                    # crop middle part (128*128)
                    frac = 0.50
                    left = img.size[0]*((1-frac)/2)
                    upper = img.size[1]*((1-frac)/2)
                    right = img.size[0]-((1-frac)/2)*img.size[0]
                    bottom = img.size[1]-((1-frac)/2)*img.size[1]
                    cropped_img = img.crop((left, upper, right, bottom))
                    
                    img_buffer = BytesIO()
                    cropped_img.save(img_buffer, format=img.format)
                    byte_data = img_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data)
                    base64_str = base64_str.decode("utf-8")
                    out_file.write('1_' + str(count) + '\t' + base64_str + '\n')
                    # except Exception as e:
                    #     pass 
                    if count % 1000 == 0:
                        print("processed", count, "images!")
                    # continue
    print("compeleted! # of total instances:", count)


if args.mode == 2:
    path_of_the_directory = '.../../../datasets/pretraining/CheXpert-v1.0-small/train_resize'
    count = 0
    with open(".../../../datasets/pretraining/chexpert_image_string.tsv", 'w') as out_file:
        for path, dirc, files in os.walk(path_of_the_directory):
            for name in files:
                if name.endswith(ext):
                    count += 1
                    file_directory = os.path.join(path, name)
                    try:
                        img = Image.open(file_directory)
                        img_buffer = BytesIO()
                        img.save(img_buffer, format=img.format)
                        byte_data = img_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data)
                        base64_str = base64_str.decode("utf-8")
                        out_file.write('1_' + str(count) + '\t' + base64_str + '\n')
                    except Exception as e:
                        pass 
                    if count % 1000 == 0:
                        print("processed", count, "images!")
                    continue
    print("compeleted! # of total instances:", count)


if args.mode == 3:
    count = 0
    with open(".../../../datasets/pretraining/chexpert_image_string.tsv", 'r') as f1, open("../a_dataset/chexpert_image_code.tsv", 'r') as f2:
        with open(".../../../datasets/pretraining/chexpert.tsv", 'w') as out:
            for x,y in zip(csv.reader(f1, delimiter='\t'),csv.reader(f2, delimiter='\t')):
                out.write(x[0] + '\t' + x[1] + '\t' + y[1] + '\n')
                count += 1
                if count % 1000 == 0:
                    print("processed", count, "images!")
    print("compeleted! # of total instances:", count)
