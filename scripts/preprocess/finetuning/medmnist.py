#import csv
from PIL import Image
from io import BytesIO
import base64
import os
import argparse
import numpy as np
import pickle


INFO_DICT = {
    "pathmnist": {
        "python_class": "PathMNIST",
        "description":
        "The PathMNIST is based on a prior study for predicting survival from colorectal cancer histology slides, providing a dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task. We resize the source images of 3×224×224 into 3×28×28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The CRC-VAL-HE-7K is treated as the test set.",
        "url":
        "https://zenodo.org/record/6496656/files/pathmnist.npz?download=1",
        "MD5": "a8b06965200029087d5bd730944a56c1",
        "task": "multi-class",
        "label": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 89996,
            "val": 10004,
            "test": 7180
        },
        "license": "CC BY 4.0"
    },
    "dermamnist": {
        "python_class": "DermaMNIST",
        "description":
        "The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images categorized as 7 different diseases, formulized as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3×600×450 are resized into 3×28×28.",
        "url":
        "https://zenodo.org/record/6496656/files/dermamnist.npz?download=1",
        "MD5": "0744692d530f8e62ec473284d019b0c7",
        "task": "multi-class",
        "label": {
            "0": "actinic keratoses and intraepithelial carcinoma",
            "1": "basal cell carcinoma",
            "2": "benign keratosis-like lesions",
            "3": "dermatofibroma",
            "4": "melanoma",
            "5": "melanocytic nevi",
            "6": "vascular lesions"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 7007,
            "val": 1003,
            "test": 2005
        },
        "license": "CC BY 4.0"
    },
    "octmnist": {
        "python_class": "OCTMNIST",
        "description":
        "The OCTMNIST is based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal diseases. The dataset is comprised of 4 diagnosis categories, leading to a multi-class classification task. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are gray-scale, and their sizes are (384−1,536)×(277−512). We center-crop the images and resize them into 1×28×28.",
        "url":
        "https://zenodo.org/record/6496656/files/octmnist.npz?download=1",
        "MD5": "c68d92d5b585d8d81f7112f81e2d0842",
        "task": "multi-class",
        "label": {
            "0": "choroidal neovascularization",
            "1": "diabetic macular edema",
            "2": "drusen",
            "3": "normal"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 97477,
            "val": 10832,
            "test": 1000
        },
        "license": "CC BY 4.0"
    },
    "pneumoniamnist": {
        "python_class": "PneumoniaMNIST",
        "description":
        "The PneumoniaMNIST is based on a prior dataset of 5,856 pediatric chest X-Ray images. The task is binary-class classification of pneumonia against normal. We split the source training set with a ratio of 9:1 into training and validation set and use its source validation set as the test set. The source images are gray-scale, and their sizes are (384−2,916)×(127−2,713). We center-crop the images and resize them into 1×28×28.",
        "url":
        "https://zenodo.org/record/6496656/files/pneumoniamnist.npz?download=1",
        "MD5": "28209eda62fecd6e6a2d98b1501bb15f",
        "task": "binary-class",
        "label": {
            "0": "normal",
            "1": "pneumonia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 4708,
            "val": 524,
            "test": 624
        },
        "license": "CC BY 4.0"
    },
    "retinamnist": {
        "python_class": "RetinaMNIST",
        "description":
        "The RetinaMNIST is based on the DeepDRiD challenge, which provides a dataset of 1,600 retina fundus images. The task is ordinal regression for 5-level grading of diabetic retinopathy severity. We split the source training set with a ratio of 9:1 into training and validation set, and use the source validation set as the test set. The source images of 3×1,736×1,824 are center-cropped and resized into 3×28×28.",
        "url":
        "https://zenodo.org/record/6496656/files/retinamnist.npz?download=1",
        "MD5": "bd4c0672f1bba3e3a89f0e4e876791e4",
        "task": "ordinal-regression",
        "label": {
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 1080,
            "val": 120,
            "test": 400
        },
        "license": "CC BY 4.0"
    },
    "breastmnist": {
        "python_class": "BreastMNIST",
        "description":
        "The BreastMNIST is based on a dataset of 780 breast ultrasound images. It is categorized into 3 classes: normal, benign, and malignant. As we use low-resolution images, we simplify the task into binary classification by combining normal and benign as positive and classifying them against malignant as negative. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images of 1×500×500 are resized into 1×28×28.",
        "url":
        "https://zenodo.org/record/6496656/files/breastmnist.npz?download=1",
        "MD5": "750601b1f35ba3300ea97c75c52ff8f6",
        "task": "binary-class",
        "label": {
            "0": "malignant",
            "1": "normal, benign"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 546,
            "val": 78,
            "test": 156
        },
        "license": "CC BY 4.0"
    },
    "bloodmnist": {
        "python_class": "BloodMNIST",
        "description":
        "The BloodMNIST is based on a dataset of individual normal cells, captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. It contains a total of 17,092 images and is organized into 8 classes. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images with resolution 3×360×363 pixels are center-cropped into 3×200×200, and then resized into 3×28×28.",
        "url":
        "https://zenodo.org/record/6496656/files/bloodmnist.npz?download=1",
        "MD5": "7053d0359d879ad8a5505303e11de1dc",
        "task": "multi-class",
        "label": {
            "0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 11959,
            "val": 1712,
            "test": 3421
        },
        "license": "CC BY 4.0"
    },
    "organamnist": {
        "python_class": "OrganAMNIST",
        "description":
        "The OrganAMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). It is renamed from OrganMNIST_Axial (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in axial views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/6496656/files/organamnist.npz?download=1",
        "MD5": "866b832ed4eeba67bfb9edee1d5544e6",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 34581,
            "val": 6491,
            "test": 17778
        },
        "license": "CC BY 4.0"
    },
    "organcmnist": {
        "python_class": "OrganCMNIST",
        "description":
        "The OrganCMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). It is renamed from OrganMNIST_Coronal (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in coronal views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/6496656/files/organcmnist.npz?download=1",
        "MD5": "0afa5834fb105f7705a7d93372119a21",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 13000,
            "val": 2392,
            "test": 8268
        },
        "license": "CC BY 4.0"
    },
    "organsmnist": {
        "python_class": "OrganSMNIST",
        "description":
        "The OrganSMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). It is renamed from OrganMNIST_Sagittal (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in sagittal views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/6496656/files/organsmnist.npz?download=1",
        "MD5": "e5c39f1af030238290b9557d9503af9d",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 13940,
            "val": 2452,
            "test": 8829
        },
        "license": "CC BY 4.0"
    },
}

path_of_the_directory = '../../../datasets/MedMNIST/2d'
ext = ('.npz')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0, help='0: convert numpy to png, or 1: preprocess')
args = parser.parse_args()

if args.mode == 0:
    for path, dirc, files in os.walk(path_of_the_directory):
        for name in files:
            if name.endswith('.npz'):
                data_cat = name[:-4]
                cat_directory = os.path.join(path, data_cat)
                os.mkdir(cat_directory)
                train_directory = os.path.join(cat_directory, 'train')
                os.mkdir(train_directory)
                val_directory = os.path.join(cat_directory, 'val')
                os.mkdir(os.path.join(cat_directory, 'val'))
                test_directory = os.path.join(cat_directory, 'test')
                os.mkdir(os.path.join(cat_directory, 'test'))
                
                file_directory = os.path.join(path, name)
                data = np.load(file_directory)
                for data_type in data.files:
                    if data_type == 'train_images':
                        count = 0
                        for im_array in data[data_type]:
                            img = Image.fromarray(im_array).convert('RGB')
                            count += 1
                            save_name = os.path.join(train_directory, str(count)+'.png')
                            img.save(save_name)
                            if count % 1000 == 0:
                                print("processed", count, data_cat, "training images!")
                    elif data_type == 'val_images':
                        count = 0
                        for im_array in data[data_type]:
                            img = Image.fromarray(im_array).convert('RGB')
                            count += 1
                            save_name = os.path.join(val_directory, str(count)+'.png')
                            img.save(save_name)
                            if count % 1000 == 0:
                                print("processed", count, data_cat, "val images!")
                    elif data_type == 'test_images':
                        count = 0
                        for im_array in data[data_type]:
                            img = Image.fromarray(im_array).convert('RGB')
                            count += 1
                            save_name = os.path.join(test_directory, str(count)+'.png')
                            img.save(save_name) 
                            if count % 1000 == 0:
                                print("processed", count, data_cat, "test images!")

elif args.mode == 1:
    train_data_name = ''
    val_data_name = ''
    test_data_name = ''
    path_of_the_directory = '../../../datasets/MedMNIST/2d'
    save_path_of_the_directory = '.../../../datasets/finetuning/MedMNIST'
    if not os.path.exists(save_path_of_the_directory):
        os.mkdir(save_path_of_the_directory)
        print("The MedMNIST directory is created!")
    for path, dirc, files in os.walk(path_of_the_directory):
        for name in files:
            if name.endswith('.npz'):
                data_cat = name[:-4]
                train_data_name = os.path.join(save_path_of_the_directory, data_cat+'_train.tsv')
                val_data_name = os.path.join(save_path_of_the_directory, data_cat+'_val.tsv')
                test_data_name = os.path.join(save_path_of_the_directory, data_cat+'_test.tsv')

                path_of_the_directory_sub = os.path.join(path_of_the_directory, data_cat)
                with open(train_data_name, 'w') as out_train, \
                    open(val_data_name, 'w') as out_val, \
                    open(test_data_name, 'w') as out_test:

                    info_file = np.load(os.path.join(path, name))
                    train_labels = info_file['train_labels']
                    val_labels = info_file['val_labels']
                    test_labels = info_file['test_labels']

                    train_count = 0
                    val_count = 0
                    test_count = 0
                    for path_sub, dir_sub, files_sub in os.walk(path_of_the_directory_sub):
                        for name_sub in files_sub:
                            file_directory = os.path.join(path_sub, name_sub)
                            img = Image.open(file_directory)
                            img_buffer = BytesIO()
                            img.save(img_buffer, format=img.format)
                            byte_data = img_buffer.getvalue()
                            base64_str = base64.b64encode(byte_data)
                            base64_str = base64_str.decode("utf-8")
                            if 'train' in path_sub:
                                label_index = str(train_labels[int(name_sub[:-4])-1][0]) #np.int8
                                label_name = INFO_DICT[data_cat]['label'][label_index]
                                out_train.write(base64_str + '\t' + label_index + '\t' + label_name + '\n')
                                train_count += 1
                            elif 'val' in path_sub:
                                label_index = str(val_labels[int(name_sub[:-4])-1][0]) #np.int8
                                label_name = INFO_DICT[data_cat]['label'][label_index]
                                out_val.write(base64_str + '\t' + label_index + '\t' + label_name + '\n')
                                val_count += 1
                            elif 'test' in path_sub:
                                label_index = str(test_labels[int(name_sub[:-4])-1][0]) #np.int8
                                label_name = INFO_DICT[data_cat]['label'][label_index]
                                out_test.write(base64_str + '\t' + label_index + '\t' + label_name + '\n')
                                test_count += 1
                    
                    print("compeleted preprocessing", data_cat)
                    print("# of training instances:", train_count, "; # of val instances:", val_count, "; # of test instances:", test_count)
