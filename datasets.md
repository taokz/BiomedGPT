# Datasets

We provide links to download the raw datasets below, and we share our preprocessed python scripts in the `./scripts/preprocess/` folder. Before processing data, we need to put the downloaded compressed file into `./datasets/` and uncompress it (**change the folder's name if required**). You can also process the data on your own according to the instructions given by <a href="https://github.com/OFA-Sys/OFA#image-processing"> OFA </a>. There are several useful notes below.

## Pretraining Datasets
 The pretraining datasets used in BiomedGPT are all accessible. However, you should request the access to some datasets. Here we provide the public links to these data, it is recommended that you download the data from the links first, and then process the downloaded dataset using our scripts.
-   _MedICat_:  https://github.com/allenai/medicat
-   _IU X-ray and Peir Gross_:  https://github.com/nlpaueb/bioCaption
-   _SLAKE_:  https://www.med-vqa.com/slake/
-   _PathVQA_:  https://github.com/UCSD-AI4H/PathVQA/tree/master/data
-   _DeepLesion_:  https://nihcc.app.box.com/v/DeepLesion
-   _OIA-DDR_:  https://github.com/nkicsl/DDR-dataset
-   _CheXpert_:  https://aimi.stanford.edu/chexpert-chest-x-rays
-   _CytoImageNet_: https://github.com/stan-hua/CytoImageNet
-   _ISIC2020_: https://challenge2020.isic-archive.com
-   _Retinal Fundus_: https://www.kaggle.com/c/diabetic-retinopathy-detection
-   _PubMed Abstracts_: https://github.com/ncbi-nlp/BLUE_Benchmark
-   _NCBI BioNLP_: https://www.ncbi.nlm.nih.gov/research/bionlp/Data/
-   _MIMIC-III Clinic Notes_: https://physionet.org/content/mimiciii/1.4/
<br></br>

## Downstream Datasets
Partial datasets are used in the pretraining 
-   _MedMNIST v2_:  https://zenodo.org/record/6496656
-   _MeQSum_:  https://github.com/abachaa/MeQSum
-   _iCliniq and HealthCareMagic_:  https://github.com/UCSD-AI4H/Medical-Dialogue-System
-   _ROCO_:  https://github.com/razorx89/roco-dataset/tree/master
-   _VQA-RAD_:  https://vision.aioz.io/f/777a3737ee904924bf0d/?dl=1
<br></br>


## Data Preparation Notes
- PathVQA's `trainval_ans2label.pkl` is located in `PathVQA/split/qas`.
- Before preprocessing the VQA-RAD dataset, it's necessary to inspect the data and search for any instances of `\t`. These instances might cause issues and it's recommended to manually remove them. For instance, changing instances like `slee\t n` to `sleen`. Neglecting this step and proceeding with preprocessing could lead to errors during training.
- For preprocessing the `MedMNIST` dataset, the following steps are employed: First, the `.npy` files are converted to `.png` images using the command `python medmnist.py --mode 0`. Subsequently, these `.png` images are converted into a `.tsv` file using the command `--mode 1`.