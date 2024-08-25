<!---
Copyright 2022 The OFA-Sys Team. 
Copyright 2023 Kai Zhang @ Lehigh. 
All rights reserved.
This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory.
-->

# BiomedGPT
[BiomedGPT](https://arxiv.org/abs/2305.17100) is pre-trained and fine-tuned with multi-modal & multi-task biomedical datasets. Details of used datasets are shown in [datasets.md](datasets.md). If you have any questions, feel free to contact us or post issues. 

Please check out this [Colab notebook](https://colab.research.google.com/drive/1AMG-OwmDpnu24a9ZvCNvZi3BZwb3nSfS?usp=sharing) for Fairseq-free inference. Warning: Extensive experiments using transformers have not been conducted, so we cannot confirm whether the results from transformers and fairseq are fully aligned.

# Checkpoints
We provid pretrained checkpoints of BiomedGPT (<a href="https://www.dropbox.com/sh/cu2r5zkj2r0e6zu/AADZ-KHn-emsICawm9CM4MqVa?dl=0">Dropbox</a>), which can be put in the `scripts/` folder for further development. For finetuned checkpoints, please refer to [checkpoints.md](checkpoints.md). 
## Note:
We emphasize that BiomedGPT, including its files, code, and checkpoints, is strictly for academic research purposes. Commercial and clinical uses are strictly prohibited for three key reasons: First, BiomedGPT is based on the OFA framework, which carries a non-commercial license that we have inherited. Second, our model is not licensed for use in healthcare settings. Finally, we have not implemented sufficient security measures, and the current model cannot guarantee the accuracy required for medical diagnoses.

<br></br>

# Installation
```bash
git clone https://github.com/taokz/BiomedGPT
conda env create -f biomedgpt.yml
python -m pip install pip==21.2.4
pip install fairseq
```
<br></br>


# Implementation
We provide the preprocessing, pretraining, finetuning and inference scripts in the `scripts/` folder. You can follow the directory setting below:

```
BiomedGPT/
├── checkpoints/
├── datasets/
│   ├── pretraining/
│   ├── finetuning/
│   └── ...
├── scripts/
│   ├── preprocess/
│   │   ├── pretraining/
│   │   └── finetuning/
│   ├── pretrain/
│   ├── vqa/
│   └── ...
└── ...
```

## Pretraining
Please follow [datasets.md](datasets.md) to prepare pretraining datasets, which includes 4 TSV files: <code>vision_language.tsv</code>, <code>text.tsv</code>, <code>image.tsv</code> and <code>detection.tsv</code> in the directory of `./datasets/pretraining/`.

<pre>
cd scripts/pretrain
bash pretrain_tiny.sh
</pre>
Feel free to modify the hyperparameters in the bash script for your requirements or ablation study.
<br></br>

## Downstreams
We provide the run scripts of fine-tuning and inference. There will be log files during execution. Before fine-tuning or inference, please refer to 
<details>
    <summary><b>Visual Question Answering</b></summary>
<pre>
cd scripts/vqa
# for fine-tuning
bash train_vqa_rad_beam.sh
# for inference
bash evaluate_vqa_rad_beam.sh
</pre>
</details>
<details>
    <summary><b>Image Captioning</b></summary>
<pre>
cd scripts/caption
# for fine-tuning
bash train_peir_gross.sh
# for inference
bash evaluate_peir_gross.sh
</pre>
</details>
<details>
    <summary><b>Text Summarization</b></summary>
<pre>
cd scripts/text_sum
# for fine-tuning
bash train_meqsum.sh
# for inference
bash evaluate_meqsum.sh
</pre>
</details>
<details>
    <summary><b>Natural Language Inference</b></summary>
<pre>
cd scripts/mednli
# for fine-tuning
bash train_mednli.sh
# for inference
bash evaluate_mednli.sh
</pre>
</details>
<details>
    <summary><b>Image Classification</b></summary>
<pre>
cd scripts/image_cls
# for fine-tuning: I provide a template, please set different hyparameters for each dataset in MedMNIST if required.
bash train_medmnist.sh 
# for inference: a template
bash evaluate_medmnist.sh
</pre>
</details>

<br></br>

# Related Codebase
* [OFA](https://github.com/OFA-Sys/OFA)
* [Fairseq](https://github.com/pytorch/fairseq)
* [taming-transformers](https://github.com/CompVis/taming-transformers)
* [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)
* [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
<br></br>


# Citation
If you use BiomedGPT model or our code for publications, please cite 🤗: 
```
@article{zhang2024generalist,
  title={A generalist vision--language foundation model for diverse biomedical tasks},
  author={Zhang, Kai and Zhou, Rong and Adhikarla, Eashan and Yan, Zhiling and Liu, Yixin and Yu, Jun and Liu, Zhengliang and Chen, Xun and Davison, Brian D and Ren, Hui and others},
  journal={Nature Medicine},
  pages={1--13},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```
<br></br>
