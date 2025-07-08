<!---
Copyright 2022 The OFA-Sys Team. 
Copyright 2023 Kai Zhang @ Lehigh. 
All rights reserved.
This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory.
-->

# [Nature Medicine'24] <img src="examples/logo.jpg" alt="logo" width="35">iomedGPT
*A Generalist Vision-Language Foundation Model for Diverse Biomedical Tasks.* [![Arxiv](https://img.shields.io/badge/arXiv-2305.17100-B21A1B)](https://arxiv.org/abs/2305.17100
)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/biomedgpt-a-unified-and-generalist-biomedical/medical-visual-question-answering-on-vqa)](https://paperswithcode.com/sota/medical-visual-question-answering-on-vqa?p=biomedgpt-a-unified-and-generalist-biomedical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/biomedgpt-a-unified-and-generalist-biomedical/medical-visual-question-answering-on-pathvqa)](https://paperswithcode.com/sota/medical-visual-question-answering-on-pathvqa?p=biomedgpt-a-unified-and-generalist-biomedical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/biomedgpt-a-unified-and-generalist-biomedical/image-captioning-on-iu-x-ray)](https://paperswithcode.com/sota/image-captioning-on-iu-x-ray?p=biomedgpt-a-unified-and-generalist-biomedical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/biomedgpt-a-unified-and-generalist-biomedical/text-summarization-on-meqsum)](https://paperswithcode.com/sota/text-summarization-on-meqsum?p=biomedgpt-a-unified-and-generalist-biomedical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/biomedgpt-a-unified-and-generalist-biomedical/natural-language-inference-on-mednli)](https://paperswithcode.com/sota/natural-language-inference-on-mednli?p=biomedgpt-a-unified-and-generalist-biomedical)

**BiomedGPT** is pre-trained and fine-tuned with multi-modal & multi-task biomedical datasets. Details of used datasets are shown in [datasets.md](datasets.md). If you have any questions, feel free to contact us or post issues. 

- **[2025/07/07]** Released larger-scale checkpoints—up to 5× larger (930M parameters)—including stronger *large* and *xlarge* pre-trained models. [[ckpt](https://www.dropbox.com/sh/cu2r5zkj2r0e6zu/AADZ-KHn-emsICawm9CM4MqVa?dl=0)] [[technical report](https://arxiv.org/pdf/2505.17436)]

## Installation (Linux)

1. Clone this repository and navigate to the BiomedGPT folder
```bash
git clone https://github.com/taokz/BiomedGPT
cd BiomedGPT/
```

2. Install required packages
```Shell
conda create --name biomedgpt python=3.7.4
python -m pip install pip==21.2.4
pip install -r requirements.txt
```

### Quick Start with Huggingface's transformers

Please check out this [Colab notebook](https://colab.research.google.com/drive/1AMG-OwmDpnu24a9ZvCNvZi3BZwb3nSfS?usp=sharing) for Fairseq-free inference. 

**Warning:** Extensive experiments using transformers have not been conducted, so we cannot confirm whether the results from transformers and fairseq are fully aligned.

## Checkpoints
We provid pretrained checkpoints of BiomedGPT (<a href="https://www.dropbox.com/sh/cu2r5zkj2r0e6zu/AADZ-KHn-emsICawm9CM4MqVa?dl=0">Dropbox</a>), which can be put in the `scripts/` folder for further development. For finetuned checkpoints, please refer to [checkpoints.md](checkpoints.md). 

transformers-compatible weights are accessible through the [collection ](https://huggingface.co/collections/PanaceaAI/biomedgpt-v1-66ca7c51e378662e15178be3).

### Note:
We emphasize that BiomedGPT, including its files, code, and checkpoints, is strictly for academic research purposes. Commercial and clinical uses are strictly prohibited for three key reasons: First, BiomedGPT is based on the OFA framework, which carries a non-commercial license that we have inherited. Second, our model is not licensed for use in healthcare settings. Finally, we have not implemented sufficient security measures, and the current model cannot guarantee the accuracy required for medical diagnoses.


## Implementation
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

### Zero-shot VQA inference using pre-trained checkpoints
Add ```--zero-shot``` argument in the script. Example script: ```/scripts/vqa/evaluate_vqa_rad_zero_shot.sh```.

**Warning:** The current implementation is not yet designed for chatbot or copilot applications, as its primary focus is on learning general representations in medicine that can be transferred to downstream tasks, as outlined in our paper. Large-scale training and instruction tuning for improving robust conversational abilities are still in progress.

## Downstreams
We provide the run scripts of fine-tuning and inference. There will be log files during execution. Before fine-tuning or inference, please refer to 
<details>
    <summary><b>Visual Question Answering</b></summary>
<pre>
cd scripts/vqa
# for fine-tuning
bash train_vqa_rad_beam.sh
# for inference using fine-tuned weights
bash evaluate_vqa_rad_beam.sh
# for zero-shot inference using instruction-tuned weights
bash evaluate_vqa_rad_unconstrained.sh
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

@article{peng2025scaling,
  title={Scaling Up Biomedical Vision-Language Models: Fine-Tuning, Instruction Tuning, and Multi-Modal Learning},
  author={Peng, Cheng and Zhang, Kai and Lyu, Mengxian and Liu, Hongfang and Sun, Lichao and Wu, Yonghui},
  journal={arXiv preprint arXiv:2505.17436},
  year={2025}
}
```
<br></br>
