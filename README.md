# PAL: <span style="font-size: 1.5em; color: #1E90FF;">P</span>luralistic <span style="font-size: 1.5em; color: #1E90FF;">AL</span>ignment Framework
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

## 📝 [PAL: Sample-Efficient Personalized Reward Modeling for Pluralistic Alignment](https://openreview.net/pdf?id=1kFDrYCuSu)

[Daiwei Chen](https://chendaiwei-99.github.io), [Yi Chen](https://www.deepneural.network/), [Aniket Rege](https://aniketrege.github.io/), [Zhi Wang](https://zwang.org/), [Ramya Korlakai Vinayak](https://ramyakv.github.io/)

[ 🌐 [PAL Project Page](https://pal-alignment.github.io/) ] [ 📜 [arXiv](https://arxiv.org/abs/2406.08469) ]

[ 📊 [Persona Dataset](https://huggingface.co/datasets/kitkatdafu/persona_in_pal); [Pick-a-Pic Dataset (embeddings)](https://huggingface.co/datasets/ramya-ml/pickapic-embeds);  [Pick-a-Filter Dataset (embeddings)](https://huggingface.co/datasets/ramya-ml/pick-a-filter-embeds) ] 

# 📰 News

- NEW 🔥 01/21/2025: **PAL** has been accepted at **ICLR 2025**.
-  10/09/2024: **PAL** has been accepted at **NeurIPS 2024** workshops: [AFM](https://adaptive-foundation-models.org/), [Behavioral ML](https://sites.google.com/view/behavioralml/), [FITML](https://sites.google.com/view/neurips2024-ftw/home), [Pluralistic-Alignment](https://pluralistic-alignment.github.io/), [SoLaR](https://solar-neurips.github.io/).
- 06/18/2024: **PAL** has been accepted at **ICML 2024** workshops: [TF2M](https://sites.google.com/view/tf2m) and [MFHAIA](https://sites.google.com/view/mhf-icml2024).

# 📍 Overview

This repository contains the code for training and evaluating reward models with ***<u>PAL</u>***, a sample-efficient framework for **P**luralistic **AL**ignment of foundation models. PAL enables efficient reward modeling that caters to <u>***diverse human preferences***</u>, allowing for **personalized adaptation** in both text and image generation tasks. The model balances *<u>**commonalities across users**</u>* with individual-specific customizations, achieving *<u>**few-shot localization**</u>* for new users and reducing the *<u>**sample requirements** to adapt to new users</u>*.

![Ideal Point Model Explained](img.png)

# 💬 Contents

- [Overview](#overview📍)
- [Key Features](#🎯-Key-Features)
- [Installation](#💻-installation)
- [Usage](#🧰-usage)
  - [Data Preparation](##data-preparation)
  - [Configurations](##configurations)
  - [Training](##training)
  - [Integration](##integration)
- [Citation](#📑-citation)

# 🎯 Key Features

💠 <span style="color:lightblue; font-weight:bold;">Diverse Preference Alignment</span>: PAL can handle diverse human preferences rather than assuming a single, universal preference, addressing the variability in individual values.

💠 <span style="color:lightblue; font-weight:bold;">Accuracy-Compute Optimality</span>: e.g. on Reddit TL;DR (Text Summarization), PAL is 1.7% more accurate for seen users and 36% more accurate for unseen users with 20× fewer parameters compared to strong baselines.

💠  <span style="color:lightblue; font-weight:bold;">Modular Design</span>: PAL's architecture is modular, allowing levaraging of shared, common preferences while adapting to specific individual preferences.

💠 <span style="color:lightblue; font-weight:bold;">Few-shot Generalization</span>: PAL enables sample-efficient adaptation to new users' preferences with few labeled examples.

# 💻 Installation

> All code has been tested on Linux with `CUDA=12.4`; functionality on other systems  is not guaranteed.

1. Clone this repository and navigate into the directory

   ```shell
   git clone https://github.com/RamyaLab/pluralistic-alignment.git
   cd pluralistic-alignment
   ```

2. Install required packages with [conda](https://docs.anaconda.com/anaconda/install/)

   ``` sh
   conda env create --file environment.yml
   ```

> Note: Ensure your environment supports **PyTorch** and **CUDA** (if you are using GPU acceleration). The `environment.yml` contains detailed package versions and setup instructions.

 # 🧰 Usage



 ```mermaid
graph LR
    A[Prepare the Preference Dataset with user IDs] --> B[Design Configurations]
    B --> C[Train the PAL Model]
    C --> D[Convert PAL Model into Standard Reward Model]
    D --> E[Ready for Further Applications]
 ```

## Data Preparation

When preparing a dataset of preferences to train and evaluate PAL reward models, each sample should also contain a unique `user_id` to learn each user's preference. The format of each sample should be `(user_id, prompt, (response_1), (response_2)),y`, where `user_id` is a unique string identifier for a specific user;  `prompt`, `response_1`, `response_2` are the prompt and corresponding generative model completions,  $y\in \\{-1, +1\\}$ represents the user's preference over responses. **<u>*(Notice:  modify `dataset_factory.py` to add your own custom dataset)*</u>**.

🎯 *<u>For more details about dataset preparation, please refer to `dataset_factory.py`.</u>*

> We currently only provide <u>*the variant of Reddit TL;DR Summary Dataset*</u> used in the PAL paper in this repository.

## Training Configurations

The [config](config/) folder in this repository contains various configurations that allow for easy training customization. These configuration subfolders are  `ds_config`, `loss_config`, `optim_config`, and `prefLearner_config`.

🎯 <u>*For more details, please review each file individually.*</u>

```
config
├── ds_config
│   ├── summary.yaml
│   └── ...
├── loss_config
│   ├── b-cumulative.yaml
│   └── ...
├── optim_config
│   ├── vanilla-e20.yaml
│   └── ...
└── prefLearner_config
    ├── b-dim512-k2-opt350m-mlp2.yaml
    ├── b-dim768-k2-distillbert65m-mlp2.yaml
    ├── b-dim1024-k2-bgem3-mlp2.yaml
    ├── b-dim1536-k2-qwen1-5b-mlp2.yaml
    ├── b-dim1536-k2-stella1-5b-mlp2.yaml
    └── ...
```

## Training Demos

The following scripts outline different training demos for PAL-B models, targeting various levels of model adaptation and user generalization.

### 1. Train PAL-B-Large

This script trains the PAL-B-Large model by finetuning the foundation model, projector, and user weights.

```sh
# Train PAL-B-Large (Large: finetune the foundation + projectors + user weights)
CUDA_VISIBLE_DEVICES=0 python -u main_pal_b.py \
  --prefLearner_config ./config/prefLearner_config/b-dim512-k2-opt350m-mlp2.yaml \
  --run_name summary-pal-b-large-k2-mlp2 \
  2>&1 >| ./logs/summary-pal-b-large-k2-mlp.log 
```

### 2. Train PAL-B-Tiny

This script trains the PAL-B-Tiny model by fixing the foundation model and only learning the projector and user weights.

```sh
# Train PAL-B-Tiny (Tiny: fix the foundation model and only learn the projectors + user weights)
CUDA_VISIBLE_DEVICES=1 python -u main_pal_b_fix_llm.py \
  --prefLearner_config ./config/prefLearner_config/b-dim512-k2-opt350m-mlp2.yaml \
  --run_name summary-pal-b-tiny-k2-mlp2 \
  2>&1 >| ./logs/summary-pal-b-tiny-k2-mlp2.log 
```

### 3. Train PAL-B (Few-Shot) on New Users

This script performs new user generalization by adapting the model to unseen users. It only learns the weights of new users based on a few samples per user.

```sh
# New User Generalization with n samples per unseen user (Only learn the weights of new users)
CUDA_VISIBLE_DEVICES=2 python -u main_pal_b_unseen.py \
  --ds_config ./config/ds_config/summary_unseen_{num_of_samples_per_unseen_user}samples.yaml \
  --prefLearner_config ./config/prefLearner_config/b-dim512-k2-opt350m-mlp2.yaml \
  --optim_config ./config/optim_config/vanilla-e20.yaml \
  --loss_config ./config/loss_config/b-cumulative.yaml \
  --state_dict_path /path/to/the/well-trained/pal/model.ckpt \
  --run_name summary-unseen-pal-b-cumulative-k2-mlp2-e20-{num_of_samples_per_unseen_user}sample \
  2>&1 >| ./logs/summary-unseen-pal-b-k2-mlp2-{num_of_samples_per_unseen_user}sample.log
```


## Experiment Reproduction
Note: by default, we use five runs $i$ of the below experiments to calculate error bars.
### 1: On Reddit TL;DR Summary dataset, increasing # groups (i.e. the plurality) in our PAL model leads to a significant boost in preference prediction accuracy.
```sh
  # set # user preference groups to 1
  for i in {1..5}; do
      CUDA_VISIBLE_DEVICES=0 python -u main_pal_b.py \
          --prefLearner_config ./config/prefLearner_config/b-dim512-k1-opt350m-mlp2.yaml \
          --run_name summary-b-cumulative-k1-mlp2-run${i} \
          --device 0 \
          2>&1 >| ./logs/summary-b-cumulative-k1-mlp2-${i}.log
  done

  # set # user preference groups to 2
  for i in {1..5}; do
      CUDA_VISIBLE_DEVICES=1 python -u main_pal_b.py \
          --prefLearner_config ./config/prefLearner_config/b-dim512-k2-opt350m-mlp2.yaml \
          --run_name summary-b-cumulative-k2-mlp2-run${i} \
          --device 0 \
          2>&1 >| ./logs/summary-b-cumulative-k2-mlp2-${i}.log
  done
```
### 2. With the freshly trained PAL model above, we can generalize to new, unseen users with very few preference pairs $j$
```sh
for j in 2 5 10 20 50 100; do
  for i in {1..5}; do
      CUDA_VISIBLE_DEVICES=1 python -u main_pal_b_unseen.py \
          --ds_config ./config/ds_config/summary_unseen_${j}samples.yaml \
          --prefLearner_config ./config/prefLearner_config/b-dim512-k2-opt350m-mlp2.yaml \
          --optim_config ./config/optim_config/vanilla-e20.yaml \
          --loss_config ./config/loss_config/b-cumulative.yaml \
          --state_dict_path /path/to/the/model(k=2)/trained/in/stage/1.ckpt \
          --run_name summary-unseen-b-cumulative-k2-mlp2-e20-sample${j}-run${i}
  done
done
```



## Integration

We provide code in the [integration](integration/) subfolder to **convert trained PAL models into standard reward model** for downstram use, e.g. RLHF to train generative models.

> In the standard reward model setup, the reward model takes a prompt and response as input and generates a scalar reward value. 
> In contrast, our PAL model takes a prompt, two responses, and a user ID as input to predict the user’s preference over the two responses given the prompt.

To convert the PAL model to the standard scalar reward model, use the following functions:

- `load_pal_rm_a()`
- `load_pal_rm_b()`

# 📑 Citation

If you find **<u>*PAL*</u>** useful for your research and applications, please consider citing:

```
@inproceedings{chen2025pal,
      title={{PAL}: Sample-Efficient Personalized Reward Modeling for Pluralistic Alignment},
      author={Chen, Daiwei and Chen, Yi and Rege, Aniket and Wang, Zhi and Vinayak, Ramya Korlakai},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=1kFDrYCuSu}
}
```




