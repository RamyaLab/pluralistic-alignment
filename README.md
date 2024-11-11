# PAL: Personalized Reward Modeling / Pluralistic Alignment Framework

Sample-Efficient Personalized Reward Modeling for Pluralistic Alignment

[Daiwei Chen](), [Yi Chen](https://www.deepneural.network/), [Aniket Rege](https://aniketrege.github.io/), [Zhi Wang](https://zwang.org/), [Ramya Korlakai Vinayak](https://ramyakv.github.io/)

[ 🌐 [PAL Project Page](https://pal-alignment.github.io/) ] [ 📜 [arXiv](https://arxiv.org/abs/2406.08469) ] [ 🤗 [HuggingFace (TODO)]() ] [ 📊 [Demo Datasets (TODO)]() ]

# News 📰

- 🔥 [NEW!] **PAL** has been accepted at **2024 NIPS** workshops: [AFM](https://adaptive-foundation-models.org/), [Behavioral ML](https://sites.google.com/view/behavioralml/), [FITML](https://sites.google.com/view/neurips2024-ftw/home), [Pluralistic-Alignment](https://pluralistic-alignment.github.io/), [SoLaR](https://solar-neurips.github.io/).
- **PAL** is under review for **2024 ICLR** conference
- **PAL** has been accepted at **2024 ICML** workshops: [TF2M](https://sites.google.com/view/tf2m) and [MFHAIA](https://sites.google.com/view/mhf-icml2024).

# Overview📍

This repository contains the code for ***<u>PAL</u>*** (Personalized Alignment Learning), a novel framework for **pluralistic alignment** in foundation models. PAL enables efficient reward modeling that caters to <u>***diverse human preferences***</u>, allowing for **personalized adaptation** in both text and image generation tasks. The model balances *<u>**commonalities across users**</u>* with individual-specific customizations, achieving *<u>**few-shot localization**</u>* for new users and reducing the *<u>**sample requirements per user**</u>*.

# Contents 💬

- [Overview](#overview📍)
- [Key Features](#Key-Features)
- [Installation](#Installation)
- [Usage](#usage)
  - [Data Preparation](##data-preparation)
  - [Training](##training)
  - [Evaluation](##evaluation)
  - [Integration](##integration)

- [Demo](#demo)
- [Contributing](#contributing)

# Key Features 🎯

💠 <u>***Diverse Preference Alignment***</u>: PAL can handle diverse human preferences rather than assuming a single, universal preference, addressing the variability in individual values.

💠 <u>***Higher Performance with Fewer Parameters***</u>: e.g. For a T2T task, PAL is 1.7% more accurate for seen users and 36% more accurate for unseen users, with 20× fewer parameters.

💠 <u>***Modular Design***</u>: PAL's architecture is modular, allowing it to leverage shared common preferences while adapting to specific individual preferences.

💠 <u>***Few-shot Generalization***</u>: PAL enables quick adaptation to new users' preferences with few examples, making it more efficient for personalized alignment.

# Installation 💻

> All code has been tested on Linux; functionality on other systems is not guaranteed.

1. Clone this repository and navigate into the PAL

   ```shell
   git clone https://github.com/RamyaLab/pluralistic-alignment.git
   cd pluralistic-alignment
   ```

2. Install Packages

   ``` sh
   pip install -r requirements.txt
   ```

> Ensure your environment supports **PyTorch** and **CUDA** (if you are using GPU acceleration). The requirements.txt contains detailed package versions and setup instructions.

 # Usage 🧰

## Data Preparation

Compared with typical preference datasets, each sample should also contain the `user_id` to learn each user's preference. The format of each sample should be `(user_id, prompt, (response_1), (response_2)),y`, where `user_id` is a unique identifier (string) for a specific user,  `prompt`, `response_1`, `response_2` are texts or other modalities,  $y\in \{-1, 1\}$ represents the user's preference. (Notice:  modify `dataset_factory.py` is required for your customized datasets).

🎯 For detailed implementation, please refer to `dataset_factory.py`.

## Training

















## Evaluation







## Integration







# Demo









# Contributing











# Citations









# Acknowledgment

- The structure of the README.md mimics the README.md of the repo of LLAVA







