from datasets import load_dataset
import torch
from joblib import dump
import numpy as np
import pandas as pd

def generate_dataset_wo_embed(df, majority, minority):
    ds = []
    for _, row in df.iterrows():
        worker = row["worker"]
        prompt_text, left_text, right_text = (
            row["info"]["post"],
            row["summaries"][0]["text"],
            row["summaries"][1]["text"],
        )
        if None in (worker, prompt_text, left_text, right_text):
            continue
        left_len, right_len = len(left_text), len(right_text)
        if worker in minority:
            # prefer the shorter summary
            if left_len < right_len:
                y = -1
            else:
                y = 1
        elif worker in majority:
            # prefer the longer summary
            if left_len > right_len:
                y = -1
            else:
                y = 1
        else:
            raise ValueError("Worker not in top ten")
        sample = {}
        sample['uid'] = worker
        sample['prompt'] = 'Human: ' + prompt_text + ' Assistant:'
        sample['left'] = ' ' + left_text
        sample['right'] = ' ' + right_text
        sample['y'] = y
        ds.append(sample)
    return ds

def selection_predicate(df):
    left_policies = df.summaries.apply(lambda x: x[0]["policy"])
    right_policies = df.summaries.apply(lambda x: x[1]["policy"])
    # select rows where ppo is not in the policy and sup is in the policy
    ppo_not_in_left = left_policies.apply(lambda x: "ppo" not in x)
    ppo_not_in_right = right_policies.apply(lambda x: "ppo" not in x)
    sup_in_left = left_policies.apply(lambda x: "sup" in x)
    sup_in_right = right_policies.apply(lambda x: "sup" in x)
    predicate = ppo_not_in_left & ppo_not_in_right & sup_in_left & sup_in_right
    return predicate

def preprocess_ds(**kwargs):
    # follow the paper Li et al. 2024
    dataset = load_dataset("openai/summarize_from_feedback", "comparisons", trust_remote_code=True)
    df_train = dataset["train"].to_pandas()
    df_test = dataset["validation"].to_pandas()
    df_train_selected = df_train[selection_predicate(df_train)]
    df_test_selected = df_test[selection_predicate(df_test)]
    # select the 10 workers who answered the most questions
    top_ten_workers = df_train_selected.worker.value_counts().head(10).index
    minority = top_ten_workers[[3, 4, 5]]
    majority = top_ten_workers[[0, 1, 2, 6, 7, 8, 9]]
    df_train_selected_top_ten_seen = df_train_selected[df_train_selected.worker.isin(top_ten_workers)]
    df_test_selected_top_ten_seen = df_test_selected[df_test_selected.worker.isin(top_ten_workers)]
    # pal-rm new dataset generation process
    train_ds = generate_dataset_wo_embed(df_train_selected_top_ten_seen, majority, minority)
    test_ds = generate_dataset_wo_embed(df_test_selected_top_ten_seen, majority, minority)
    return train_ds, test_ds

def custom_train_test_split(df, user_col, n_train_samples, **kwargs):
    from sklearn.model_selection import train_test_split
    # Group the dataframe by user
    grouped = df.groupby(user_col)
    train_data = []
    test_data = []
    for _, user_data in grouped:
        if len(user_data) > n_train_samples:
            # Randomly select N samples for training
            train, test = train_test_split(user_data, train_size=n_train_samples, random_state=42)
            assert len(train) == n_train_samples
            train_data.append(train)
            test_data.append(test)
        else:
            raise ValueError('User has fewer than N samples')
    # Combine the results
    train_df = pd.concat(train_data, axis=0)
    test_df = pd.concat(test_data, axis=0)
    return train_df, test_df

def preprocess_unseen_ds(num_samples_per_unseen_train, **kwargs):
    dataset = load_dataset("openai/summarize_from_feedback", "comparisons", trust_remote_code=True)
    df_train = dataset["train"].to_pandas()
    df_test = dataset["validation"].to_pandas()
    df_train_selected = df_train[selection_predicate(df_train)]
    df_test_selected = df_test[selection_predicate(df_test)]
    # select the ~10 workers who answered the most questions
    top_ten_workers = df_train_selected.worker.value_counts().head(10).index
    df_test_unseen = df_test_selected[df_test_selected.worker.isin(top_ten_workers) == False]
    # get unseen workers who has at least 100 samples
    unseen_worker_100 = df_test_unseen.worker.value_counts().index[df_test_unseen.worker.value_counts() >= 100]
    # exclude any row if None in (worker, prompt_text, left_text, right_text):
    def extract_texts(row):
        try:
            prompt_text = row["info"]["post"]
            left_text = row["summaries"][0]["text"]
            right_text = row["summaries"][1]["text"]
            return pd.notna(prompt_text) and pd.notna(left_text) and pd.notna(right_text)
        except (KeyError, IndexError):
            return False
    df_test_unseen_selected = df_test_unseen[df_test_unseen.worker.isin(unseen_worker_100) & df_test_unseen.apply(extract_texts, axis=1)]
    majority_unseen = ['RMwrIV50cNusBthNvLs1wSNdqFpQAg', 'M4bdOszgybjO2qg2Dth5I1GOYAvE7V',
        'gMlGeJl1vsMERrbmC7W717zpVevUh8', 'OKFDIsAZl6Qa0m9x26f5Ao4S0uc7Ca',
        'uvzut5OK2bvei9zoCDdktcfLENYioY', 'iL7GfrbN2PeB3KInidqSxUdxYcTZmG',
        'BYcMzzjuFgaA59QDKoAgY07PyyG0qC', 'p7cM83bE3XsWlS9lTIvYCNfCVgOeTK',
        'rmgbTjW1stlproQnuHE2bUpK78Jxle', 'FCzllSEpfOHCBBEJqq4VeHRQR5JdoX',
        'P2p07Up4eJyvxrrVYgwtb60krbFbxI', 'qNGw27c8LHVPn31uJvjg3k0MZQcqQv',
        'thott7XepukYSbOL2QgSlyXd0rgHvr', '3AFaFd3w9NjDGnO51kupLyK1N44DQ2',
        'I2enBRrckFHw3KjJRSfKgsBwg4tmZy', 'D8z53gLFLFqhZowaegbtxmSGa0jqv0',
        'a0nhcJKlk2aO06xuBTMkeHdx8MDY5w', 'EeOYhWlpz7e45kXvg0RrfkjalZgkiz',
        'a7zXgbkuY6lk3vdt0q2Qf7SrZQgZ86', '9UQLCFxeYndGNfHEUP3yRt8XZhrrPr',
        '9YhRKOwkljIQ3NKf5CCV7BGjOcdPHu', 'mjwVX7RHTcfOfLTYGdBvms3vy8LTtP'] # 35 random users
    minority_unseen = list(set(unseen_worker_100) - set(majority_unseen))
    # randomly choose N samples from each unseen worker
    df_train_unseen, df_test_unseen = custom_train_test_split(df_test_unseen_selected, 'worker', num_samples_per_unseen_train)
    train_ds_unseen = generate_dataset_wo_embed(df_train_unseen, majority_unseen, minority_unseen)
    test_ds_unseen = generate_dataset_wo_embed(df_test_unseen, majority_unseen, minority_unseen)
    return train_ds_unseen, test_ds_unseen