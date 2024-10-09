import os, sys, argparse
from tqdm import tqdm
import torch
from src.datamodule import CustomInstructionDataModule
from src.vanilla_ideal_point_model.lightningmodule import LearnerWrapLightning as LearnerWrapLightningVanilla
from src.moe_ideal_point_model.lightningmodule import LearnerWrapLightning as LearnerWrapLightningMoe

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path',type=str)
    parser.add_argument('--state_dict_path',type=str)
    return parser.parse_args()

def init_learner_datamodule_vanilla(conf_learner, conf_ds, ckpt_path):
    learner = LearnerWrapLightningVanilla(**conf_learner)
    datamodule = CustomInstructionDataModule(**conf_ds)
    user_ids = datamodule.getAllUserIds()
    learner.preference_learner.user_learner.init_weight(user_ids)
    learner.preference_learner.update_trainable_params()
    learner = load_ckpt_learner(learner, ckpt_path)
    return learner, datamodule

def init_learner_datamodule_moe(conf_learner, conf_ds, ckpt_path):
    learner = LearnerWrapLightningMoe(**conf_learner)
    datamodule = CustomInstructionDataModule(**conf_ds)
    user_ids = datamodule.getAllUserIds()
    learner.preference_learner.user_learner.init_weight(user_ids)
    learner.preference_learner.update_trainable_params()
    learner = load_ckpt_learner(learner, ckpt_path)
    return learner, datamodule

def init_learner_datamodule_vanilla_temp_for_pickapicv1(conf_learner, conf_ds, ckpt_path):
    learner = LearnerWrapLightningVanilla(**conf_learner)
    datamodule = CustomInstructionDataModule(**conf_ds)
    user_ids = torch.load("/home/daiwei/research/diverse-alignment/refactored-diverse-alignment/necessary_cache/pickapicv1-dataset-tables/uid_rid_table.pt")
    learner.preference_learner.user_learner.init_weight(user_ids)
    learner.preference_learner.update_trainable_params()
    learner = load_ckpt_learner(learner, ckpt_path)
    return learner, datamodule

def init_learner_datamodule_moe_temp_for_pickapicv1(conf_learner, conf_ds, ckpt_path):
    learner = LearnerWrapLightningMoe(**conf_learner)
    datamodule = CustomInstructionDataModule(**conf_ds)
    user_ids = torch.load("/home/daiwei/research/diverse-alignment/refactored-diverse-alignment/necessary_cache/pickapicv1-dataset-tables/uid_rid_table.pt")
    learner.preference_learner.user_learner.init_weight(user_ids)
    learner.preference_learner.update_trainable_params()
    learner = load_ckpt_learner(learner, ckpt_path)
    return learner, datamodule

def init_learner_datamodule_moe_temp_for_pickapicv2(conf_learner, conf_ds, ckpt_path):
    learner = LearnerWrapLightningMoe(**conf_learner)
    datamodule = CustomInstructionDataModule(**conf_ds)
    user_ids = torch.load("/home/daiwei/research/diverse-alignment/refactored-diverse-alignment/necessary_cache/pickapicv2-dataset-tables/uid_rid_table.pt")
    user_ids = list(user_ids.keys())
    learner.preference_learner.user_learner.init_weight(user_ids)
    learner.preference_learner.update_trainable_params()
    learner = load_ckpt_learner(learner, ckpt_path)
    return learner, datamodule

def load_ckpt_learner(learner, ckpt_path):
    state_dict = torch.load(ckpt_path,map_location='cpu')['state_dict']
    learner.load_state_dict(state_dict)
    return learner

if __name__ == '__main__':
    args = get_args()
    state_dict = torch.load(args.ckpt_path,map_location='cpu')['state_dict']
    torch.save(state_dict,args.state_dict_path)
    print(f"success store state_dict in {args.state_dict_path}")



