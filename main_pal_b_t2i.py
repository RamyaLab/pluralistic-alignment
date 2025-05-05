from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm
import os
import argparse
from uuid import uuid4
import torch

from src.utils import create_filename
from src.datamodule import CustomInstructionDataModule
from src.pal_rm_b_t2i.lightningmodule import LearnerWrapLightning
from src.tensor_initializer import TensorInitializer

def main_only_seen(conf_learner,conf_seen_ds,conf_seen_wandb,user_ids:list,ckpt_filename:str,folder_path:str,devices:list=[0]):
    
    k = conf_learner.preference_learner_params.k
    
    ########################################################################################
    ############# learn the model & protos & seen users' weights together ##################
    ########################################################################################
    # init model checkpoint recorder
    checkpoint_callback = ModelCheckpoint(
        monitor='Validation_Loss',
        dirpath='./ckpts/',
        # filename='seen-'+ckpt_filename+'-{epoch:02d}-{Validation_Loss:.2f}',
        filename='seen-'+ckpt_filename+'-{epoch:02d}',    # remove validation loss, sometimes doesn't match up. To fixup.
        save_last=True,
        enable_version_counter=True,
        verbose=True,
    )
    # init the wandb logger for seen user dataset
    wandb_logger = WandbLogger(**conf_seen_wandb)
    merged_config = OmegaConf.merge(conf_learner, conf_seen_ds, conf_seen_wandb)
    config_dict = OmegaConf.to_container(merged_config, resolve=True)
    # init learner and the seen user dataset
    learner = LearnerWrapLightning(**conf_learner)
    datamodule = CustomInstructionDataModule(**conf_seen_ds)
    # some initialization for the learner
    # user_ids = datamodule.getAllUserIds()
    # user_ids = torch.load("./user_ids_registration/v1_user_ids.pt")
    learner.preference_learner.user_learner.init_weight(user_ids)
    learner.preference_learner.update_trainable_params()
    # train on the seen user dataset
    learner.learner_mode = 'new_pair'
    trainer = L.Trainer(
        max_epochs=conf_learner.max_epochs_new_pair,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        # strategy=DDPStrategy(find_unused_parameters=True),
        devices=devices,
    )
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(config_dict)
    trainer.fit(learner, datamodule=datamodule)
    trainer.test(learner, datamodule=datamodule, ckpt_path="best")
    wandb_run = wandb_logger.experiment.finish()

def get_args():
    parser = argparse.ArgumentParser(description='pal_b')
    parser.add_argument('--k', type=int, default=2, help='number of prototypes')
    parser.add_argument('--conf_learner', type=str, help='path to the learner config')
    parser.add_argument('--conf_ds', type=str, help='path to the seen dataset config')
    parser.add_argument('--cache', type=str, help='path to the cache folder')
    parser.add_argument('--project', type=str, help='project name')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--device', type=int, default=0, help='gpu devices')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
    # load default configs
    conf_learner = OmegaConf.load(args.conf_learner)
    conf_ds = OmegaConf.load(args.conf_ds)
    conf_wandb = OmegaConf.load("./config_wandb/wandb_template.yaml")
    user_ids = torch.load(os.path.join(args.cache,"user_ids.pt"))
    # modify configs
    conf_learner.preference_learner_params.k = args.k
    conf_learner.max_epochs_new_pair = 50
    conf_ds.batch_size = 16384
    conf_wandb.project = args.project
    conf_wandb.name = args.name
    # init save path
    random_uuid4 = str(uuid4())[:8]  # This takes only the first 8 characters.
    filename = create_filename(args.name, str(args.k), str(conf_ds.batch_size))
    filename += f"-{random_uuid4}"
    folder_path = os.path.join("./figs", filename)
    print("ckpt store path:", filename)
    
    main_only_seen(
        conf_learner=conf_learner,
        conf_seen_ds=conf_ds,
        conf_seen_wandb=conf_wandb,
        user_ids=list(user_ids),
        ckpt_filename=filename,
        folder_path=folder_path,
        devices=[args.device],
    )
