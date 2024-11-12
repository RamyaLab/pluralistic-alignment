import os
from uuid import uuid4
from omegaconf import OmegaConf
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from src.pal_rm_a.lightningmodule import LearnerWrapLightning
from src.dataset_factory import dataset_factory
from src.datamodule import TokenizedRewardDataModule

import argparse
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ckpt_learner(learner, ckpt_path):
    state_dict = torch.load(ckpt_path,map_location='cpu')['state_dict']
    keys = list(state_dict.keys())
    for key in keys:
        if 'prefLearner.user_learner.W'in key:
            del state_dict[key]
    learner.load_state_dict(state_dict)
    del state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_config', type=str, default='./config/ds_config/summary_unseen.yaml')
    parser.add_argument('--prefLearner_config', type=str, default='./config/prefLearner_config/a-dim512-k2-opt350m-mlp2.yaml')
    parser.add_argument('--optim_config', type=str, default='./config/optim_config/vanilla.yaml')
    parser.add_argument('--loss_config', type=str, default='./config/loss_config/a-cumulative.yaml')
    parser.add_argument('--state_dict_path', type=str, default='/home/reid/projects/refactored-diverse-alignment-unseen/ckpts/summary-a-best.ckpt')
    parser.add_argument('--run_name', type=str, default='summary-unseen-a-cumulative-k2-mlp2')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    # Load configurations
    logger.critical(f" 💠 Loading configurations...")
    ds_config = OmegaConf.load(args.ds_config)
    prefLearner_config = OmegaConf.load(args.prefLearner_config)
    optim_config = OmegaConf.load(args.optim_config)
    loss_config = OmegaConf.load(args.loss_config)
    merged_config = OmegaConf.merge(ds_config, prefLearner_config, optim_config, loss_config)
    config_dict = OmegaConf.to_container(merged_config, resolve=True)
    wandb_logger = WandbLogger(project='pal-rm-a-unseen-distilbert', name=args.run_name)
    filename = args.run_name + '-' + 'seen' + '-' + '{epoch:02d}' + '-' + str(uuid4())[:4]
    checkpoint_callback = ModelCheckpoint(monitor='new_ValLoss',dirpath='./ckpts/',filename=filename,verbose=True,enable_version_counter=True)
    
    # Load DataModule
    logger.critical(f" 💠 Loading datamodule...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-cased',fast_tokenizer=True)
    train_ds, val_ds, test_ds = dataset_factory(**ds_config, model_type='a', tokenizer=tokenizer)
    uids = torch.load(ds_config.user_ids_path)
    dm = TokenizedRewardDataModule(train_ds, val_ds, test_ds, batch_size=prefLearner_config.bs, num_workers=4, persistent_workers=False)
    
    # Load model
    logger.critical(f" 💠 Loading model...")
    learner = LearnerWrapLightning(prefLearner_config, optim_config, loss_config)
    load_ckpt_learner(learner, args.state_dict_path)
    learner.prefLearner.user_learner.init_weight(uids)
    learner.prefLearner.update_trainable_params(['itemLearnerProjector', 'userLearnerProjector', 'llm'])    # fix llm and projectors, only update user weights
    # learner.prefLearner.update_trainable_params()
    learner.set_mode("new_user")
    trainer = L.Trainer(max_epochs=optim_config.epochs_new_pair, devices=[args.device], logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=1)
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(config_dict)
    
    # Train
    logger.critical(" 💠 start training...")
    trainer.validate(learner, datamodule=dm)
    trainer.fit(learner, datamodule=dm)
    trainer.test(learner, datamodule=dm, ckpt_path='best')
    wandb_run = wandb_logger.experiment.finish()
    logger.critical(" 💠 Finish training...")
