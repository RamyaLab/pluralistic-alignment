# from flash.core.optimizers import LinearWarmupCosineAnnealingLR
import lightning as L
import torch.optim as optim
import torch
from ..loss_function import LossFunction, ProtoBoundLossFunction
from .learner import prefLearner_factory
from .utils import calc_hinge_loss, calc_ce_loss
from ..llms_dataset import preprocess_tokenized_ds_rm_pal_b
from math import floor

from typing import Literal, Optional, Tuple
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
remember to modify the configure_optimizers whenever you do any modification to 
the learner, otherwise your new parameters or net won't be updated!
'''

class LearnerWrapLightning(L.LightningModule):
    def __init__(
        self,
        prefLearner_pms: dict,
        optim_pms: dict,    # lr, wd, epochs_new_pair, epochs_new_user, warmup_ratio
        loss_pms: dict, # loss_type, upper_bound, lmd, loss_cumulative
    ):
        super().__init__()
        self.init_model(prefLearner_pms)
        self.init_loss(**loss_pms)
        self.init_optim(**optim_pms)
        self.automatic_optimization = False

    def init_model(self, prefLearner_pms):
        self.prefLearner_pms = prefLearner_pms
        PrefLearner = prefLearner_factory(self.prefLearner_pms.pref_learner_type)
        self.prefLearner = PrefLearner(**prefLearner_pms)
    
    def init_optim(self, optimizer_hyperparams, epochs_new_pair, epochs_new_user, warmup_ratio):
        self.optim_pms = optimizer_hyperparams
        self.warmup_ratio = warmup_ratio
        self.epochs_new_pair = epochs_new_pair
        self.epochs_new_user = epochs_new_user

    def init_loss(self, loss_weighting, upper_bound, lmd, loss_type):
        self.loss_weighting = loss_weighting
        loss_type += "_elementwise"
        self.loss_fn = LossFunction(loss_type)
        self.bound_loss_fn = ProtoBoundLossFunction(upper_bound, lmd)
        
    def set_mode(self, learner_mode: Literal["new_pair", "new_user"]):
        self.learner_mode = learner_mode

    def _wrap_forward(self, batch, batch_idx):
        # sample: ({'input': x, 'inds': [divergence_inds, end_inds]}, y)
        # x: (str(i), 
        # {
        # input_ids: prompt_input_ids,\
        # 'attention_mask': prompt_attention_mask,
        # },
        # {
        # 'left_input_ids': chosen_input_ids,\
        # 'right_input_ids': rejected_input_ids,\
        # 'left_attention_mask': chosen_attention_mask,\
        # 'right_attention_mask': rejected_attention_mask
        # })
        # batch = preprocess_tokenized_ds_rm_pal_b(batch, self.device)    # only for HH dataset, will be deprecated later
        sample, y = batch
        x, inds = sample['input'], sample['inds']
        # logger.critical(f"inds: {inds}")
        if self.prefLearner_pms.pref_learner_type in ['dist','dist_normalization','norm','angle_hinge']:
            y_hat = self.prefLearner(x)
            record = calc_hinge_loss(y_hat, y, self.loss_fn, inds, weightingMethod=self.loss_weighting)
        elif self.prefLearner_pms.pref_learner_type in ['angle','dist_logistic']:
            y = torch.tensor(y).long()
            y = ((y+1)/2).to(torch.long)
            y_hat = self.prefLearner(x)
            record = calc_ce_loss(y_hat, y, self.loss_fn, inds, weightingMethod=self.loss_weighting)
        loss, accu_mean_token, accu_last_token = record['loss'], record['accu_all_tokens'], record['accu_last_token']
        bound_items = self.prefLearner.user_learner.return_user_ideal_points()
        bound_loss =  self.bound_loss_fn(bound_items)
        return loss, bound_loss, accu_mean_token, accu_last_token

    def training_step(self, batch, batch_idx):
        assert hasattr(self, 'learner_mode'), "Please call set_mode() before training."
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss, bound_loss, accu_mean_token, accu_last_token = self._wrap_forward(batch, batch_idx)
        self.manual_backward(loss)
        optimizer.step()
        if self.learner_mode == "new_pair":
            self.log("TrainLoss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("TrainAccuLastToken", accu_last_token, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("TrainAccuMeanToken", accu_mean_token, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            if self.prefLearner_pms.pref_learner_type == 'angle':
                self.log("logitScale", self.prefLearner.logit_scale.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif self.learner_mode == "new_user":
            self.log("new_TrainLoss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("new_TrainAccuLastToken", accu_last_token, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("new_TrainAccuMeanToken", accu_mean_token, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, bound_loss, accu_mean_token, accu_last_token = self._wrap_forward(batch, batch_idx)
        if "new_pair" in self.learner_mode:
            self.log("ValLoss", loss, on_epoch=True, prog_bar=False, logger=True)
            self.log("ValAccuLastToken", accu_last_token, on_epoch=True, prog_bar=True, logger=True)
            self.log("ValAccuMeanToken", accu_mean_token, on_epoch=True, prog_bar=True, logger=True)
        elif "new_user" in self.learner_mode:
            self.log("new_ValLoss", loss, on_epoch=True, prog_bar=False, logger=True)
            self.log("new_ValAccuLastToken", accu_last_token, on_epoch=True, prog_bar=True, logger=True)
            self.log("new_ValAccuMeanToken", accu_mean_token, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, bound_loss, accu_mean_token, accu_last_token = self._wrap_forward(batch, batch_idx)
        if "new_pair" in self.learner_mode:
            self.log("TestLoss", loss, on_epoch=True, prog_bar=False, logger=True)
            self.log("TestAccuLastToken", accu_last_token, on_epoch=True, prog_bar=True, logger=True)
            self.log("TestAccuMeanToken", accu_mean_token, on_epoch=True, prog_bar=True, logger=True)
        elif "new_user" in self.learner_mode:
            self.log("new_TestLoss", loss, on_epoch=True, prog_bar=False, logger=True)
            self.log("new_TestAccuLastToken", accu_last_token, on_epoch=True, prog_bar=True, logger=True)
            self.log("new_TestAccuMeanToken", accu_mean_token, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        if self.learner_mode == 'new_pair':
            logger.info('new_pair learner mode')
            parameter_groups = []
            for trainable_key in self.optim_pms.keys():
                hyperparams = self.optim_pms[trainable_key]
                if trainable_key in self.prefLearner.trainable_params.keys():
                    logger.critical(f"✅ trainable_key: {trainable_key}")
                    params = self.prefLearner.trainable_params[trainable_key]
                    parameter_groups.append({"params": params, **hyperparams})
                else:
                    logger.critical(f"🚫 trainable_key: {trainable_key} not in the prefLearner.trainable_params")
            optimizer = optim.Adam(parameter_groups)
        elif self.learner_mode == 'new_user':
            logger.info('new_user learner mode')
            params = self.prefLearner.trainable_params["W"]
            hyperparams = self.optim_pms["W"]
            optimizer = optim.AdamW(params, **hyperparams)
        return {'optimizer': optimizer}