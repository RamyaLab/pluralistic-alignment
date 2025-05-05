import lightning as L
import torch.optim as optim
import torch
from ..loss_function import LossFunction, ProtoBoundLossFunction
from .learner import prefLearner_factory
from math import floor
from copy import deepcopy

from typing import Optional, Literal
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
remember to modify the configure_optimizers whenever you do any modification to 
the learner, otherwise your new parameters or net may not be updated!
'''

class LearnerWrapLightning(L.LightningModule):
    def __init__(
            self,
            preference_learner_params: dict,    
            optimizer_hyperparams: dict,    # lr / wd 
            learner_mode: Literal["new_pair", "new_pair_finetune_weight", "new_user"],
            loss_type: Literal["hinge", "logistic", "CE", "hinge_w", "logistic_w", "CE_w"],
            upper_bound: float, # beta
            lmd: float,
            max_epochs_new_pair: float,
            max_epochs_new_user: float,
            warmup_ratio: float,    # NotImplementedYet: lr_scheduler
        ):
        super().__init__()
        
        self.preference_learner_params = preference_learner_params
        PrefLearner = prefLearner_factory(self.preference_learner_params.pref_learner_type)
        self.optimizer_hyperparams = optimizer_hyperparams
        self.preference_learner = PrefLearner(**preference_learner_params)
        self.learner_mode = learner_mode
        self.loss_fn = LossFunction(loss_type)
        self.bound_loss_fn = ProtoBoundLossFunction(upper_bound, lmd)
        self.max_epochs_new_pair = max_epochs_new_pair
        self.max_epochs_new_user = max_epochs_new_user
        self.warmup_ratio = warmup_ratio
        self.automatic_optimization = False
    
    def _wrap_forward(self, batch, batch_idx):
        x,y = batch
        batch_u_ids = x[0]
        if self.preference_learner_params.pref_learner_type in ['dist','dist_normalization','norm','angle_hinge']:
            y_hat = self.preference_learner(x)
            loss = self.loss_fn(y_hat, y, batch_u_ids=batch_u_ids)
            accu = torch.mean(((y_hat * y) > 0).to(torch.float))
        elif self.preference_learner_params.pref_learner_type in ['angle','dist_logistic']:
            y = (y+1)/2
            y = y.to(torch.long)
            # fix-patch for an issue in old simulated dataset {-1,3}
            if y.max() > 1:
                y = y / 2
                y = y.to(torch.long)
            y_hat = self.preference_learner(x)
            loss = self.loss_fn(y_hat, y, batch_u_ids=batch_u_ids)
            accu = (y_hat.argmax(dim=1) == y).float().mean().item()
        else:
            raise ValueError(f'Unknown preference learner type: {self.preference_learner_params.pref_learner_type}')
        bound_items = self.preference_learner.user_learner.return_user_ideal_points()
        bound_loss =  self.bound_loss_fn(bound_items)
        return loss, bound_loss, accu

    def training_step(self, batch, batch_idx):
        # NOTICE!!!
        # don't know the mechanism of train/eval mode transform in the lightning module
        # we explicitly set the user_learner.train() here
        # it is necessary for the user_learner to change the softmax logic when using partiton model
        self.preference_learner.user_learner.train()
        optimizer = self.optimizers()
        # lr_scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        loss, bound_loss, accu = self._wrap_forward(batch, batch_idx)
        total_loss = loss + bound_loss
        self.manual_backward(total_loss)
        optimizer.step()
        # if self.trainer.is_last_batch:
        #     lr_scheduler.step()
        if "new_pair" in self.learner_mode:
            self.log("Train_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("Train_Loss_Bound", bound_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("Train_Loss_Total", total_loss, on_step=True, on_epoch=True, logger=True)
            self.log("Train_Accuracy", accu, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            if self.preference_learner_params.pref_learner_type == 'angle':
                self.log("logitScale", self.preference_learner.logit_scale.item(), on_epoch=True, logger=True)
            if self.preference_learner_params.is_temperature_learnable:
                self.log("temperature", self.preference_learner.softmax_w.temperature.item(), on_epoch=True, logger=True)
        elif "new_user" in self.learner_mode:
            self.log("New_User_Train_Loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
            self.log("New_User_Train_Accuracy", accu, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        self.preference_learner.user_learner.eval()
        loss, _, accu = self._wrap_forward(batch, batch_idx)
        if "new_pair" in self.learner_mode:
            self.log("Validation_Loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log("Validation_Accuracy", accu, on_epoch=True, prog_bar=True, logger=True)
        elif "new_user" in self.learner_mode:
            self.log("New_User_Validation_Loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log("New_User_Validation_Accuracy", accu, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.preference_learner.user_learner.eval()
        loss, _, accu = self._wrap_forward(batch, batch_idx)
        if "new_pair" in self.learner_mode:
            self.log("Test_Loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log("Test_Accuracy", accu, on_epoch=True, prog_bar=True, logger=True)
        elif "new_user" in self.learner_mode:
            self.log("New_User_Test_Loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log("New_User_Test_Accuracy", accu, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.learner_mode == "new_pair":
            print('new_pair learner mode, please check your learnable modules carefully!')
            parameter_groups = []
            for trainable_key in self.optimizer_hyperparams.keys(): # all learnable module configurations in yaml file
                hyperparams = self.optimizer_hyperparams[trainable_key]
                if trainable_key in self.preference_learner.trainable_params.keys():    # double check if the learnable module name in the learner
                    logger.critical(f"✅ trainable_key: {trainable_key}")
                    params = self.preference_learner.trainable_params[trainable_key]
                    parameter_groups.append(
                        {"params": params, **hyperparams}
                    )
                else:
                    logger.critical(f"🚫 trainable_key: {trainable_key} not in the preference_learner.trainable_params")
            optimizer = optim.AdamW(parameter_groups)
        
        elif self.learner_mode == "new_pair_finetune_weight":    # finetune the weights
            raise DeprecationWarning("This mode is deprecated, please only use either new_pair or new_user")
            print('new_pair learner finetune weight mode')
            params = self.preference_learner.trainable_params["W"]
            hyperparams = self.optimizer_hyperparams["W"]
            optimizer = optim.AdamW(params, **hyperparams)

        elif self.learner_mode == "new_user":
            print('new_user learner mode')
            params = self.preference_learner.trainable_params["W"]
            hyperparams = self.optimizer_hyperparams["W"]
            optimizer = optim.AdamW(params, **hyperparams)

        else:
            raise ValueError(f'Unknown learner mode type: {self.learner_mode}')
        
        return {
            "optimizer": optimizer,
        }
