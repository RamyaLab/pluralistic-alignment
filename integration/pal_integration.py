import os, sys
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from types import MethodType

import torch
import lightning as L
from src.pal_rm_a.lightningmodule import LearnerWrapLightning as LearnerWrapLightningA
from src.pal_rm_b.lightningmodule import LearnerWrapLightning as LearnerWrapLightningB

import logging
logger = logging.getLogger(__name__)

def load_ckpt_learner(learner, ckpt_path):
    state_dict = torch.load(ckpt_path,map_location='cpu')['state_dict']
    learner.load_state_dict(state_dict)
    return learner

def wrap_mix_forward_a(pal: torch.nn.Module, mix_weight: torch.tensor):
    # override the forward function of the pal to be a standard reward model
    # after the modification, the pal will be able to output:
    # model a: the reward diff given a prompt
    # model b: the reward logits given a prompt
    def mix_forward_userlearner(self):
        assert sum(mix_weight) == 1
        w = self.softmax(mix_weight.unsqueeze(0))
        w = mix_weight.unsqueeze(0)
        if self.learner_type == 'f(Pw)':
            res = self.projector(w @ self.P)
            return res
        elif self.learner_type == 'f(P)w':
            res = w @ self.projector(self.P)
            return res
        elif self.learner_type == 'Pw':
            res = w @ self.P
            return res
        else:
            raise NotImplementedError
    def mix_forward_itemlearner(self, items):
        embeds = self.llm(input_ids=items['input_ids'], attention_mask=items['attention_mask']).last_hidden_state
        # embeds shape: (bs, seq_len, hidden_size)
        shape = embeds.shape
        embeds = embeds.view(-1, shape[-1]) # (bs*seq_len, hidden_size)
        projected_embeds = self.projector(embeds)
        return projected_embeds.view(shape[0], shape[1], -1)
    def mix_map(self, x):
        # {
        # 'left_input_ids': chosen_input_ids,\
        # 'right_input_ids': rejected_input_ids,\
        # 'left_attention_mask': chosen_attention_mask,\
        # 'right_attention_mask': rejected_attention_mask
        # }
        items = x
        x_prime = self.item_learner(items)
        u_prime = self.user_learner()
        bs = items['left_input_ids'].size(0)
        u_prime = u_prime.repeat(bs, 1)
        return x_prime, u_prime
    def mix_forward_preflearner(self, x):
        if self.pref_learner_type == 'dist':
            x_prime, u_prime = self.map_to_pref_embedding_space(x)
            x_to_u_dist = torch.linalg.norm(x_prime - u_prime.unsqueeze(1), dim=-1)
            return -x_to_u_dist
        else:
            raise NotImplementedError
    def forwad(self, batch):
        y_hat = self.prefLearner(batch)
        return y_hat
    pal.prefLearner.user_learner.forward = MethodType(mix_forward_userlearner, pal.prefLearner.user_learner)
    pal.prefLearner.item_learner.forward = MethodType(mix_forward_itemlearner, pal.prefLearner.item_learner)
    pal.prefLearner.map_to_pref_embedding_space = MethodType(mix_map, pal.prefLearner)
    pal.prefLearner.forward = MethodType(mix_forward_preflearner, pal.prefLearner)
    pal.forward = MethodType(forwad, pal)
    pal.eval()
    return pal

def wrap_mix_forward_b(pal: torch.nn.Module, mix_weight: torch.tensor):
    # override the forward function of the pal to be a standard reward model
    # after the modification, the pal will be able to output:
    # model a: the reward diff given a prompt
    # model b: the reward logits given a prompt
    def mix_forward_userlearner(self, prompt_tokens, rm_cached=None):
        logger.info(f"{mix_weight=}")
        logger.info(f"{mix_weight.shape=}")
        if rm_cached is None:
            prompt_logits = self.infer_gk(prompt_tokens)
        else:
            prompt_logits, rm_cached = self.infer_gk(prompt_tokens, rm_cached)
        bs = prompt_tokens['input_ids'].size(0)
        assert sum(mix_weight) == 1
        # w = self.softmax(mix_weight.repeat(bs, 1))
        w = mix_weight.repeat(bs, 1)
        logger.info(f"{w=}")
        logger.info(f"{w.shape=}")
        w = w.unsqueeze(-1).unsqueeze(-1)
        y_hat = (w * prompt_logits).sum(dim=1)
        self.tmp_store_user_ideal_points = y_hat
        return y_hat, rm_cached
    def mix_forward_itemlearner(self, items, rm_cached=None):

        input_ids = items['input_ids']
        attention_mask = items['attention_mask']

        # print(input_ids[:, -1:].size())

        if rm_cached is None:
            llm_res = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            llm_res = self.llm(
                input_ids=input_ids[:, -1:], # attention_mask=attention_mask,
                past_key_values=rm_cached["item_learner"],
                use_cache=False
            )
            rm_cached["item_learner"] = llm_res.past_key_values

        embeds = llm_res.last_hidden_state
        # embeds shape: (bs, seq_len, hidden_size)
        shape = embeds.shape
        embeds = embeds.view(-1, shape[-1]) # (bs*seq_len, hidden_size)
        projected_embeds = self.projector(embeds)

        if rm_cached is None:
            return projected_embeds.view(shape[0], shape[1], -1)
        else:
            return projected_embeds.view(shape[0], shape[1], -1), rm_cached
    def mix_map_preflearner(self, x, rm_cached=None):
        # ({
        # 'input_ids': prompt_input_ids,\
        # 'attention_mask': prompt_attention_mask,
        # },\
        # {
        # 'input_ids': eval_input_ids,\
        # 'attention_mask': eval_attention_mask,\
        # })
        prompt, items = x
        if rm_cached is None:
            items_prime = self.item_learner(items)
            prompt_prime = self.user_learner(prompt)
            return items_prime, prompt_prime
        else:
            items_prime, rm_cached = self.item_learner(items, rm_cached)
            prompt_prime, rm_cached = self.user_learner(prompt, rm_cached)
            return items_prime, prompt_prime, rm_cached
    def mix_forward_preflearner(self, x, rm_cached=None):
        items, prompt = x
        if rm_cached is None:
            items_prime, prompt_prime = self.map_to_pref_embedding_space((prompt, items))
        else:
            items_prime, prompt_prime, rm_cached = self.map_to_pref_embedding_space((prompt, items), rm_cached)
        logger.info(f"{items_prime[0]=}")
        logger.info(f"{prompt_prime[0]=}")
        logger.info(f"{items_prime.shape=}")
        logger.info(f"{prompt_prime.shape=}")
        if self.pref_learner_type == 'angle':
            # prompt_last_prime = prompt_prime[:, -1, :]
            # prompt_last_prime_repeat = prompt_last_prime.unsqueeze(1).repeat(1, items_prime.size(1), 1)
            # items_prime = items_prime / torch.norm(items_prime, dim=-1, keepdim=True)
            # prompt_last_prime_repeat = prompt_last_prime_repeat / torch.norm(prompt_last_prime_repeat, dim=-1, keepdim=True)
            prompt_last_prime = prompt_prime[:, -1, :]
            prompt_last_prime = prompt_last_prime.unsqueeze(1)
            prompt_last_prime = prompt_last_prime / torch.norm(prompt_last_prime, dim=-1, keepdim=True)
            items_last_prime = items_prime[:, -1, :]
            items_last_prime = items_last_prime.unsqueeze(1)
            items_last_prime = items_last_prime / torch.norm(items_last_prime, dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            clamped_logit_scale = torch.clamp(logit_scale, max=100)
            logger.info(f"{prompt_last_prime.shape=}")
            logger.info(f"{items_last_prime.shape=}")
            sim_score = (prompt_last_prime * items_last_prime).sum(dim=-1) * clamped_logit_scale   # (bs, max_token_length)
            if rm_cached is None:
                return sim_score
            else:
                return sim_score, rm_cached
        else:
            raise NotImplementedError
    def forwad(self, batch, rm_cached=None):
        if rm_cached is None:
            y_hat = self.prefLearner(batch)
            return y_hat
        else:
            y_hat, rm_cached = self.prefLearner(batch, rm_cached)
            return y_hat, rm_cached
    pal.prefLearner.user_learner.forward = MethodType(mix_forward_userlearner, pal.prefLearner.user_learner)
    pal.prefLearner.item_learner.forward = MethodType(mix_forward_itemlearner, pal.prefLearner.item_learner)
    pal.prefLearner.map_to_pref_embedding_space = MethodType(mix_map_preflearner, pal.prefLearner)
    pal.prefLearner.forward = MethodType(mix_forward_preflearner, pal.prefLearner)
    pal.forward = MethodType(forwad, pal)
    return pal

def wrap_rm_forward_b(pal):
    # this is the forward function for only reward prediction
    def rm_forward(self, batch):
        sample = batch
        x, inds = sample['input'], sample['inds']
        # logger.critical(f"inds: {inds}")
        if self.prefLearner_pms.pref_learner_type in ['dist','dist_normalization','norm','angle_hinge']:
            y_hat = self.prefLearner(x)
        elif self.prefLearner_pms.pref_learner_type in ['angle','dist_logistic']:
            y_hat = self.prefLearner(x)
        return y_hat
    pal.forward = MethodType(rm_forward, pal)
    return pal

def load_pal_rm_a(
    mix_weight: torch.tensor,
    pref_Learner_config_path: str,
    optim_config_path: str,
    loss_config_path: str,
    ds_config_path: str,
    state_dict_path: str,
    **kwargs,
):
    logger.critical(' 💠 load configurations...')
    prefLearner_config = OmegaConf.load(pref_Learner_config_path)
    optim_config = OmegaConf.load(optim_config_path)
    loss_config = OmegaConf.load(loss_config_path)
    ds_config = OmegaConf.load(ds_config_path)
    uids = torch.load(ds_config.user_ids_path)
    logger.critical(' 💠 initiaize pal_rm model...')
    pal = LearnerWrapLightningA(prefLearner_config, optim_config, loss_config)
    pal.prefLearner.user_learner.init_weight(uids)
    pal_rm = wrap_mix_forward_a(pal, mix_weight)
    load_ckpt_learner(pal_rm, state_dict_path)
    logger.critical(' 💠 complete reformat: pal -> pal_rm!')
    return pal_rm

def load_pal_rm_b(
    mix_weight: torch.tensor,
    pref_Learner_config_path: str,
    optim_config_path: str,
    loss_config_path: str,
    ds_config_path: str,
    state_dict_path: str,
    **kwargs,
):
    logger.critical(' 💠 load configurations...')
    prefLearner_config = OmegaConf.load(pref_Learner_config_path)
    optim_config = OmegaConf.load(optim_config_path)
    loss_config = OmegaConf.load(loss_config_path)
    ds_config = OmegaConf.load(ds_config_path)
    uids = torch.load(ds_config.user_ids_path)
    logger.critical(' 💠 initiaize pal_rm model...')
    pal = LearnerWrapLightningB(prefLearner_config, optim_config, loss_config)
    pal.prefLearner.user_learner.init_weight(uids)
    pal_rm = wrap_mix_forward_b(pal, mix_weight)
    load_ckpt_learner(pal_rm, state_dict_path)
    logger.critical(' 💠 complete reformat: pal -> pal_rm!')
    return pal_rm

def load_pal_rm_b_vanilla_forward(
    mix_weight: torch.tensor,
    pref_Learner_config_path: str,
    optim_config_path: str,
    loss_config_path: str,
    ds_config_path: str,
    state_dict_path: str,
    **kwargs,
):
    logger.critical(' 💠 load configurations...')
    prefLearner_config = OmegaConf.load(pref_Learner_config_path)
    optim_config = OmegaConf.load(optim_config_path)
    loss_config = OmegaConf.load(loss_config_path)
    ds_config = OmegaConf.load(ds_config_path)
    uids = torch.load(ds_config.user_ids_path)
    logger.critical(' 💠 initiaize pal_rm model...')
    pal = LearnerWrapLightningB(prefLearner_config, optim_config, loss_config)
    pal.prefLearner.user_learner.init_weight(uids)
    pal_rm = wrap_rm_forward_b(pal, mix_weight)
    load_ckpt_learner(pal_rm, state_dict_path)
    logger.critical(' 💠 complete reformat: pal -> pal_rm!')
    return pal_rm

# if __name__ == '__main__':
    
#     logger.critical(' 💠 load configurations...')
#     parser = argparse.ArgumentParser(description='PAL integration')
#     parser.add_argument(
#         '--pal_model_path', 
#         type=str, 
#         default='./ckpts', 
#         help='Model name'
#     )
#     parser.add_argument(
#         '--output_path', 
#         type=str, 
#         default='./mix', 
#         help='Model name'
#     )
#     args = parser.parse_args()
#     model_path = args.pal_model_path
#     mix_model_path = args.output_path
    
#     logger.critical(' 💠 load pal model...')
#     model = L.load_from_checkpoint(model_path)
#     model_type = ...    # TODO: find model type from model_path
#     mixture_weights = ...   # TODO: determine the mixture weights
#     mix_model_path = mix_model_path + '_mix_' + str(mixture_weights) + '.pt'
#     logger.critical(f'saving path: {mix_model_path}')
    
#     logger.critical(' 💠 reframe the pal model...')
#     if model_type == 'a':
#         ...
        
#     elif model_type == 'b':
#         ...
#     else:
#         raise ValueError(f' ❌ Unknown model type: {model_type}')
    
#     logger.critical(' 💠 complete!')