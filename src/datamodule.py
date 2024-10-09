import lightning as L
from torch.utils.data import DataLoader
from .dataset import PreferenceDataset, CustomInstructionDataset, SummaryDataset, CustomInstructionDataset_llm
from .llms_dataset import DataCollatorReward
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreferenceDataModule(L.LightningDataModule):

    train_ds_list: list
    val_ds_list: list
    test_ds_list: list
    batch_size: int
    num_workers: int
    persistent_workers: bool

    def __init__(
        self,
        train_ds_list: list,
        val_ds_list: list,
        test_ds_list: list,
        batch_size: int,
        num_workers: int = 4,
        persistent_workers: bool = True,
    ):
        super().__init__()

        self.train_ds_list = train_ds_list
        self.val_ds_list = val_ds_list
        self.test_ds_list = test_ds_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage: str):
        self.train_ds = PreferenceDataset(self.train_ds_list)
        self.val_ds = PreferenceDataset(self.val_ds_list)
        self.test_ds = PreferenceDataset(self.test_ds_list)

    def train_dataloader(self):
        logger.warning("do not use this method when you are doing analysis, use train_dataloader_eval instead")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )
    
    def train_dataloader_eval(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

class CustomInstructionDataModule(L.LightningDataModule):
    def __init__(self, ds_path, train_instr_file_path, val_instr_file_path, test_instr_file_path, batch_size: int = 256, num_workers: int = 16, *args, **kwargs):
        super().__init__()
        self.train_instr_file_path = train_instr_file_path
        self.val_instr_file_path = val_instr_file_path
        self.test_instr_file_path = test_instr_file_path
        self.ds_path = ds_path
        self.batch_size = batch_size
        self.num_workers = num_workers
    def prepare_data(self):
        pass
    def setup(self, stage: str):
        self.train_ds = CustomInstructionDataset(self.train_instr_file_path, self.ds_path)
        self.val_ds = CustomInstructionDataset(self.val_instr_file_path, self.ds_path)
        self.test_ds = CustomInstructionDataset(self.test_instr_file_path, self.ds_path)
    def train_dataloader(self):
        
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=True,
        )
    def train_dataloader_eval(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def predict_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def getUserIds(self, instr_file_path:str, ds_path:str):
        ds = CustomInstructionDataset(instr_file_path, ds_path)
        dl = DataLoader(ds,4096,shuffle=False,num_workers=16)
        user_ids_dict = set()
        tqdmr = tqdm(dl,desc="getUserIds")
        for x,y in tqdmr:
            user_ids = x[0]
            [user_ids_dict.add(id) for id in user_ids]
        del dl
        del ds
        return list(user_ids_dict)
    def getTrainUserIds(self):
        return self.getUserIds(self.train_instr_file_path, self.ds_path)
    def getAllUserIds(self):
        all_user_ids = []
        all_user_ids.extend(self.getUserIds(self.train_instr_file_path, self.ds_path))
        all_user_ids.extend(self.getUserIds(self.val_instr_file_path, self.ds_path))
        all_user_ids.extend(self.getUserIds(self.test_instr_file_path, self.ds_path))
        return all_user_ids


class CustomInstructionDataModule_llm(L.LightningDataModule):
    def __init__(self, ds_path, train_instr_file_path, val_instr_file_path, test_instr_file_path, batch_size: int = 256, num_workers: int = 16, *args, **kwargs):
        super().__init__()
        self.train_instr_file_path = train_instr_file_path
        self.val_instr_file_path = val_instr_file_path
        self.test_instr_file_path = test_instr_file_path
        self.ds_path = ds_path
        self.batch_size = batch_size
        self.num_workers = num_workers
    def prepare_data(self):
        pass
    def setup(self, stage: str):
        self.train_ds = CustomInstructionDataset_llm(self.train_instr_file_path, self.ds_path)
        self.val_ds = CustomInstructionDataset_llm(self.val_instr_file_path, self.ds_path)
        self.test_ds = CustomInstructionDataset_llm(self.test_instr_file_path, self.ds_path)
    def train_dataloader(self):
        
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=True,
        )
    def train_dataloader_eval(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def predict_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def getUserIds(self, instr_file_path:str, ds_path:str):
        ds = CustomInstructionDataset_llm(instr_file_path, ds_path)
        dl = DataLoader(ds,4096,shuffle=False,num_workers=16)
        user_ids_dict = set()
        tqdmr = tqdm(dl,desc="getUserIds")
        for x,y in tqdmr:
            user_ids = x[0]
            [user_ids_dict.add(id) for id in user_ids]
        del dl
        del ds
        return list(user_ids_dict)
    def getTrainUserIds(self):
        return self.getUserIds(self.train_instr_file_path, self.ds_path)
    def getAllUserIds(self):
        all_user_ids = []
        all_user_ids.extend(self.getUserIds(self.train_instr_file_path, self.ds_path))
        all_user_ids.extend(self.getUserIds(self.val_instr_file_path, self.ds_path))
        all_user_ids.extend(self.getUserIds(self.test_instr_file_path, self.ds_path))
        return all_user_ids



class SummaryDataModule(L.LightningDataModule):
    def __init__(self, ds_path, prompt_embeds_path, summary_embeds_path, train_instr_file_path, val_instr_file_path, test_instr_file_path, batch_size: int = 256, num_workers: int = 16, *args, **kwargs):
        super().__init__()
        self.train_instr_file_path = train_instr_file_path
        self.val_instr_file_path = val_instr_file_path
        self.test_instr_file_path = test_instr_file_path
        self.ds_path = ds_path
        self.prompt_embeds_path = prompt_embeds_path
        self.summary_embeds_path = summary_embeds_path
        self.batch_size = batch_size
        self.num_workers = num_workers
    def prepare_data(self):
        pass
    def setup(self, stage: str):
        self.train_ds = SummaryDataset(self.train_instr_file_path, self.ds_path, self.prompt_embeds_path, self.summary_embeds_path)
        self.val_ds = SummaryDataset(self.val_instr_file_path, self.ds_path, self.prompt_embeds_path, self.summary_embeds_path)
        self.test_ds = SummaryDataset(self.test_instr_file_path, self.ds_path, self.prompt_embeds_path, self.summary_embeds_path)
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=True,
        )
    def train_dataloader_eval(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def predict_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False,
            shuffle=False,
        )
    def getUserIds(self, instr_file_path:str, ds_path:str):
        ds = SummaryDataset(instr_file_path, ds_path, self.prompt_embeds_path, self.summary_embeds_path)
        dl = DataLoader(ds,4096,shuffle=False,num_workers=16)
        user_ids_dict = set()
        tqdmr = tqdm(dl,desc="getUserIds")
        for x,y in tqdmr:
            user_ids = x[0]
            [user_ids_dict.add(id) for id in user_ids]
        del dl
        del ds
        return list(user_ids_dict)
    def getTrainUserIds(self):
        return self.getUserIds(self.train_instr_file_path, self.ds_path)
    def getAllUserIds(self):
        all_user_ids = []
        all_user_ids.extend(self.getUserIds(self.train_instr_file_path, self.ds_path))
        all_user_ids.extend(self.getUserIds(self.val_instr_file_path, self.ds_path))
        all_user_ids.extend(self.getUserIds(self.test_instr_file_path, self.ds_path))
        return all_user_ids
    
class TokenizedRewardDataModule(L.LightningDataModule):
    
    def __init__(self, train_ds, val_ds, test_ds, batch_size, num_workers=4, persistent_workers=False, data_collator=None):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.data_collator = data_collator
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.data_collator, \
            num_workers=self.num_workers, persistent_workers=self.persistent_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.data_collator, \
            num_workers=self.num_workers, persistent_workers=self.persistent_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.data_collator, \
            num_workers=self.num_workers, persistent_workers=self.persistent_workers, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.data_collator, \
            num_workers=self.num_workers, persistent_workers=self.persistent_workers, shuffle=False)


if __name__ == '__main__':
    
    ...
