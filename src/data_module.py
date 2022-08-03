import os
import sqlite3
import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.dataset_norms import dataset_norms


class VesselDataset(Dataset):
    def __init__(self, data_dir, dataframe:pd.DataFrame, transform):
        self.data_dir = data_dir
        self.dataframe = dataframe
        self.ids = self.dataframe.id.values
        self.labels = dataframe.label.values
        self.imos = dataframe.IMO.values
        self.transform = transform
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        with Image.open(os.path.join(self.data_dir, str(self.ids[index]) + '.jpg')) as img:
            img = self.transform(img.convert('RGB'))
        label = torch.tensor(self.labels[index])
        imo = torch.tensor(self.imos[index])
        return img, torch.stack([label, imo])


class VesselDataModule(pl.LightningDataModule):
    def __init__(self, args):
        self.args = args
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_norms['imagenet21k']['mean'], dataset_norms[('imagenet21k')]['std']),
            transforms.RandomHorizontalFlip(),

        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_norms['imagenet21k']['mean'], dataset_norms[('imagenet21k')]['std']),
        ])
        self.prepare_data_per_node = False
        self.save_hyperparameters(args)

    def setup(self, stage):
        conn = sqlite3.connect(self.args.database)
        df = pd.read_sql_query("SELECT id, category, IMO FROM scraped_ships", conn)
        df['label'] = pd.Categorical(df.category).codes
        data_split = pd.read_csv(self.args.data_splits, header=None, names=['id', 'set'], skipinitialspace=True)
        df = df[~(df.IMO == '')]
        train_df = df[df.id.isin(data_split[data_split.set == 'TRAIN'].id)]
        self.train_df, self.val_df = train_test_split(train_df, test_size=0.2, stratify=train_df.IMO)
        self.test_seen_df = df[df.id.isin(data_split[data_split.set == 'PROBE'].id)]
        self.test_unseen_df = df[df.id.isin(data_split[data_split.set == 'TEST'].id)]
        self.train_ds = VesselDataset(self.args.data_dir, self.train_df, self.train_transform)
        self.val_ds = VesselDataset(self.args.data_dir, self.val_df, self.test_transform)
        self.test_seen_ds = VesselDataset(self.args.data_dir, self.test_seen_df, self.test_transform)
        self.test_unseen_ds = VesselDataset(self.args.data_dir, self.test_unseen_df, self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    def test_dataloader(self):
        loaders = {
            'seen': DataLoader(self.test_seen_ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers),
            'unseen': DataLoader(self.test_unseen_ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers)  
        }
        return loaders