import os
import random
import sqlite3
import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as T
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
        self.train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(dataset_norms['imagenet21k']['mean'], dataset_norms[('imagenet21k')]['std']),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomPerspective(p=0.25),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.1),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            T.RandomAutocontrast(p=0.1),
            T.Resize((args.img_size, args.img_size))
        ])
        self.test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(dataset_norms['imagenet21k']['mean'], dataset_norms[('imagenet21k')]['std']),
            T.Resize((args.img_size, args.img_size))
        ])
        self.prepare_data_per_node = False
        self.save_hyperparameters(args, ignore="load_from")

    def setup(self, stage):
        conn = sqlite3.connect(self.args.database)
        df = pd.read_sql_query("SELECT id, category, IMO FROM scraped_ships", conn)
        df['label'] = pd.Categorical(df.category).codes
        data_split = pd.read_csv(self.args.data_splits, header=None, names=['id', 'set'], skipinitialspace=True)
        df = df[~(df.IMO == '')]
        # TODO Remove lines below
        ####################################
        # df = df[df['label'].isin([80 ,45 ,70 ,35 ,134])]
        # df = df[~df['IMO'].isin([7397464, 7814101, 8128602, 1006245, 8404991, 5273339, 9378840, 7700180, 9159933, 9546497, 8814275, 8431645])]
        ####################################
        train_df = df[df.id.isin(data_split[data_split.set == 'TRAIN'].id)]
        self.train_df, self.val_df = train_test_split(train_df, test_size=0.2, stratify=train_df.IMO)
        self.test_seen_df = df[df.id.isin(data_split[data_split.set == 'PROBE'].id)]
        self.test_unseen_df = df[df.id.isin(data_split[data_split.set == 'TEST'].id)]
        self.train_ds = VesselDataset(self.args.data_dir, self.train_df, self.train_transform)
        self.train_ds_eval = VesselDataset(self.args.data_dir, self.train_df, self.test_transform)
        self.val_ds = VesselDataset(self.args.data_dir, self.val_df, self.test_transform)
        self.test_seen_ds = VesselDataset(self.args.data_dir, self.test_seen_df, self.test_transform)
        self.test_unseen_ds = VesselDataset(self.args.data_dir, self.test_unseen_df, self.test_transform)
        # retreive images for visualization
        self.viz_images = []
        for _ in range(self.args.n_viz_images):
            anchor = self.test_seen_df.sample()
            same_imo = self.train_df[(train_df.IMO == anchor['IMO'].values[0])].sample()
            same_cat = self.train_df[(train_df.label == anchor['label'].values[0]) & (self.train_df.IMO != anchor['IMO'].values[0])].sample()
            diff_cat = self.train_df[(train_df.label != anchor['label'].values[0])].sample()
            keys = ['anchor', 'same_imo', 'same_cat', 'diff_cat']
            vals = [{
                'id': im['id'].values[0],
                'IMO': im['IMO'].values[0],
                'category': im['category'],
                'image': Image.open(os.path.join(self.args.data_dir, str(im['id'].values[0]) + '.jpg'))
            } for im in [anchor, same_imo, same_cat, diff_cat]]
            self.viz_images.append(dict(zip(keys, vals)))

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