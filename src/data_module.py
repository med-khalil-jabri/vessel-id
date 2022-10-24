from array import array
import os
import random
import sqlite3
import torch
import numpy as np
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
        self.category_labels = dataframe.category_label.values
        self.imo_labels = dataframe.imo_label.values
        self.transform = transform
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        with Image.open(os.path.join(self.data_dir, str(self.ids[index]) + '.jpg')) as img:
            img = self.transform(img.convert('RGB'))
        category_label = torch.tensor(self.category_labels[index], dtype=torch.int64)
        imo_label = torch.tensor(self.imo_labels[index], dtype=torch.int64)
        ids = torch.tensor(self.ids[index], dtype=torch.int64)
        return img, torch.stack([category_label, imo_label, ids])


class VesselDataModule(pl.LightningDataModule):
    def __init__(self, args):
        self.args = args
        augmentations = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomPerspective(p=0.25),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.1),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            T.RandomAutocontrast(p=0.1)
        ]
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(dataset_norms['imagenet21k']['mean'], dataset_norms[('imagenet21k')]['std']),
            T.Resize((args.img_size, args.img_size))
        ])
        if self.args.augment:
            self.train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(dataset_norms['imagenet21k']['mean'], dataset_norms[('imagenet21k')]['std']),
            *augmentations,
            T.Resize((args.img_size, args.img_size))
        ])
        else:
            self.train_transform = transforms
        self.test_transform = transforms
        self.prepare_data_per_node = False
        self.save_hyperparameters(args, ignore="load_from")

    def setup(self, stage):
        conn = sqlite3.connect(self.args.database)
        df = pd.read_sql_query("SELECT id, category, IMO FROM scraped_ships", conn)
        df = df[~(df.IMO == '')]
        # Group similar categories
        for label in ["Tankers", "General Cargo", "Sailing", "Cruise", "Containerships", "Carriers", "Bulkers", "Reefers", "Ferries", "Fishing"]:
            df.loc[df.category.str.contains(label), "category"] = label
        # Group non categorized categories
        for label in ["Photos Taken At Unofficial Rotterdam Meeting 2009", "Photos Taken At Rotterdam 2008 Meeting", "Auxiliaries"]:
            df.loc[df.category.str.contains(label), "category"] = "Non categorized"
        df.loc[df.category.isin(["Research And Survey Vessels", "Buoy/lighthouse Maintenance Vessels & Lightships"]), "category"] = "Lightships"
        df.loc[df.category.isin(["General Cargo", "Carriers"]), "category"] = "Carriers"
        # Remove useless entries
        df.drop(df[df['category'].isin(["Ship Interior", "Barges", "Formation And Group Shots", "Scrapyard Ships", "Wheelhouse", "_ Armaments", "_flight Decks", "_ Ships Crests", "Storm Pictures", "Ships' Lifeboats And Tenders"])].index, inplace=True)
        # Remove entries with extremely low occurences
        df.drop(df[df['IMO'].isin(df['IMO'].value_counts()[df['IMO'].value_counts() <= 5].index)].index, inplace=True)
        df.drop(df[df['category'].isin(df['category'].value_counts()[df['category'].value_counts() <= 5].index)].index, inplace=True)
        # Remove entries with inconsistent IMO
        dominant_label = df.groupby('IMO', as_index=False).category.agg(pd.Series.mode).rename(columns={"category": "dominant_category_label"})
        df = pd.merge(df, dominant_label, on=['IMO'])
        df.drop(df[df['category'] != df['dominant_category_label']].index, inplace=True)
        # Numeric labels
        df['category_label'] = pd.Categorical(df.category).codes
        df['imo_label'] = pd.Categorical(df.IMO).codes
        # small dataset for debugging
        # # TODO Remove lines below
        ####################################
        if self.args.debug:
            df = df[df['category_label'].isin([80 ,45 ,70 ,35 ,134])]
            df = df[~df['IMO'].isin([7397464, 7814101, 8128602, 1006245, 8404991, 5273339, 9378840, 7700180, 9159933, 9546497, 8814275, 8431645])]
        ####################################
        # Data splitting
        data_split = pd.read_csv(self.args.data_splits, header=None, names=['id', 'set'], skipinitialspace=True)
        train_df = df[df.id.isin(data_split[data_split.set == 'TRAIN'].id)]
        self.train_df, self.val_df = train_test_split(train_df, test_size=0.2, stratify=train_df.IMO)
        self.test_seen_df = df[df.id.isin(data_split[data_split.set == 'PROBE'].id)]
        self.test_unseen_df = df[df.id.isin(data_split[data_split.set == 'TEST'].id)]
        self.train_ds = VesselDataset(self.args.data_dir, self.train_df, self.train_transform)
        self.train_ds_eval = VesselDataset(self.args.data_dir, self.train_df, self.test_transform)
        self.val_ds = VesselDataset(self.args.data_dir, self.val_df, self.test_transform)
        self.test_seen_ds = VesselDataset(self.args.data_dir, self.test_seen_df, self.test_transform)
        self.test_unseen_ds = VesselDataset(self.args.data_dir, self.test_unseen_df, self.test_transform)
        # mapping IMOs to category weights
        imo2label = train_df.groupby('imo_label', as_index=False).category_label.agg(pd.Series.mode)
        weights = 1. / train_df.category_label.value_counts().rename("category_weight")
        weights[:] = weights / weights.sum()
        weights = pd.merge(imo2label, weights, right_index=True, left_on="category_label")
        imo2cat_weight = np.zeros(weights.imo_label.max() + 1)
        imo2cat_weight[weights.imo_label.to_numpy()] = weights.category_weight.to_numpy()
        self.imo2cat_weight = torch.from_numpy(imo2cat_weight)
        # retreive images for visualization
        self.viz_images = []
        for _ in range(self.args.n_val_viz):
            anchor = self.test_seen_df.sample()
            same_imo = self.train_df[(train_df.imo_label == anchor['imo_label'].values[0])].sample()
            same_cat = self.train_df[(train_df.category_label == anchor['category_label'].values[0]) & (self.train_df.imo_label != anchor['imo_label'].values[0])].sample()
            diff_cat = self.train_df[(train_df.category_label != anchor['category_label'].values[0])].sample()
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
        return [
            DataLoader(self.test_seen_ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers),
            DataLoader(self.test_unseen_ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers)   
        ]