import os
import argparse
import pytorch_metric_learning as pml
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from src.dataset_norms import dataset_norms
from src.data_module import VesselDataModule
from src.vit_module import ViTModule


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=99, help='Random seed')
    # data
    parser.add_argument('--data-dir', type=str, default='./data/images', help='')
    parser.add_argument('--database', type=str, default='./data/scraped_ships.db', help='')
    parser.add_argument('--data-splits', type=str, default='./data/data_splits.txt', help='')
    # ViT
    parser.add_argument('--output-size', type=int, default=512, help='')
    parser.add_argument('--init-head', dest='init-head', action='store_true')
    parser.add_argument('--no-init-head', dest='init-head', action='store_false')
    parser.set_defaults(init_head=True)
    parser.add_argument('--classifier', type=str, default='token', help='')
    parser.add_argument('--hidden-size', type=int, default=768, help='')
    parser.add_argument('--img-size', type=int, default=256, help='')
    parser.add_argument('--patch-size', type=int, default=16, help='')
    parser.add_argument('--load-from', type=str, help='weights/imagenet21k/ViT-B16.npz')
    parser.add_argument('--dropout-rate', type=int, default=0, help='')
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.add_argument('--no-vis', dest='vis', action='store_false')
    parser.set_defaults(vis=True)
    parser.add_argument('--num-layers', type=int, default=12, help='')
    parser.add_argument('--mlp-dim', type=int, default=3072, help='')
    parser.add_argument('--num-heads', type=int, default=12, help='number of attention heads')
    parser.add_argument('--global-feature-embedding', type=str, default='mean', choices=['mean', 'cls'], help='Whether to use the class token or average over all tokens to get the embeddings')
    parser.add_argument('--attention-dropout-rate', type=int, default=0, help='')
    # Training
    parser.add_argument('--batch-size', type=int, default=128, help='')
    parser.add_argument('--lr', type=float, default=1e-5, help='')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='')
    args = parser.parse_args()
    return args


def train(args):
    seed_everything(args.seed, workers=True)
    data_module = VesselDataModule(args)
    model = ViTModule(args)
    trainer = Trainer()
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    args = get_args()
    train(args)