from statistics import mode
from pytorch_lightning import Trainer, seed_everything
from src.data_module import VesselDataModule
from src.vit_module import ViTModule
from train import get_args


def test(args):
    seed_everything(args.seed, workers=True)
    data_module = VesselDataModule(args)
    assert args.load_from.endswith('.ckpt'), "Load checkpoint using --load_from"
    model = ViTModule.load_from_checkpoint(args.load_from, args=args, data_module=data_module, strict=False)
    trainer = Trainer.from_argparse_args(args)
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    args = get_args()
    test(args)