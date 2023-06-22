from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks import LogPredictionsCallback
from model import VitModel
from torchvision.datasets import FGVCAircraft
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
import wandb
from dataset.dataset import Aircraft
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)
wandb_logger = WandbLogger(project='FGVCAircraft',  # group runs in "MNIST" project
                           log_model='all')  # log all new checkpoints during training
log_predictions_callback = LogPredictionsCallback(wandb_logger)

checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(
        (128, 128), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.Resize((128, 128)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],)
])
train_dataset = Aircraft('./dataset', train=True, download=False,
                         transform=train_transform, test=True, class_type="manufacturer")
test_dataset = Aircraft('./dataset', train=False, download=False,
                        transform=val_transform, test=True, class_type="manufacturer")
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)

trainer = Trainer(
    logger=wandb_logger,                    # W&B integration
    callbacks=[checkpoint_callback],        # our model checkpoint callback
    accelerator='gpu', devices=1, max_epochs=1000, deterministic=True)                           # number of epochs

model = VitModel(lr=0.0001, img_size=128, patch_size=16, in_ch=3, num_classes=23,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                 drop_rate=0.01, warmup=100, max_iters=2000, wandb_logger=wandb_logger)

training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, )
validation_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

trainer.fit(model, training_loader, validation_loader)
