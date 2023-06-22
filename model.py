from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from torch.optim import Adam
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.nn.modules.normalization import LayerNorm
from torch import nn
import torch
from scheduler import CosineWarmupScheduler
import wandb
import torchvision.transforms as T
import math
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1./dim**0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        self.qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out.weight)
        self.out.bias.data.fill_(0)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        z = self.attn_dropout(attn)

        x = (z @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        x = self.proj_dropout(x)

        if return_attention:
            return x, attn
        else:
            return x


class ImgPatches(nn.Module):
    def __init__(self, img_size=224, in_ch=3, embed_dim=768, patch_size=16):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.patched_img = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        B, C, H, W = img.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        patches = self.patched_img(img).flatten(2).transpose(1, 2)
        return patches


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              attn_dropout=drop_rate, proj_dropout=drop_rate, )
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU, dropout=drop_rate)

    def forward(self, x):
        xn1 = self.norm1(x)  # [b, num_patches+1, emb_dim]
        xa = self.attn(xn1)
        x = x + xa
        xn2 = self.norm2(x)
        xmlp = self.mlp(xn2)
        return x + xmlp


class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x

    @torch.no_grad()
    def get_attention_maps(self, x):

        for block in self.blocks:
            _, attn_map = block.attn(x, return_attention=True)
            # print(attn_map)

            x = block(x)
        return attn_map


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class ViT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_ch=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                 drop_rate=0.):
        super().__init__()

        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = ImgPatches(
            img_size=img_size, patch_size=patch_size, in_ch=in_ch, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(1, num_patches + 1, embed_dim)) # zero
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = nn.LayerNorm(embed_dim)

        self.transformer = Transformer(
            depth=depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate)
        # Classifier head
        self.to_latent = nn.Identity()

        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        # [b, c, h, w]

        B = x.shape[0]
        # [b, num_patch, embeding_dim]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        # [b, num_patch+1, embeding_dim]
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.head(x)

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        B = x.shape[0]
        # [b, num_patch, embeding_dim]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        # [b, num_patch+1, embeding_dim]
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attention_maps = self.transformer.get_attention_maps(x)

        return attention_maps


class VitModel(LightningModule):

    def __init__(self, lr=1e-5, img_size=224, patch_size=16, in_ch=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                 drop_rate=0., warmup=0, max_iters=1000, wandb_logger=None):
        '''method used to define our model parameters'''
        super().__init__()

        self.model = ViT(img_size, patch_size, in_ch, num_classes,
                         embed_dim, depth, num_heads, mlp_ratio,
                         drop_rate)
        # loss
        self.loss = CrossEntropyLoss()

        # optimizer parameters
        self.lr = lr
        self.warmup = warmup
        self.max_iters = max_iters
        self.wandb_logger = wandb_logger
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):
        '''method used for inference input -> output'''

        batch_size, channels, width, height = x.size()

        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)

        return {"loss": loss, "pred": preds, "gt": y}

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        attn = self.model.get_attention_maps(x)
        acc = accuracy(preds, y)
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        transform = T.ToPILImage()
        data = [[wandb.Image(x_i), wandb.Image(transform(
                attn_i[0])), str(y_i.cpu().numpy()), str(pred_i.cpu().numpy())] for x_i, attn_i, y_i, pred_i in zip(x, attn, y, preds)]

        columns = ['image', "attention", 'ground truth', 'prediction']

        self.wandb_logger.log_table(key="samples", columns=columns, data=data)
        return {"loss": loss, "pred": preds}

    def configure_optimizers(self):
        '''defines model optimizer'''
        ''''https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html'''
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup, max_iters=self.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        attention_map = self.vit.get_attention_maps(x)
        map = torch.max(attention_map, 1).values[0]
        return preds, loss, acc, map


# https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map
