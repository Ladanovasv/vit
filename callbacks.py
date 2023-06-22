from PIL import Image
from pytorch_lightning.callbacks import Callback
import wandb
import torchvision.transforms as T


class LogPredictionsCallback(Callback):
    def __init__(self, wandb_logger):
        super().__init__()
        self.wandb_logger = wandb_logger

    def validation_epoch_end(self, validation_step_outputs):

        acc = 0
        losses = 0
        data = []
        transform = T.ToPILImage()

        for out in validation_step_outputs:
            losses += out["loss"]
            acc += accuracy(out["pred"], out["gt"])
            data.append([wandb.Image(transform(out["x"][0])), wandb.Image(transform(
                out["attn"][0])), str(out["gt"].cpu().numpy()[0]), str(out["pred"].cpu().numpy()[0])])
        self.log('val_loss', losses/len(validation_step_outputs))
        self.log('val_accuracy', acc/len(validation_step_outputs))

        columns = ['image', "attention", 'ground truth', 'prediction']

        self.wandb_logger.log_table(key="samples", columns=columns, data=data)
