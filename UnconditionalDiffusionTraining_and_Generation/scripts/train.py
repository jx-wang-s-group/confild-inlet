#Imports
import sys
import torch
import numpy as np
from src.script_util import create_modified_model, create_gaussian_diffusion
from src.train_util import TrainLoop
from torch.utils.data import DataLoader
from src.dataset import Paper2DS_Re
from src.dist_util import setup_dist, dev
from src.logger import configure, log
from basicutility import ReadInput as ri

## Setup
torch.manual_seed(42)
np.random.seed(42)

inp = ri.basic_input(sys.argv[1])

setup_dist()
configure(dir=inp.log_path, format_strs=["stdout","log","tensorboard_new"])

## HyperParams
batch_size = inp.batch_size
test_batch_size = inp.test_batch_size

image_size= inp.image_size
num_channels= inp.num_channels
num_res_blocks= inp.num_res_blocks
num_head_channels= inp.num_head_channels
attention_resolutions=inp.attention_resolutions
channel_mult = inp.channel_mult

steps= inp.steps
noise_schedule= inp.noise_schedule

microbatch= inp.microbatch
final_lr = inp.final_lr
lr = inp.lr
ema_rate= inp.ema_rate
log_interval= inp.log_interval
save_interval= inp.save_interval
lr_anneal_steps= inp.lr_anneal_steps

train_data = np.load(inp.train_data_path)
train_data_batched = train_data

## Generate Reynolds Number list
Re_list  = (1000*np.arange(1,11)).astype(np.float32)
win_size = 256

log("creating data loader...")
max_val, min_val = np.max(train_data), np.min(train_data)
dl_train = DataLoader(Paper2DS_Re(train_data_batched, win_size, Re_list, max_val, min_val), batch_size=batch_size, shuffle=True)
dl_valid = DataLoader(Paper2DS_Re(train_data_batched, win_size, Re_list, max_val, min_val), batch_size=test_batch_size, shuffle=True) ## Validation set not used in this study. Can be used by loading validation data

def dl_iter(dl):
    while True:
        yield from dl 

## Unet Model
log("creating model and diffusion...")

unet_model = create_modified_model(image_size=image_size,
                          num_channels= num_channels,
                          num_res_blocks= num_res_blocks,
                          num_head_channels=num_head_channels,
                          attention_resolutions=attention_resolutions,
                        )

unet_model.to(dev())

## Gaussian Diffusion
diff_model = create_gaussian_diffusion(steps=steps,
                                       noise_schedule=noise_schedule
                                       )

## Training Loop
log("training...")

train_uncond_model = TrainLoop(
                                model=unet_model,
                                diffusion=diff_model,
                                train_data = dl_iter(dl_train),
                                valid_data=dl_iter(dl_valid),
                                batch_size= batch_size,
                                microbatch= microbatch,
                                lr = lr,
                                final_lr=final_lr,
                                ema_rate=ema_rate,
                                log_interval=log_interval,
                                save_interval=save_interval,
                                lr_anneal_steps=lr_anneal_steps,
                                resume_checkpoint=inp.resume_checkpoint,)

train_uncond_model.run_loop()