# # Imports
import torch
import numpy as np
from functools import partial
import os
from ConditionalDiffusionGeneration.src.guided_diffusion.unet import create_modified_model
from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import get_conditioning_method
from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_noise, get_operator
from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import create_sampler
from ConditionalDiffusionGeneration.src.util.img_utils import mask_generator
from time import time

if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
  
device = torch.device(dev)  

train_data = np.load("")
max_val, min_val = np.max(train_data), np.min(train_data)
time_length = 2 #8
num_samples = 1

loop_over = {"add iterations and corresponding ema num"} #ex: "20k":"020000", "50k":"050000", "100k":"100000", "150k":"150000"
for model_train_iter, iter_val in loop_over.items():
  if not os.path.exists(f"/add/path/here/{model_train_iter}_ar"):
    os.mkdir(f"/add/path/here/{model_train_iter}_ar")
    print(f"Directory {model_train_iter}_ar created")
  
  u_net_model = create_modified_model(image_size= 256,
                          num_channels= 128,
                          out_channels = 1,
                          num_res_blocks= 2,
                          num_head_channels=64,
                          attention_resolutions="32,16,8",
                          model_path=f'/add/path/here/ema_0.9999_{iter_val}.pt'
                          )

  u_net_model.to(device);
  u_net_model.eval();

  ## Data Loader
  Re_list= (1000*np.arange(1,11)).astype(np.float32)
  Re_intp_list= (1000*np.arange(1,10) + 500.).astype(np.float32)

  for Re in [Re_list]:
      torch.manual_seed(100)
      np.random.seed(100)
      print("-------------------------------------------------------------------------------------------------------")
      print(f"Generating {int(Re)} now...")

      unnorm_phy_samples = []
      for _ in range(num_samples):
        print("Generating the uncond sequence first....")
        
        operator =  get_operator(device=device, name='noise') 
        noiser = get_noise(sigma=0.0, name='gaussian')
        cond_method = get_conditioning_method(operator=operator, noiser=noiser, name='vanilla') 
        measurement_cond_fn = partial(cond_method.conditioning)

        sampler = create_sampler(sampler='ddpm',
                                steps=1000,
                                noise_schedule="cosine",
                                model_mean_type="epsilon",
                                model_var_type="fixed_large",
                                dynamic_threshold=False,
                                clip_denoised=True,
                                rescale_timesteps=False,
                                timestep_respacing="")

        sample_fn = partial(sampler.p_sample_loop, model=u_net_model, measurement_cond_fn=measurement_cond_fn,Re = torch.ones(1, device=device)*Re)
        x_start = torch.randn(1, 1, 256, 256, device=device).requires_grad_()
        
        start_time = time()
        
        one_traj = sample_fn(x_start=x_start, measurement=None, record=False, save_root=None)

        print("Generating the autoregressive sequence....")
        
        operator =  get_operator(device=device, name='inpainting') 
        mask_gen =  mask_generator(mask_type='std_box', std_box_hstart=128, std_box_wstart=0, h=128, w=256)
        mask = mask_gen(torch.randn(1, 1, 256,256)).to(device)
        noiser = get_noise(sigma=0.0, name='gaussian')
        cond_method = get_conditioning_method(operator=operator, noiser=noiser, name='ps', scale=1.) 
        measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
        sample_fn = partial(sampler.p_sample_loop, model=u_net_model, measurement_cond_fn=measurement_cond_fn,Re = torch.ones(1, device=device)*Re)

        ar_seq = [one_traj[..., :128, :]]
        for section in range(2*time_length - 1):
            if time_length == 1:
                ar_seq.append(one_traj[..., 128:, :])
                break
            if section == 0:
                true_measurement = mask * one_traj
            else :
                true_measurement =  mask * torch.cat((sample[..., 128:256, :].detach(), torch.zeros(1, 1, 128, 256).to(device)), dim=2)
            x_start = torch.randn(1, 1, 256, 256, device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=true_measurement, record=False, save_root=None)
            ar_seq.append(sample[..., 128:256, :].detach())
        
        end_time = time()
        
        print("-------------------------------------------------------------------------------------------------------")
        print(f"The time to generate the sequence is: {end_time - start_time} seconds")
        print("-------------------------------------------------------------------------------------------------------")
       
        ar_seq_th = torch.cat(ar_seq, dim=2)

        phy_samples = (ar_seq_th[:, 0]).detach().cpu().numpy()
        unnorm_phy_samples.append((phy_samples + 1)*(max_val - min_val)/2 + min_val)
      
      unnorm_phy_samples = np.concatenate(unnorm_phy_samples, axis=0)
      print(f"Saving generated sequence for {int(Re)}....")
      np.save(f"/add/path/here/{model_train_iter}_ar/Re_{int(Re)}_long",unnorm_phy_samples)