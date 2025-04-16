import torch
import numpy as np
from functools import partial
from ConditionalDiffusionGeneration.src.guided_diffusion.unet import create_modified_model
from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import get_conditioning_method
from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_noise, get_operator
from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import create_sampler
from ConditionalDiffusionGeneration.src.util.img_utils import mask_generator
import os
from time import time

if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
  
device = torch.device(dev)  

Re_list  = (1000*np.arange(1,10) + 500.).astype(np.float32)
train_data = np.load("/add/path/here")
max_val, min_val = np.max(train_data), np.min(train_data)
time_length = 8
num_samples = 1

u_net_model = create_modified_model(image_size= 256,
                           num_channels= 128,
                           out_channels = 1,
                           num_res_blocks= 2,
                           num_heads=4,
                           num_head_channels=64,
                           attention_resolutions="32,16,8",
                           model_path=f'/add/path/here/.pt' # load the diffusion model
                        )

u_net_model.to(device);
u_net_model.eval();

sampler = create_sampler(sampler='ddpm',
                        steps=1000,
                        noise_schedule="cosine",
                        model_mean_type="epsilon",
                        model_var_type="fixed_large",
                        dynamic_threshold=False,
                        clip_denoised=True,
                        rescale_timesteps=False,
                        timestep_respacing="")

model_train_iter = None #"500k" #load the corresponding model iterations
for Re in Re_list:
    if Re in [Re_list[0], Re_list[4], Re_list[-1]]:
        print(f"Data already generated for {int(Re)}")
        continue
    
    if not os.path.exists(f"/add/path/here/test/{model_train_iter}_ar_mean"):
        os.mkdir(f"/add/path/here/test/{model_train_iter}_ar_mean")
        print(f"Directory {model_train_iter}_ar_mean created")
    
    torch.manual_seed(69)
    np.random.seed(69)
    print("-------------------------------------------------------------------------------------------------------")
    print(f"Generating {int(Re)} now...")

    unnorm_phy_samples = []
    true_measurement = torch.tensor(np.load(f"/add/path/here/interpolated_umean_{int(Re)}.npy"), device=device)[..., 0]
    
    for sample_count in range(num_samples):
        operator =  get_operator(device=device, name='mean_op',
                         model_dets="/add/path/here",
                         max_val=max_val,
                         min_val=min_val) 
        noiser = get_noise(sigma=0.0, name='gaussian')
        cond_method = get_conditioning_method(operator=operator, noiser=noiser, name='ps', scale=1.) 
        measurement_cond_fn = cond_method.conditioning

        sample_fn = partial(sampler.p_sample_loop, model=u_net_model, measurement_cond_fn=measurement_cond_fn,Re = torch.ones(1, device=device)*Re)
        x_start = torch.randn(1, 1, 256, 256, device=device).requires_grad_()
        
        start_time = time()
        
        one_traj = sample_fn(x_start=x_start, measurement=true_measurement, record=False, save_root=None)

        print(f"Generating the autoregressive sequence for {sample_count}th sample....")

        operator_seq =  get_operator(device=device, name='inpainting') 
        mask_gen =  mask_generator(mask_type='std_box', std_box_hstart=128, std_box_wstart=0, h=128, w=256)
        mask = mask_gen(torch.randn(1, 1, 256,256)).to(device)
        operator_mean =  get_operator(device=device, name='mean_op_Paper2',
                            model_dets="/add/path/here",
                            max_val=max_val,
                            min_val=min_val) 
        noiser = get_noise(sigma=0.0, name='gaussian')
        
        cond_method = get_conditioning_method(operator=None, operator_list=[operator_seq, operator_mean], noiser=noiser, name='seq-cons', scale_seq=1.0, scale_sen=1.0)
        measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
        sample_fn = partial(sampler.p_sample_loop, model=u_net_model, measurement_cond_fn=measurement_cond_fn,Re = torch.ones(1, device=device)*Re)

        ar_seq = [one_traj[..., :128, :]]
        
        for section in range(2*time_length - 1):
            measurements = []
            if section == 0:
                measurements.append(mask * one_traj)
                measurements.append(true_measurement)
            else :
                measurements.append(mask * torch.cat((sample[..., 128:256, :].detach(), torch.zeros(1, 1, 128, 256).to(device)), dim=2))
                measurements.append(true_measurement)
            x_start = torch.randn(1, 1, 256, 256, device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=measurements, record=False, save_root=None)
            ar_seq.append(sample[..., 128:256, :].detach())
        ar_seq_th = torch.cat(ar_seq, dim=2)

        end_time = time()
        
        print("-------------------------------------------------------------------------------------------------------")
        print(f"The time to generate the sequence is: {end_time - start_time} seconds")
        print("-------------------------------------------------------------------------------------------------------")

        phy_samples = (ar_seq_th[:, 0]).detach().cpu().numpy()
        unnorm_phy_samples.append((phy_samples + 1)*(max_val - min_val)/2 + min_val)

    unnorm_phy_samples = np.concatenate(unnorm_phy_samples, axis=0)
    print(f"Saving generated sequence for {int(Re)}....")
    np.save(f"/add/path/here/{model_train_iter}_ar_mean/Re_{int(Re)}.npy",unnorm_phy_samples)


