"""Disco Spinach V0.01.ipynb


# Generates images from text prompts with CLIP guided diffusion.

[Original notebook] By Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings). 
It uses a 512x512 unconditional ImageNet diffusion model fine-tuned from OpenAI's 512x512 class-conditional ImageNet diffusion model (https://github.com/openai/guided-diffusion) together with CLIP (https://github.com/openai/CLIP) to connect text prompts with images.
Original colab notebook is located at:
    https://colab.research.google.com/drive/1V66mUeJbXrTuQITvJunvnWVn96FEbSI3

# Changelog:
4/2/23: initial release
note:
- uses the 512x512_diffusion_uncond_finetune_008100.pt (trained by Katherine Crowson)

Features borrowed from Disco Diffusion:
- mounting a google drive for running the model via google-colab (without redownloading each new session)
- allow choice of steps
- init image (lpips loss approach)
- 'use secondary model' (requires SecondaryModel.py - saves alot of VRAM)
- use of multiple Clip Models
- Clip models used are the Disco Diffusion defaults: ViT-B/16, ViT-B/32, and RN50
- range_loss
- sat_scale (saturation scale)
- clamp_grad (clamp gradient - this is particularly important for image quality when using the secondary model)
- cutn_batches

Additional Features:
- Scheduled parameters (heavily inspired by Ethan Smith's mathrock-diffusion)
- Gradio UI: allows for access via API (for Krita integration)
- 'Skip steps' are assigned proportion of total steps (ie 0.5) rather than integer valued
"""

# @title Licensed under the MIT License

# Copyright (c) 2021 Katherine Crowson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# --

# MIT License

# Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# --

# Licensed under the MIT License

# Copyright (c) 2021 Maxwell Ingham

# Copyright (c) 2022 Adam Letts

# Copyright (c) 2022 Alex Spirin

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import time

import subprocess, os

def wget(url, outputdir):
    res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

models_path = './models'

def git(op, address):
    res = subprocess.run(['git', op, address], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pip(op, *args):
    res = subprocess.run(['pip', op, *args], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)


#!git clone https://github.com/openai/CLIP
git('clone', 'https://github.com/openai/CLIP')

#!git clone https://github.com/crowsonkb/guided-diffusion
git('clone', 'https://github.com/crowsonkb/guided-diffusion')

#!pip install -e ./CLIP
#pip('install', '-e', './Clip')

#!pip install -e ./guided-diffusion
#pip('install', '-e', './guided-diffusion')

#!pip install lpips
pip('install', 'lpips')

# download mirror: https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt

if not os.path.exists(models_path+"/512x512_diffusion_uncond_finetune_008100.pt"):
  wget('https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt', models_path)

# Imports

import gc
import io
import math
import sys

import lpips
from PIL import Image
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from SecondaryModel import SecondaryDiffusionImageNet2, alpha_sigma_to_t

sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')

import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

import gradio as gr
import numpy as np

import ast
import json
# Define necessary functions

out_dir = './output'

os.makedirs(out_dir, exist_ok=True)


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

def symm_loss(im, horizontal, loss_metric, offset, twist, ignore_color):
    """
    im (array) : input image
    horizontal (bool) : if true, horizontal. otherwise, vertical
    loss_metric (func) : image comparison metric (such as lpips metric or l1 (pixelwise) loss)
    offset (int) : horizontal or vertical offset
    twist (bool) : if true, flip on other axis before comparing halves ('S shape' symmetry)
    ignore_color (bool) : if true, only compare values (results in image with more color)
    """
    length = im.shape[3] if horizontal else im.shape[2] # length = width or height
    half_length = int(length / 2)   
    if abs(offset) >= half_length:
        length_str = "width" if horizontal else "height"
        axis_str = "horizontal" if horizontal else "vertical"
        raise ValueError(axis_str + "-offset must be less than half the image " + length_str)   
    slice_length = half_length - abs(offset)
    start_offset = max(2*offset, 0)   
    if horizontal:
        h1, h2 = (im[:, :, :, start_offset:start_offset+slice_length], 
                    im[:, :, :, start_offset+slice_length:start_offset+slice_length*2])
    else:
        h1, h2 = (im[:, :, start_offset:start_offset+slice_length, :], 
                    im[:, :, start_offset+slice_length:start_offset+slice_length*2, :])
    if ignore_color:
        gray = T.Grayscale(3)
        h1 = gray(h1)
        h2 = gray(h2)  
    if twist:
        h2 = TF.vflip(h2) if horizontal else TF.hflip(h2)
        h2 = TF.hflip(h2) if horizontal else TF.vflip(h2)
    return loss_metric(h1, h2)

def get_past_outputs():
    outputs = os.listdir('./output/')
    output_nums = [x for x in outputs if '.txt' not in x]
    output_nums = [x.split('_')[-1] for x in output_nums]
    output_nums = [x.split('.')[0] for x in output_nums]
    output_nums.sort()
    return int(output_nums[-1])

def rb_swap(pil_img):
    # Get the image data as a numpy array
    image_data = np.array(pil_img)
    # Swap the red and blue channels
    image_data[:, :, 0], image_data[:, :, 2] = image_data[:, :, 2], image_data[:, :, 0].copy()
    # Create a new image from the modified data
    pil_img_out = Image.fromarray(image_data)
    return pil_img_out

def clean_param_schedule(param_str):
    param_str = param_str.replace(' ', '')
    param_str = param_str.replace(',', '|')
    param_str = param_str.replace('\\', '|')
    return param_str

def schedule_of(param_str, current_t, net_steps):
    steps_proportion = (net_steps - current_t) / net_steps
    param_list = param_str.split('|')
    param_val = param_list[0]
    for i in range(len(param_list)):
        if i / len(param_list) < steps_proportion:
            param_val = param_list[i]
    #print(steps_proportion)
    #print(param_val)
    #print(current_t)
    return int(param_val)

# Model settings

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',  # Modify this value to decrease the number of
                                   # timesteps.
    'image_size': 512,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_checkpoint': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})

#steps = 50
use_secondary_model = True

# Load models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

lpips_model = lpips.LPIPS(net='vgg').to(device)

if use_secondary_model:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(torch.load(f'./models/secondary_model_imagenet_2.pth', map_location='cpu'))
    secondary_model.eval().requires_grad_(False).to(device)


ViTB32 = True #@param{type:"boolean"}
ViTB16 = True #@param{type:"boolean"}
#ViTL14 = False #@param{type:"boolean"}
RN50 = True #@param{type:"boolean"}

clip_models = []
if ViTB32 is True: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)) 
if ViTB16 is True: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device) ) 
#if ViTL14 is True: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device) ) 
if RN50 is True: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])



"""## Settings for this run:"""

#prompts = ['Shrine to the spirit of wonder, by James Gurney and Maxfield Parish']
image_prompts = []
batch_size = 1 #for now, this must remain 1 to save the img properly
#clip_guidance_scale = 9000   # Controls how much the image should look like the prompt.
#tv_scale = 150              # Controls the smoothness of the final output.
tv_scale = 0 
#range_scale = 65            # Controls how far out of range RGB values are allowed to be.
#range_scale = 150            # Controls how far out of range RGB values are allowed to be.

sat_scale = 0

clamp_grad = True #@param{type: 'boolean'}
#clamp_max = 0.05 #@param{type: 'number'}


cutn = 16
cutn_batches = 2
cut_pow = 0.5
n_batches = 1
#init_image = None   # This can be an URL or Colab local path and must be in quotes.
#skip_amount = 0.4
#skip_timesteps = 200  # This needs to be between approx. 200 and 500 when using an init image.
                    # Higher values make the output look more like the init.
#init_scale = 1800   # This enhances the effect of the init image, a good value is 1000.
#seed = 54

"""### Actually do the run..."""

display_rate = 25
clip_size = 224

# TODO:
# provide options for 'save out' and 'save in'
# allow for viewing and rendering partials
# allow for multiple renders
# allow choice of CLIP models
# consider:
# add in django cuts


def do_run(prompt, width_height, steps, cgs, range_scale, clamp_max, cut_pow, seed, img_in, init_strength, init_scale, init_width_height): #symmetry_switch, symmetry_loss_scale):
    
    out_settings = {
        "prompt": prompt,
        "width_height": width_height,
        "steps": steps,
        "cgs": cgs,
        "range_scale": range_scale,
        "clamp_max": clamp_max,
        "cut_pow": cut_pow,
        "seed": seed,
        "init_strength": init_strength, 
        "init_scale": init_scale,
        "init_width_height": init_width_height
    }

    #symmetry_loss_axis = 'h'
    #symmetry_switch = 90
    #symmetry_loss_scale = 2400
    #h_symmetry_offset = 0
    #v_symmetry_offset = 0
    #symmetry_twist = False
    #symmetry_ignore_color = False
    #print(img_in[:100])
    #note we are passing an integer in from the other interface
    cut_pow = cut_pow / 100

    #cgs = '6500|4000|2000' #for testing
    init_scale = clean_param_schedule(init_scale)
    cgs = clean_param_schedule(cgs)
    range_scale = clean_param_schedule(range_scale)
    clamp_max = clean_param_schedule(clamp_max)
    #symmetry_switch = 100.0 * (1.0 - (symmetry_switch / steps))
    print("prompt: " + prompt)
    print("width, height: " + width_height)
    print("steps: " + str(steps))
    print("init_strength: " + str(init_strength))
    print("init_scale: " + str(init_scale))
    print("cgs: " + str(cgs))
    print("range scale: " + str(range_scale))
    print("clamp max: " + str(clamp_max))
    print("cut power: " + str(cut_pow))
    print("seed: " + str(seed))

    net_steps = steps * (1 - init_strength)
    print("net steps: " + str(net_steps))

    w_h = width_height.split(', ')
    #width, height = width_height.split(', ')
    width = int(w_h[0])
    height = int(w_h[1])

    byte_arr = ast.literal_eval(img_in)

    init_w_h = init_width_height.split(', ')
    init_w_h = [int(x) for x in init_w_h]
    #print(byte_arr[:100])
    #print(type(byte_arr))
    #img_in = Image.frombytes("RGBA", (width, height), byte_arr)
    img_in = Image.frombytes("RGBA", tuple(init_w_h), byte_arr)
    img_in = rb_swap(img_in) # swap red and blue channels
    print(img_in.size)
    img_in = img_in.resize((width, height))
    print(img_in.size)
    #img_in.save('tester.png', 'PNG')

    #skip_timesteps = math.floor(steps * skip_amount)
    skip_timesteps = math.floor(steps * init_strength)
    #Update Model Settings
    timestep_respacing = f'ddim{steps}'
    diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
    model_config.update({
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
    })

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(models_path + '/512x512_diffusion_uncond_finetune_008100.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    prompts = [prompt]
    clip_guidance_scale = cgs
    result = []

    if seed is not None:
        torch.manual_seed(seed)

    make_cutouts = MakeCutouts(clip_size, cutn, cut_pow)
    #side_x = model_config['image_size']
    #side_y = model_config['image_size']
    side_x = width
    side_y = height

    #target_embeds, weights = [], []
    model_stats = []
    for clip_model in clip_models:

        model_stat = {"clip_model":clip_model, "target_embeds":[], "make_cutouts":None, "weights":[]}
        for prompt in prompts:
            txt, weight = parse_prompt(prompt)
            txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
            model_stat["target_embeds"].append(txt)
            model_stat["weights"].append(weight)

        for prompt in image_prompts:
            path, weight = parse_prompt(prompt)
            img = Image.open(fetch(path)).convert('RGB')
            img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = clip_model.encode_image(normalize(batch)).float()
            model_stat["target_embeds"].append(embed)
            model_stat["weights"].extend([weight / cutn] * cutn)

        model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
        model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)

    #init = None
    #if init_image is not None:
        #init = Image.open(fetch(init_image)).convert('RGB')
    init = img_in.convert('RGB')
    init = init.resize((side_x, side_y), Image.LANCZOS)
    init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    def cond_fn(x, t, out, y=None):
        n = x.shape[0]
        x = x.detach().requires_grad_()
        if use_secondary_model is True:
            alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
            sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
            cosine_t = alpha_sigma_to_t(alpha, sigma)
            out = secondary_model(x, cosine_t[None].repeat([n])).pred
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
        else:
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
        for model_stat in model_stats:
            for i in range(cutn_batches):
                clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
                image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                #print("dists: ")
                #print(dists.sum())
                dists = dists.view([cutn, n, -1])
                losses = dists.mul(model_stat['weights']).sum(2).mean(0)
                scheduled_cgs = schedule_of(clip_guidance_scale, np.array(t.cpu())[0], net_steps)
                x_in_grad += torch.autograd.grad(losses.sum() * scheduled_cgs, x_in)[0] / cutn_batches
        tv_losses = tv_loss(x_in)
        if use_secondary_model is True:
            range_losses = range_loss(out)
        else:
            range_losses = range_loss(out['pred_xstart'])
        #print("range loss:")
        #print(range_losses.sum())
        sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
        scheduled_range_scale = schedule_of(range_scale, np.array(t.cpu())[0], net_steps)
        loss = tv_losses.sum() * tv_scale + range_losses.sum() * scheduled_range_scale + sat_losses * sat_scale        
        #print("loss: ")
        #print(loss)
        #if init is not None and init_scale:

        init_losses = lpips_model(x_in, init)
        scheduled_init_scale = schedule_of(init_scale, np.array(t.cpu())[0], net_steps)
        loss = loss + init_losses.sum() * scheduled_init_scale

        #symm_loss_metric = lpips_model
        """
        if 'h' in symmetry_loss_axis and np.array(t.cpu())[0] > 10 * symmetry_switch:
            sloss = symm_loss(x_in, True,
                            lpips_model,
                            h_symmetry_offset, 
                            symmetry_twist,
                            symmetry_ignore_color)
            loss = loss + sloss.sum() * symmetry_loss_scale          
        if 'v' in symmetry_loss_axis and np.array(t.cpu())[0] > 10 * symmetry_switch:
            sloss = symm_loss(x_in, False,
                            lpips_model, 
                            v_symmetry_offset, 
                            symmetry_twist,
                            symmetry_ignore_color)
            loss = loss + sloss.sum() * symmetry_loss_scale
        """
        x_in_grad += torch.autograd.grad(loss, x_in)[0]
        grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
        if clamp_grad:
            magnitude = grad.square().mean().sqrt()
            scheduled_clamp_max = schedule_of(clamp_max, np.array(t.cpu())[0], net_steps)
            scheduled_clamp_max = scheduled_clamp_max / 1000
            return grad * magnitude.clamp(max=scheduled_clamp_max) / magnitude
        return grad

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    for i in range(n_batches):
        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            randomize_class=True,
            cond_fn_with_grad=True,
        )

        for j, sample in enumerate(samples):
            #display.clear_output(wait=True)
            #if j % display_rate == 0 or cur_t == 0:
            if cur_t == 0:
                for k, image in enumerate(sample['pred_xstart']):
                    old_largest = get_past_outputs()
                    img_in_fname = f'img_in_{old_largest + 1:05}.png'
                    img_in.save(out_dir + '/' + img_in_fname, 'PNG')
                    filename = f'artifact_{old_largest + 1:05}.png'
                    img_result = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                    img_result = img_result.convert('RGBA')
                    img_result.save(out_dir +'/' + filename)
                    img_result = rb_swap(img_result)
                    img_result = img_result.resize(tuple(init_w_h))
                    settings_filename = f'artifact_{old_largest + 1:05}_settings.txt'
                    with open(out_dir + '/' + settings_filename, "w+") as f:   #save settings
                        json.dump(out_settings, f, ensure_ascii=False, indent=4)
                    #print(len(np.array(img_result, np.int32).tobytes()))
                    #if cur_t == 0:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    yield img_result
                    #display.clear_output(wait=True)
                    #tqdm.write(f'Batch {i}, step {j}, output {k}:')
                    #if j == 0:
                    #  display.display(display.Image(filename), display_id="main_display")
                    #else:
                    #  display.update_display(display.Image(filename), display_id="main_display")
            cur_t -= 1
    #return TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
    #print("do we reach this point at all?")
    # note, we do not reach this point
    #gc.collect()
    #torch.cuda.empty_cache()

#clear cache before running
gc.collect()
torch.cuda.empty_cache()
#do_run()

demo = gr.Interface(
    fn=do_run,
    title="Disco Spinach",
    inputs=[gr.Textbox(lines=1, value='Spirit of Symmetry, Flow, Mysterious Power, Jeff Simpson and Takashi Murakami'),
            gr.Textbox(lines=1, value='512, 512'),
            gr.Number(label="Steps", value=100, precision=0),
            gr.Textbox(label="Clip Guidance Scale", lines=1, value='6000|4000'),
            #gr.Number(label="Clip Guidance Scale", value=6000, precision=0),
            gr.Textbox(label="Range Scale", lines=1, value='150|120'),
            #gr.Number(label="Range Scale", value=120, precision=0),
            #gr.Number(label="Clamp Max ", value=50, precision=0),
            gr.Textbox(label="Clamp Max ", lines=1, value='50|60'),
            gr.Number(label='Cut Power', value=50, precision=0),
            gr.Number(label="Seed", value=64, precision=0),
            gr.Textbox(label="Init ImgData (byte string)"), #gr.Image(type="pil")
            gr.Number(label="Init Img Skip Steps", value=0.5, precision=2),
            #gr.Number(label="Init Img scale", value=1000, precision=0),
            gr.Textbox(label="Init Img scale", lines=1, value='1000|1200'),
            gr.Textbox(lines=1, value='512, 512')
            #gr.Number(label="Symmetry Switch", value=70, precision=0),
            #gr.Number(label="Symmetry Loss", value=1000, precision=0)
            #gr.Number(label="Display Rate", value=10, precision=0), #CGS
            ], #steps
    outputs="image")

demo.queue()
demo.launch()