
##
##      CNET AI
##

import os
import sys
import argparse
from datetime import datetime

## init
sys.path.insert(0, './ai')

from share import *
import config

import PIL
import cv2
from PIL import Image

import einops
import numpy as np
import torch
import random

try:
    from pytorch_lightning import seed_everything
    from annotator.util import resize_image, HWC3
    from annotator.canny import CannyDetector
    from cldm.model import create_model, load_state_dict
    from cldm.ddim_hacked import DDIMSampler

    preprocessor = None
    _rootModels = './ai/models'
    model_name = 'control_v11p_sd15_canny'
    model = create_model(f'{_rootModels}/{model_name}.yaml').cpu()
    model.load_state_dict(load_state_dict(f'{_rootModels}/v1-5-pruned.ckpt', location='cuda'), strict=False)
    model.load_state_dict(load_state_dict(f'{_rootModels}/{model_name}.pth', location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
except Exception as err:
    raise err


## where to save the user profile?
def fnGetUserdataPath(_username):
    _path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_PROFILE_DIR = os.path.join(_path, '_profile')
    USER_PROFILE_DIR = os.path.join(DEFAULT_PROFILE_DIR, _username)
    return {
        "location": USER_PROFILE_DIR,
        "voice": False,
        "picture": True
    }

## WARMUP Data
def getWarmupData(_id):
    try:
        import time
        from werkzeug.datastructures import MultiDict
        ts=int(time.time())
        sample_args = MultiDict([
            ('-u', 'test_user'),
            ('-uid', str(ts)),
            ('-t', _id),
            ('-cycle', '0'),
            ('-p', 'women from forest deep look photo realistic'),
            ('-o', 'warmup.jpg'),
            ('-filename', 'warmup.jpg')
        ])
        return sample_args
    except:
        print("Could not call warm up!\r\n")
        return None

def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold, input_file, output_file, indir, outdir):
    global preprocessor
    global ddim_sampler
    global model
    aOutput=[]

    if det == 'Canny':
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        ## added bit ==> WE SAVE !!
        file_name, file_extension = os.path.splitext(input_file)
        _samp=len(x_samples)
        for x_sample in x_samples:
            # x_sample = 255. * einops.rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            _filename=os.path.splitext(output_file)[0]+"_"+f"{_samp}_{ddim_steps}"+file_extension
            _filename=os.path.join(outdir, _filename)
            Image.fromarray(x_sample.astype(np.uint8)).save(_filename)
            aOutput.append(_filename)

    return aOutput


def fnRun(_args): 
    # Create the parser
    vq_parser = argparse.ArgumentParser(description='Control Net')

    # OSAIS arguments
    vq_parser.add_argument("-odir", "--outdir", type=str, help="Output directory", default="./_output/", dest='outdir')
    vq_parser.add_argument("-idir", "--indir", type=str, help="input directory", default="./_input/", dest='indir')

    # Add the arguments
    vq_parser.add_argument("-filename","--filename", type=str, help="Input image", default="transparent.png", dest='input_file')
    vq_parser.add_argument("-p",    "--prompt", type=str, help="Text prompts", default=None, dest='prompt')
    vq_parser.add_argument("-seed",  "--seed", type=int, help="Seed", default=12345, dest='seed')
    vq_parser.add_argument("-preproc",  "--preprocessor", type=str, help="", default="Canny", dest='det')

    vq_parser.add_argument("-low",  "--low", type=int, help="Canny low threshold", default=100, dest='low_threshold')
    vq_parser.add_argument("-high",  "--high", type=int, help="Canny high threshold", default=200, dest='high_threshold')
    vq_parser.add_argument("-nsamples",  "--n_samples", type=int, help="Images", default=1, dest='num_samples')
    vq_parser.add_argument("-res",  "--res", type=int, help="Image Resolution", default=512, dest='image_resolution')
    vq_parser.add_argument("-detres",  "--detres", type=int, help="Preprocessor Resolution", default=512, dest='detect_resolution')
    vq_parser.add_argument("-strength",  "--strength", type=float, help="Control Strength", default=1.0, dest='strength')
    vq_parser.add_argument("-scale",  "--scale", type=float, help="Guidance scale", default=9.0, dest='scale')
    vq_parser.add_argument("-steps",  "--ddim_steps", type=int, help="Steps", default=20, dest='ddim_steps')
    vq_parser.add_argument("-eta",  "--eta", type=float, help="DDIM ETA", default=1, dest='eta')
    vq_parser.add_argument("-ap",    "--a_prompts", type=str, help="Added Prompt", default='best quality', dest='a_prompt')
    vq_parser.add_argument("-np",    "--n_prompts", type=str, help="Negative Prompt", default='lowres, bad anatomy, bad hands, cropped, worst quality', dest='n_prompt')
    vq_parser.add_argument("-gm",    "--guessmode", type=bool, help="Images", default=False, dest='guess_mode')
#    vq_parser.add_argument("-se",  "--se", type=int, help="Save Intervals", default=0, dest='save_interval')


    vq_parser.add_argument("-model",  "--ckpt", type=str, help="model", default="", dest='model')
    vq_parser.add_argument("-o", "--output", type=str, nargs="?", help="filename to write results to", default="result.jpg", dest="output_file")


    # Execute the parse_args() method
    try:
        args = vq_parser.parse_args(_args)
        print(args)

        beg_date = datetime.utcnow()

        _output = os.path.join(args.outdir, args.output_file)
        _input=None
        numpy_image=None
        if(args.input_file):
            _input = os.path.join(args.indir, args.input_file)
            pil_image = Image.open(_input).convert("RGB")
            numpy_image = np.array(pil_image)
        aRes=process(args.det, numpy_image, args.prompt, args.a_prompt, args.n_prompt, args.num_samples, args.image_resolution, args.detect_resolution, args.ddim_steps, args.guess_mode, args.strength, args.scale, args.seed, args.eta, args.low_threshold, args.high_threshold, args.input_file, args.output_file, args.indir, args.outdir)

        ## return output
        end_date = datetime.utcnow()
        return {
            "beg_date": beg_date,
            "end_date": end_date,
            "mCost": 1.1,            ## cost multiplier of this AI
            "aFile": aRes
        }

    except Exception as err:
        print("\r\nCRITICAL ERROR!!!")
        raise err
