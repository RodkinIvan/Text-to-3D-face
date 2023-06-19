import sys; sys.path.append('..')

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from annotator.pidinet import PidiNetDetector
from annotator.util import nms
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from gen3d.gen_3d_file import convert_3d_file


from clip2latent import models
from PIL import Image
import os


# controlnet
preprocessor = None
model_name = 'control_v11p_sd15_scribble'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# clip2latent
device = 'cuda'
checkpoint = 'https://huggingface.co/lambdalabs/clip2latent/resolve/main/ffhq-sg2-510.ckpt'
cfg_file = 'https://huggingface.co/lambdalabs/clip2latent/resolve/main/ffhq-sg2-510.yaml'
model_c2l = models.Clip2StyleGAN(cfg_file, device, checkpoint)


def infer_controlnet(prompt, input_image='image_prior.jpg', a_prompt='best quality', n_prompt='lowres, worst quality', num_samples=1, image_resolution=512, det='PIDI', detect_resolution=512, ddim_steps=20, guess_mode=False, strength=1.0, scale=9.0, seed=42, eta=1.0):
    global preprocessor

    if 'HED' in det:
        if not isinstance(preprocessor, HEDdetector):
            preprocessor = HEDdetector()

    if 'PIDI' in det:
        if not isinstance(preprocessor, PidiNetDetector):
            preprocessor = PidiNetDetector()

    with torch.no_grad():
        input_image = cv2.imread(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {'c_concat': [control], 'c_crossattn': [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {'c_concat': None if guess_mode else [control], 'c_crossattn': [model.get_learned_conditioning([n_prompt] * num_samples)]}
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

        results = [x_samples[i] for i in range(num_samples)]
        # save first image
        cv2.imwrite('face_controlnet.png', cv2.cvtColor(x_samples[0], cv2.COLOR_RGB2BGR))
    #return [detected_map] + results
    return results


@torch.no_grad()
def infer_c2l(prompt, n_samples=1, scale=2, skips=250):
    images, clip_score = model_c2l(prompt, n_samples_per_txt=n_samples, cond_scale=scale, skips=skips, clip_sort=True)
    images = images.cpu()
    make_im = lambda x: (255*x.clamp(-1, 1)/2 + 127.5).to(torch.uint8).permute(1,2,0).numpy()
    images = [Image.fromarray(make_im(x)) for x in images]
    # save first image
    images[0].save('face_c2l.png')
    return images


def process_2D_to_3D(image_path, extract_path=''):
    convert_3d_file(image_path, extract_path)
    glb_path = os.path.join(extract_path, 'avatar/model.glb')
    return glb_path


block = gr.Blocks().queue()
with block:
    title = 'Generate a 3D avatar from a text description'
    gr.Markdown("<h1 style='text-align: center'>" + title + '</h1>')

    with gr.Row():
        with gr.Column():
            gr.Markdown('## ControlNet')
            prompt_controlnet = gr.Textbox(label='Prompt')
            run_button_controlnet = gr.Button(value='Run to generate an image with ControlNet')
            result_gallery_controlnet = gr.Gallery(label='Output', show_label=False, elem_id='gallery').style(grid=1, height='auto')
            run_button_2D_to_3D_controlnet = gr.Button(value='Run to render in 3D')
            result_3d_controlnet = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label='3D Model')
        with gr.Column():
            gr.Markdown('## Clip2latent')
            prompt_c2l = gr.Textbox(label='Prompt')
            run_button_c2l = gr.Button(value='Run to generate an image with Clip2latent')
            result_gallery_c2l = gr.Gallery(label='Output', show_label=False, elem_id='gallery').style(grid=1, height='auto')
            run_button_2D_to_3D_c2l = gr.Button(value='Run to render in 3D')
            result_3d_c2l = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label='3D Model')
    
    ips_controlnet = [prompt_controlnet]
    ips_2D_to_3D_controlnet = [gr.Textbox(value='face_controlnet.png', visible=False), gr.Textbox(value='controlnet', visible=False)]
    run_button_controlnet.click(fn=infer_controlnet, inputs=ips_controlnet, outputs=[result_gallery_controlnet])
    run_button_2D_to_3D_controlnet.click(fn=process_2D_to_3D, inputs=ips_2D_to_3D_controlnet, outputs=[result_3d_controlnet])

    ips_c2l = [prompt_c2l]
    ips_2D_to_3D_c2l = [gr.Textbox(value='face_c2l.png', visible=False), gr.Textbox(value='c2l', visible=False)]
    run_button_c2l.click(fn=infer_c2l, inputs=ips_c2l, outputs=[result_gallery_c2l])
    run_button_2D_to_3D_c2l.click(fn=process_2D_to_3D, inputs=ips_2D_to_3D_c2l, outputs=[result_3d_c2l])


block.launch(server_name='0.0.0.0')
