from typing import Any
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image
import torch
import numpy as np

class FaceControlNet:
    def __init__(
            self,
            prior_sketch_path="text_to_2D/prior_sketches/mask.jpg",
            diffusion_model_name="runwayml/stable-diffusion-v1-5",
            controlnet_model_name="lllyasviel/control_v11p_sd15_openpose"
    ):  
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_name, 
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.model = StableDiffusionControlNetPipeline.from_pretrained(
            diffusion_model_name, 
            controlnet=controlnet,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        image = Image.open(prior_sketch_path).resize((512, 512))
        image = np.array(image)[:, :, None]
        self.prior_sketch = Image.fromarray(np.concatenate([image, image, image], axis=2))
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, image=self.prior_sketch, **kwds).images

