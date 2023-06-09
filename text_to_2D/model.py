from typing import Any
from diffusers import StableDiffusionControlNetPipeline
from PIL import Image


class FaceControlNet:
    def __init__(self, prior_sketch_path="text_to_2D/prior_sketches/mask.jpg", pretrained_model_name="stabilityai/stable-diffusion-2-1"):
        self.model = StableDiffusionControlNetPipeline.from_pretrained(pretrained_model_name)
        self.prior_sketch = Image.open(prior_sketch_path).resize((512, 512))
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(image=self.prior_sketch, *args, **kwds)


FaceControlNet()