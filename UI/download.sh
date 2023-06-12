wget -O Duck.glb https://github.com/gradio-app/gradio/raw/main/demo/model3D/files/Duck.glb
mkdir -p models && cd models
wget -O control_v11p_sd15_scribble.yaml https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.yaml
wget -O control_v11p_sd15_scribble.pth https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth
wget -O v1-5-pruned.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
