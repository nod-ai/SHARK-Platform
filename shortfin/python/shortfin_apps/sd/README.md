# SD Server and CLI

This directory contains a SD inference server, CLI and support components.


## Quick start

In your shortfin environment,
```
pip install transformers
pip install dataclasses-json
pip install pillow

```
```
python -m shortfin_apps.sd.server --help
```

## Run on MI300x

 - Follow quick start

 - Download runtime artifacts (vmfbs, weights):

```
mkdir vmfbs
mkdir weights
wget https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/vmfbs/stable_diffusion_xl_base_1_0_bs1_64_1024x1024_i8_punet_gfx942.vmfb -O vmfbs/stable_diffusion_xl_base_1_0_bs1_64_1024x1024_i8_punet_gfx942.vmfb
wget https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/vmfbs/stable_diffusion_xl_base_1_0_bs1_64_fp16_text_encoder_gfx942.vmfb -O vmfbs/stable_diffusion_xl_base_1_0_bs1_64_fp16_text_encoder_gfx942.vmfb
wget https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/vmfbs/stable_diffusion_xl_base_1_0_EulerDiscreteScheduler_bs1_1024x1024_fp16_gfx942.vmfb -O vmfbs/stable_diffusion_xl_base_1_0_EulerDiscreteScheduler_bs1_1024x1024_fp16_gfx942.vmfb
wget https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/vmfbs/stable_diffusion_xl_base_1_0_bs1_1024x1024_fp16_vae_gfx942.vmfb -O vmfbs/stable_diffusion_xl_base_1_0_bs1_1024x1024_fp16_vae_gfx942.vmfb

# You can download real weights with:
wget https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/weights/sfsd_weights_1023.zip
# Splat weights:
wget https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/weights/sfsd_splat_1023.zip
```
 - Unzip the downloaded weights archive to /weights
 - Run CLI server interface (you can find `sdxl_config_i8.json` in shortfin_apps/sd/examples):

```
python -m shortfin_apps.sd.server --model_config=./sdxl_config_i8.json --clip_vmfb=./vmfbs/stable_diffusion_xl_base_1_0_bs1_64_fp16_text_encoder_gfx942.vmfb --unet_vmfb=./vmfbs/stable_diffusion_xl_base_1_0_bs1_64_1024x1024_i8_punet_gfx942.vmfb --scheduler_vmfb=./vmfbs/stable_diffusion_xl_base_1_0_EulerDiscreteScheduler_bs1_1024x1024_fp16_gfx942.vmfb --vae_vmfb=./vmfbs/stable_diffusion_xl_base_1_0_bs1_1024x1024_fp16_vae_gfx942.vmfb --clip_params=./weights/stable_diffusion_xl_base_1_0_text_encoder_fp16.safetensors --unet_params=./weights/stable_diffusion_xl_base_1_0_punet_dataset_i8.irpa --vae_params=./weights/stable_diffusion_xl_base_1_0_vae_fp16.safetensors --device=amdgpu --device_ids=0
```
with splat:
```
python -m shortfin_apps.sd.server --model_config=./sdxl_config_i8.json --clip_vmfb=./vmfbs/stable_diffusion_xl_base_1_0_bs1_64_fp16_text_encoder_gfx942.vmfb --unet_vmfb=./vmfbs/stable_diffusion_xl_base_1_0_bs1_64_1024x1024_i8_punet_gfx942.vmfb --scheduler_vmfb=./vmfbs/stable_diffusion_xl_base_1_0_EulerDiscreteScheduler_bs1_1024x1024_fp16_gfx942.vmfb --vae_vmfb=./vmfbs/stable_diffusion_xl_base_1_0_bs1_1024x1024_fp16_vae_gfx942.vmfb --clip_params=./weights/clip_splat.irpa --unet_params=./weights/punet_splat_18.irpa --vae_params=./weights/vae_splat.irpa --device=amdgpu --device_ids=0
```
 - Run a request in a separate shell:
```
python shortfin/python/shortfin_apps/sd/examples/send_request.py --file=shortfin/python/shortfin_apps/sd/examples/sdxl_request.json
```
