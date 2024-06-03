# punet - Partitioned UNet in the style used by SDXL.

This is an experimental, simplified variant of the `Unet2DConditionModel` from
diffusers. It can be run/compiled off of a `config.json` and `safetensors` files
from the `stable-diffusion-xl-base-1.0/unet` model.

It was originally written to experiment with sharding strategies (and the `p`
was for partitioned). It is now being used for quantization and other serving
optimizations as well.

It is pronounced like "pu-nay" or any other deriviation that matches the
mood of the user.

See the [huggingface repo](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/unet)
for parameters and config.

## Preparing dataset

If not sharding or quantizing the model, the official model can be imported
as is from huggingface:

```
model_dir=$(huggingface-cli download \
    stabilityai/stable-diffusion-xl-base-1.0 \
    unet/config.json unet/diffusion_pytorch_model.fp16.safetensors)
python -m sharktank.models.punet.tools.import_hf_dataset \
    --config-json $model_dir/unet/config.json \
    --output-irpa-file ~/models/punet_fp16.irpa
```

## Running reference model

```
python -m sharktank.models.punet.tools.run_diffuser_ref
```

## Run punet model

```
python -m sharktank.models.punet.tools.run_punet --irpa-file ~/models/punet_fp16.irpa
```

## License

Significant portions of this implementation were derived from diffusers,
licensed under Apache2: https://github.com/huggingface/diffusers
While much was a simple reverse engineering of the config.json and parameters,
code was taken where appropriate.
