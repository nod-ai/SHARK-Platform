# punet - Partitioned UNet in the style used by SDXL.

This is an experimental model aimed at finding a good partitioning strategy for
a class of devices which has a unique tile/memory hierarchy. It is fashioned
after UNet2DConditionModel in diffusers but only with features used by SDXL
implemented.

See the [huggingface repo](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/unet) for parameters and config.
