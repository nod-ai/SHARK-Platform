
"""Configuration objects.

Parameters that are intrinsic to a specific model.

Typically represented in something like a Huggingface config.json,
we extend the configuration to enumerate inference boundaries of some given set of compiled modules.

This is much less complex in Stable Diffusion where we aren't keeping a KVCache, but is still useful for serving.
"""

from dataclasses import dataclass
from pathlib import Path

import dataclasses_json
from dataclasses_json import dataclass_json, Undefined

import shortfin.array as sfnp


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModelParams:
    """Parameters for a specific set of compiled SD submodels, sufficient to do batching /
    invocations."""

    # Maximum length of prompt sequence.
    max_seq_len: int

    # Batch sizes that each stage is compiled for. These are expected to be
    # functions exported from the model with suffixes of "_bs{batch_size}". Must
    # be in ascending order.
    clip_batch_sizes: list[int]

    # Similarly, batch sizes that the decode stage is compiled for.
    unet_batch_sizes: list[int]

    # Same for VAE.
    vae_batch_sizes: list[int]

    # Height and Width, respectively, for which Unet and VAE are compiled. e.g. [[512, 512], [1024, 1024]]
    dims: list[list[int]]

    # Name of the IREE module for each submodel.
    clip_module_name: str = "compiled_text_encoder"
    unet_module_name: str = "compiled_punet"
    vae_module_name: str = "compiled_vae"

    # DTypes (basically defaults):
    clip_dtype: sfnp.DType = sfnp.float16
    unet_dtype: sfnp.DType = sfnp.float16
    vae_dtype: sfnp.DType = sfnp.float16

    # ABI of the module.
    module_abi_version: int = 1

    @property
    def max_clip_batch_size(self) -> int:
        return self.clip_batch_sizes[-1]
    
    @property
    def max_unet_batch_size(self) -> int:
        return self.unet_batch_sizes[-1]

    @property
    def max_vae_batch_size(self) -> int:
        return self.vae_batch_sizes[-1]

    @property
    def max_batch_size(self):
        # TODO: a little work on the batcher should loosen this up.
        return max(self.clip_batch_sizes, self.unet_batch_sizes, self.vae_batch_sizes)

    @staticmethod
    def load_json(path: Path | str):
        with open(path, "rt") as f:
            json_text = f.read()
        return ModelParams.from_json(json_text)