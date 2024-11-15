import os
import subprocess
import sys

def start_server(port, env_vars, extra_args=None):
    env = os.environ.copy()
    env.update(env_vars)

    command = [sys.executable, "-m", "shortfin_apps.sd.server"]

    if extra_args:
        command.extend(extra_args)

    subprocess.Popen(command, env=env)

for i in range(64):
    port = 8000 + i + 1
    env_vars = {
        "ROCR_VISIBLE_DEVICES": f"{i%64}",
    }

    extra_args = [
        "--device", "amdgpu",
        "--build_preference", "precompiled",
        "--topology", "cpx_single",
        "--artifacts_dir", "/home/esaimana/shark_artifacts",
        "--tuning_spec", "/home/esaimana/shark_artifacts/genfiles/sdxlconfig/attention_and_matmul_spec_gfx942.mlir",
        "--port", f"{port}"
    ]

    start_server(port, env_vars, extra_args)
