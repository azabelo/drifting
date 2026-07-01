from __future__ import annotations

import subprocess

import modal


app = modal.App("drifting-cifar10-cats")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .pip_install(
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "einops==0.8.2",
        "Pillow==12.1.1",
        "tqdm==4.67.1",
        "wandb==0.25.1",
    )
    .add_local_dir(".", remote_path="/root/drifting")
)

runs_volume = modal.Volume.from_name("drifting-cifar10-cats-runs", create_if_missing=True)
data_volume = modal.Volume.from_name("drifting-cifar10-cats-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G:4",
    cpu=16,
    memory=96 * 1024,
    timeout=60 * 60 * 12,
    volumes={"/runs": runs_volume, "/data": data_volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_on_4xa10(
    max_steps: int = 10000,
    groups_per_gpu: int = 32,
    pos_per_group: int = 8,
    gen_per_group: int = 8,
    feature_backend: str = "resnet18_cifar10",
):
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        "/root/drifting/train_cifar10_cats_pytorch.py",
        "--data-dir=/data",
        "--output-dir=/runs/cifar10-cats-drift-resnet18",
        f"--max-steps={max_steps}",
        f"--groups-per-gpu={groups_per_gpu}",
        f"--pos-per-group={pos_per_group}",
        f"--gen-per-group={gen_per_group}",
        f"--feature-backend={feature_backend}",
        "--feature-model=facebook/vit-mae-base",
        "--feature-layers=3,6,9,12",
        "--resnet-repo=Tudorx95/resnet18-cifar10-pytorch",
        "--resnet-filename=ResNet18_CIFAR10.pth",
        "--resnet-layers=conv1,layer1,layer2,layer3,layer4",
        "--sample-every=200",
        "--save-every=500",
        "--log-every=10",
        "--wandb-project=drifting-cifar10-cats",
        "--wandb-entity=azabelo2121",
        "--wandb-name=4xa10-resnet18feat-unconditional-32px-cifar10-cats",
    ]
    subprocess.run(cmd, check=True, cwd="/root/drifting")
    runs_volume.commit()
    data_volume.commit()


@app.local_entrypoint()
def main(
    max_steps: int = 10000,
    groups_per_gpu: int = 32,
    pos_per_group: int = 8,
    gen_per_group: int = 8,
    feature_backend: str = "resnet18_cifar10",
):
    train_on_4xa10.remote(
        max_steps=max_steps,
        groups_per_gpu=groups_per_gpu,
        pos_per_group=pos_per_group,
        gen_per_group=gen_per_group,
        feature_backend=feature_backend,
    )
