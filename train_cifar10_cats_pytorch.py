from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms, utils

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


CIFAR10_CAT_LABEL = 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    data_dir: str = "/data"
    output_dir: str = "/runs/cifar10-cats-drift"
    feature_backend: str = "mae"
    feature_model: str = "facebook/vit-mae-base"
    resnet_repo: str = "Tudorx95/resnet18-cifar10-pytorch"
    resnet_filename: str = "ResNet18_CIFAR10.pth"
    resnet_layers: str = "conv1,layer1,layer2,layer3,layer4"
    image_size: int = 32
    mae_size: int = 224
    max_steps: int = 5000
    groups_per_gpu: int = 8
    pos_per_group: int = 8
    gen_per_group: int = 8
    hidden_size: int = 256
    depth: int = 6
    num_heads: int = 4
    patch_size: int = 4
    mlp_ratio: float = 4.0
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-2
    warmup_steps: int = 250
    grad_clip: float = 2.0
    ema_decay: float = 0.999
    seed: int = 42
    log_every: int = 10
    sample_every: int = 200
    save_every: int = 500
    use_wandb: bool = True
    wandb_project: str = "drifting-cifar10-cats"
    wandb_entity: str = "azabelo2121"
    wandb_name: str = "unconditional-32px-cifar10-cats"
    feature_layers: str = "3,6,9,12"
    drift_r: str = "0.2,0.05,0.02"
    num_workers: int = 2
    amp: str = "bf16"
    local_smoke: bool = False


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled:
        dist.barrier()
        dist.destroy_process_group()


def rank_zero(rank: int) -> bool:
    return rank == 0


def parse_ints(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_floats(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


class CatOnlyCIFAR10(Dataset):
    def __init__(self, root: str, train: bool, image_size: int, download: bool = True):
        datasets.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
        datasets.CIFAR10.tgz_md5 = "c58f30108f718f92721af3b95e74349a"
        if download:
            root_path = Path(root)
            archive = root_path / datasets.CIFAR10.filename
            extracted = root_path / datasets.CIFAR10.base_folder
            if archive.exists() and not extracted.exists():
                archive.unlink()
        self.base = datasets.CIFAR10(root=root, train=train, download=download)
        self.indices = [i for i, label in enumerate(self.base.targets) if label == CIFAR10_CAT_LABEL]
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip() if train else nn.Identity(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = self.base[self.indices[idx]]
        if self.image_size != 32:
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        return self.transform(image)


class SyntheticSmokeDataset(Dataset):
    def __init__(self, image_size: int, length: int = 128):
        generator = torch.Generator().manual_seed(1234)
        self.images = torch.randn(length, 3, image_size, image_size, generator=generator).clamp(-1, 1)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]


class InfiniteCatSampler:
    def __init__(self, dataset: Dataset, device: torch.device, seed: int):
        self.dataset = dataset
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        count = math.prod(shape)
        idx = torch.randint(len(self.dataset), (count,), generator=self.generator)
        images = torch.stack([self.dataset[int(i)] for i in idx], dim=0)
        return images.view(*shape, *images.shape[1:]).to(self.device, non_blocking=True)


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size * self.grid_size, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_a[:, None] * h
        h = self.norm2(x) * (1 + scale_m[:, None]) + shift_m[:, None]
        x = x + gate_m[:, None] * self.mlp(h)
        return x


class SmallUnconditionalDriftGenerator(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, hidden_size)
        self.cond = nn.Parameter(torch.zeros(1, hidden_size))
        self.blocks = nn.ModuleList(
            [AdaLNBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.out = nn.Linear(hidden_size, patch_size * patch_size * 3)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(batch_size, 3, self.image_size, self.image_size, device=device)
        tokens = self.patch_embed(x)
        cond = self.cond.expand(batch_size, -1)
        for block in self.blocks:
            tokens = block(tokens, cond)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        tokens = self.norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        patches = self.out(tokens)
        return torch.tanh(self.unpatchify(patches))

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, d = patches.shape
        p = self.patch_size
        grid = int(math.sqrt(n))
        patches = patches.view(b, grid, grid, p, p, 3)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(b, 3, self.image_size, self.image_size)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in self.shadow.items():
            value.mul_(self.decay).add_(state[key].detach(), alpha=1 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1).unsqueeze(1)
    xy = x @ y.transpose(-1, -2)
    return (x2 + y2 - 2 * xy).clamp_min(eps).sqrt()


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    r_list: Iterable[float],
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    b, c_g, dim = gen.shape
    c_p = fixed_pos.shape[1]
    if weight_gen is None:
        weight_gen = torch.ones(b, c_g, device=gen.device, dtype=gen.dtype)
    if weight_pos is None:
        weight_pos = torch.ones(b, c_p, device=gen.device, dtype=gen.dtype)

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_pos.detach()], dim=1)
    targets_w = torch.cat([weight_gen, weight_pos], dim=1)
    dist = cdist(old_gen.float(), targets.float())
    weighted_dist = dist * targets_w[:, None].float()
    scale = weighted_dist.mean(dim=(1, 2), keepdim=True) / targets_w.float().mean(dim=1)[:, None, None]
    scale_inputs = (scale / math.sqrt(dim)).clamp_min(1e-3)
    old_gen_scaled = old_gen.float() / scale_inputs
    targets_scaled = targets.float() / scale_inputs
    dist_normed = dist / scale.clamp_min(1e-3)
    diag = torch.eye(c_g, device=gen.device, dtype=gen.dtype).view(1, c_g, c_g)
    dist_normed[:, :, :c_g] = dist_normed[:, :, :c_g] + 100.0 * diag

    force = torch.zeros_like(old_gen_scaled)
    info: Dict[str, torch.Tensor] = {"scale": scale.mean()}
    for radius in r_list:
        logits = -dist_normed / radius
        affinity = (logits.softmax(dim=-1) * logits.softmax(dim=-2)).clamp_min(1e-6).sqrt()
        affinity = affinity * targets_w[:, None].float()
        aff_neg = affinity[:, :, :c_g]
        aff_pos = affinity[:, :, c_g:]
        sum_pos = aff_pos.sum(dim=-1, keepdim=True)
        sum_neg = aff_neg.sum(dim=-1, keepdim=True)
        coeff = torch.cat([-aff_neg * sum_pos, aff_pos * sum_neg], dim=2)
        force_r = coeff @ targets_scaled
        force_r = force_r - coeff.sum(dim=-1, keepdim=True) * old_gen_scaled
        force_norm = force_r.square().mean().clamp_min(1e-8).sqrt()
        force = force + force_r / force_norm
        info[f"loss_{radius}"] = force_norm.detach()

    goal = (old_gen_scaled + force).detach()
    loss = ((gen.float() / scale_inputs) - goal).square().mean(dim=(1, 2))
    return loss.mean(), info


class MAEFeatures(nn.Module):
    def __init__(self, model_name: str, mae_size: int, layers: Sequence[int], smoke: bool = False):
        super().__init__()
        self.mae_size = mae_size
        self.layers = layers
        self.smoke = smoke
        if smoke:
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GELU(),
            )
        else:
            from transformers import AutoConfig, AutoImageProcessor, ViTMAEModel

            config = AutoConfig.from_pretrained(model_name)
            config.mask_ratio = 0.0
            config.output_hidden_states = True
            self.model = ViTMAEModel.from_pretrained(model_name, config=config)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = (images + 1.0) * 0.5
        x = F.interpolate(x, size=(self.mae_size, self.mae_size), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        if self.smoke:
            feat = self.model(x)
            tokens = feat.flatten(2).transpose(1, 2)
            return {
                "smoke_conv": F.layer_norm(tokens, tokens.shape[-1:]),
                "smoke_conv_mean": tokens.mean(dim=1, keepdim=True),
            }
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states
        features: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            idx = min(layer, len(hidden) - 1)
            feat = hidden[idx][:, 1:, :]
            features[f"mae_layer_{idx}"] = F.layer_norm(feat, feat.shape[-1:])
            features[f"mae_layer_{idx}_mean"] = feat.mean(dim=1, keepdim=True)
        return features


class CIFAR10ResNet18Features(nn.Module):
    def __init__(self, repo_id: str, filename: str, layers: Sequence[str]):
        super().__init__()
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        self.layers = set(layers)
        self.model = models.resnet18(weights=None, num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        norm = checkpoint.get("normalization", {})
        mean = norm.get("mean", [0.4914, 0.4822, 0.4465])
        std = norm.get("std", [0.2023, 0.1994, 0.2010])
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def _store(self, features: Dict[str, torch.Tensor], name: str, x: torch.Tensor) -> None:
        if name not in self.layers:
            return
        tokens = x.flatten(2).transpose(1, 2)
        features[f"resnet18_{name}"] = F.layer_norm(tokens, tokens.shape[-1:])
        features[f"resnet18_{name}_mean"] = tokens.mean(dim=1, keepdim=True)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = (images + 1.0) * 0.5
        x = (x - self.mean) / self.std
        features: Dict[str, torch.Tensor] = {}

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        self._store(features, "conv1", x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        self._store(features, "layer1", x)
        x = self.model.layer2(x)
        self._store(features, "layer2", x)
        x = self.model.layer3(x)
        self._store(features, "layer3", x)
        x = self.model.layer4(x)
        self._store(features, "layer4", x)
        return features


def build_feature_extractor(cfg: TrainConfig) -> nn.Module:
    backend = cfg.feature_backend.strip().lower()
    if backend == "mae":
        layers = parse_ints(cfg.feature_layers)
        return MAEFeatures(cfg.feature_model, cfg.mae_size, layers, smoke=cfg.local_smoke)
    if backend in {"resnet18_cifar10", "cifar10_resnet18", "resnet"}:
        layers = [name.strip() for name in cfg.resnet_layers.split(",") if name.strip()]
        if not layers:
            raise ValueError("--resnet-layers must contain at least one layer name")
        return CIFAR10ResNet18Features(cfg.resnet_repo, cfg.resnet_filename, layers)
    raise ValueError(f"Unsupported feature_backend={cfg.feature_backend!r}")


def feature_drift_loss(
    feature_model: nn.Module,
    generated: torch.Tensor,
    positives: torch.Tensor,
    groups: int,
    gen_per_group: int,
    pos_per_group: int,
    r_list: Sequence[float],
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    pos_flat = positives.view(groups * pos_per_group, *positives.shape[2:])
    with torch.no_grad():
        pos_features = feature_model(pos_flat)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        gen_features = feature_model(generated)

    total = generated.new_tensor(0.0)
    metrics: Dict[str, float] = {}
    count = 0
    for name, gen_feat in gen_features.items():
        pos_feat = pos_features[name].to(dtype=gen_feat.dtype)
        _, tokens, dim = gen_feat.shape
        gen_grouped = gen_feat.view(groups, gen_per_group, tokens, dim).permute(0, 2, 1, 3)
        pos_grouped = pos_feat.view(groups, pos_per_group, tokens, dim).permute(0, 2, 1, 3)
        gen_grouped = gen_grouped.reshape(groups * tokens, gen_per_group, dim)
        pos_grouped = pos_grouped.reshape(groups * tokens, pos_per_group, dim)
        loss, info = drift_loss(gen_grouped, pos_grouped, r_list)
        total = total + loss
        metrics[f"loss/{name}"] = float(loss.detach())
        metrics[f"scale/{name}"] = float(info["scale"].detach())
        count += 1
    return total / max(count, 1), metrics


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)


def lr_for_step(step: int, cfg: TrainConfig) -> float:
    if cfg.warmup_steps <= 0:
        return cfg.lr
    return cfg.lr * min(1.0, (step + 1) / cfg.warmup_steps)


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
        },
        output_dir / f"checkpoint_step_{step:07d}.pt",
    )


@torch.no_grad()
def save_samples(output_dir: Path, step: int, model: nn.Module, device: torch.device, cfg: TrainConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    samples = raw_model(64, device).float().cpu()
    samples = (samples + 1.0) * 0.5
    path = output_dir / f"samples_step_{step:07d}.png"
    utils.save_image(samples, path, nrow=8)
    return path


def init_wandb(cfg: TrainConfig, rank: int):
    if cfg.local_smoke or not cfg.use_wandb or not rank_zero(rank):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without W&B logging.", flush=True)
        return None

    mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    if mode == "offline":
        print("WANDB_API_KEY is not set; logging W&B data in offline mode.", flush=True)

    def _start(entity: str | None, name: str):
        return wandb.init(
            project=cfg.wandb_project,
            entity=entity or None,
            name=name,
            config=asdict(cfg) | {"requested_wandb_entity": cfg.wandb_entity},
            mode=mode,
        )

    try:
        return _start(cfg.wandb_entity, cfg.wandb_name)
    except Exception as exc:
        if not cfg.wandb_entity:
            print(f"W&B init failed; continuing without W&B: {exc}", flush=True)
            return None
        print(
            f"W&B entity {cfg.wandb_entity!r} was rejected ({exc}); "
            "retrying with the API key's default entity.",
            flush=True,
        )
        try:
            wandb.finish()
        except Exception:
            pass
        try:
            return _start(None, f"{cfg.wandb_name}-default-entity")
        except Exception as retry_exc:
            print(f"W&B retry failed; continuing without W&B: {retry_exc}", flush=True)
            return None


def train(cfg: TrainConfig) -> None:
    distributed, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)

    output_dir = Path(cfg.output_dir)
    if rank_zero(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    wandb_run = init_wandb(cfg, rank)

    if cfg.local_smoke:
        dataset = SyntheticSmokeDataset(cfg.image_size)
    elif distributed:
        if rank_zero(rank):
            dataset = CatOnlyCIFAR10(cfg.data_dir, train=True, image_size=cfg.image_size, download=True)
            dist.barrier()
        else:
            dist.barrier()
            dataset = CatOnlyCIFAR10(cfg.data_dir, train=True, image_size=cfg.image_size, download=False)
    else:
        dataset = CatOnlyCIFAR10(cfg.data_dir, train=True, image_size=cfg.image_size, download=True)
    sampler = InfiniteCatSampler(dataset, device, cfg.seed + rank * 100003)
    r_list = parse_floats(cfg.drift_r)

    generator = SmallUnconditionalDriftGenerator(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
    ).to(device)
    if distributed:
        generator = DistributedDataParallel(generator, device_ids=[local_rank], output_device=local_rank)
    ema = EMA(generator.module if isinstance(generator, DistributedDataParallel) else generator, cfg.ema_decay)
    optimizer = make_optimizer(generator, cfg)
    feature_model = build_feature_extractor(cfg).to(device)
    amp_dtype = torch.bfloat16 if cfg.amp == "bf16" and device.type == "cuda" else None

    if rank_zero(rank):
        print(
            f"training unconditional 32x32 CIFAR-10 cats with {world_size} process(es), "
            f"{len(dataset)} cat images, model params="
            f"{sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M",
            flush=True,
        )

    start = time.time()
    for step in range(cfg.max_steps):
        lr = lr_for_step(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        positives = sampler.sample((cfg.groups_per_gpu, cfg.pos_per_group))
        generated = generator(cfg.groups_per_gpu * cfg.gen_per_group, device)
        loss, metrics = feature_drift_loss(
            feature_model=feature_model,
            generated=generated,
            positives=positives,
            groups=cfg.groups_per_gpu,
            gen_per_group=cfg.gen_per_group,
            pos_per_group=cfg.pos_per_group,
            r_list=r_list,
            amp_dtype=amp_dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        raw_model = generator.module if isinstance(generator, DistributedDataParallel) else generator
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        optimizer.step()
        ema.update(raw_model)

        if rank_zero(rank) and (step == 0 or (step + 1) % cfg.log_every == 0):
            elapsed = time.time() - start
            msg = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "lr": lr,
                "sec_per_step": elapsed / (step + 1),
            }
            msg.update(metrics)
            print(json.dumps(msg), flush=True)
            if wandb_run is not None:
                import wandb

                wandb.log(msg, step=step + 1)

        if rank_zero(rank) and (step == 0 or (step + 1) % cfg.sample_every == 0):
            sample_path = save_samples(output_dir / "samples", step + 1, generator, device, cfg)
            if wandb_run is not None:
                import wandb

                wandb.log({"samples": wandb.Image(str(sample_path))}, step=step + 1)

        if rank_zero(rank) and ((step + 1) % cfg.save_every == 0 or step + 1 == cfg.max_steps):
            save_checkpoint(output_dir, step + 1, generator, ema, optimizer, cfg)

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(distributed)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    for field_name, field_def in TrainConfig.__dataclass_fields__.items():
        value = field_def.default
        arg_type = type(value)
        if arg_type is bool:
            arg_name = field_name.replace("_", "-")
            if value:
                parser.add_argument(f"--no-{arg_name}", action="store_false", dest=field_name, default=value)
            else:
                parser.add_argument(f"--{arg_name}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{field_name.replace('_', '-')}", type=arg_type, default=value)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
