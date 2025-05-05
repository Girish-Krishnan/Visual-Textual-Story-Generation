"""
Joint Visual–Textual Story Generator
Girish Krishnan, 2025
===================================
This single‑file prototype implements the full pipeline:
  • Text autoregressive model  (GPT‑family via 🤗 `transformers`)
  • U‑ViT latent‑diffusion      (Stable‑Diffusion via 🤗 `diffusers`)
  • CLIP‑style dual encoder     (🤗 `transformers`)
  • Vector‑Quantised VAE        (lightweight PyTorch implementation)
  • Prior over storyboard codes (small Transformer)
  • Multi‑stage training loops  (pre‑training ➜ alignment ➜ joint ELBO)

The code is runnable end‑to‑end on a multi‑GPU workstation (≥2×A100 recommended).
Heavy models are loaded from the HuggingFace Hub; adjust the identifiers in
`Config` if you prefer local checkpoints.
"""
from __future__ import annotations

import os
import math
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPModel,
    CLIPProcessor,
    get_cosine_schedule_with_warmup,
)
from diffusers import StableDiffusionPipeline, AutoencoderKL
from accelerate import Accelerator, DistributedType

# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data ------------------------------------------------------------------
    data_root: str = "data"  # folder containing train/val splits
    image_size: int = 512
    # Checkpoints -----------------------------------------------------------
    text_model_name: str = "gpt2-xl"
    diffusion_model_name: str = "stabilityai/stable-diffusion-2-1"
    clip_model_name: str = "openai/clip-vit-large-patch14"
    # Training --------------------------------------------------------------
    batch_size: int = 1
    total_steps: int = 150_000
    warmup_steps: int = 2_000
    lr: float = 2e-4
    weight_decay: float = 1e-2
    beta_kl: float = 0.1  # ELBO KL weight
    # Latent storyboard -----------------------------------------------------
    codebook_size: int = 4096
    latent_len: int = 8
    # Misc ------------------------------------------------------------------
    log_every: int = 100
    ckpt_every: int = 5_000
    output_dir: str = "checkpoints/story_gen"


cfg = Config()

# ---------------------------------------------------------------------------
# 2.  Dataset ----------------------------------------------------------------
# ---------------------------------------------------------------------------

class StoryImageDataset(Dataset):
    """Reads *.jsonl describing (prompt, story, image_path)."""

    def __init__(self, split: str, tokenizer: AutoTokenizer, image_processor: CLIPProcessor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        jsonl_path = Path(cfg.data_root) / split / "stories.jsonl"
        with open(jsonl_path, "r") as fh:
            self.records = [json.loads(l) for l in fh]
        self.split = split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt = rec["prompt"]
        story = rec["story"]
        image_fp = Path(cfg.data_root) / self.split / rec["image_path"]
        # --- text -----------------------------------------------------------
        text = f"{prompt} \n\n {story}"
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        # --- image ----------------------------------------------------------
        pil = Image.open(image_fp).convert("RGB")                       # <- open the JPEG
        image = self.image_processor(images=pil, return_tensors="pt")["pixel_values"][0]
        return {
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0],
            "pixels": image,
        }


# ---------------------------------------------------------------------------
# 3.  Vector‑Quantised VAE for storyboard tokens -----------------------------
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """EMA VQ layer (cf. van den Oord 2017)."""

    def __init__(self, codebook_size: int, dim: int, decay: float = 0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.decay = decay
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.codebook.data.clone())

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: (B, L, D)
        B, L, D = z_e.shape
        flat = z_e.view(-1, D)  # (B*L, D)
        # distances ---------------------------------------------------------
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )  # (B*L, K)
        idx = dist.argmin(1)  # (B*L)
        z_q = self.codebook[idx].view(B, L, D)
        # EMA updates -------------------------------------------------------
        if self.training:
            one_hot = F.one_hot(idx, self.codebook_size).type_as(flat)
            self.cluster_size.data.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
            dw = one_hot.T @ flat  # (K, D)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + 1e-6) / (n + self.codebook_size * 1e-6) * n
            )
            self.codebook.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
        # commitment loss ---------------------------------------------------
        loss = F.mse_loss(z_q.detach(), z_e) + 0.25 * F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()  # straight‑through estimator
        return z_q, idx.view(B, L), loss


class StoryEncoder(nn.Module):
    """Lightweight Transformer that encodes full story into latent vectors."""

    def __init__(self, hidden_size: int = 768, n_layers: int = 6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=12, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_latent = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_emb: torch.Tensor, attn_mask: torch.Tensor):
        h = self.encoder(x_emb, src_key_padding_mask=~attn_mask.bool())
        return self.to_latent(h)  # (B, L, D)


class VQVAEStoryboard(nn.Module):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tok = tokenizer
        self.embed = nn.Embedding(self.tok.vocab_size, 768)
        self.encoder = StoryEncoder(768)
        self.vq = VectorQuantizer(cfg.codebook_size, 768)
        # simple decoder for reconstruction loss (optional)
        self.decoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, self.tok.vocab_size),
        )

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        x = self.embed(input_ids)
        z_e = self.encoder(x, attn_mask)
        z_q, idx, vq_loss = self.vq(z_e)
        logits = self.decoder(z_q)
        PAD = self.tok.pad_token_id
        recon_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=PAD,
        )
        return idx, vq_loss + recon_loss


# ---------------------------------------------------------------------------
# 4.  Latent prior over storyboard tokens ------------------------------------
# ---------------------------------------------------------------------------

class PriorTransformer(nn.Module):
    def __init__(self, codebook_size: int, latent_len: int, d_model: int = 512):
        super().__init__()
        self.embed = nn.Embedding(codebook_size, d_model)
        self.pos = nn.Parameter(torch.randn(latent_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.to_logits = nn.Linear(d_model, codebook_size)

    def forward(self, idx: torch.Tensor):
        # idx: (B, L) codes: next‑token prediction
        x = self.embed(idx) + self.pos[None, : idx.size(1), :]
        tgt = x[:, :-1, :]
        memory = torch.zeros_like(tgt)  # dummy, unconditioned
        h = self.decoder(tgt, memory)
        return self.to_logits(h)  # (B, L‑1, K)


# ---------------------------------------------------------------------------
# 5.  Main training harness --------------------------------------------------
# ---------------------------------------------------------------------------

def pool_codes_to_len(codes, target_len):
    """
    Down‑sample a variable‑length code sequence (B, L) to length `target_len`
    by *chunking then taking the mode* of each chunk.
    Works for any L ≥ 1.
    """
    B, L = codes.shape
    # figure out chunk boundaries
    step = math.ceil(L / target_len)              # size of each chunk
    padded_len = step * target_len
    pad_size = padded_len - L
    if pad_size:
        pad = codes.new_full((B, pad_size), codes[0, 0].item())  # dummy pad
        codes = torch.cat([codes, pad], dim=1)    # (B, padded_len)

    # reshape into chunks and take majority vote (mode)
    chunks = codes.view(B, target_len, -1)        # (B, 8, step)
    pooled = chunks.mode(dim=2).values            # (B, 8)

    return pooled

@torch.no_grad()   # <- freeze UNet, VAE
def diffusion_nll(pipeline, images, prompt_text):
    """
    Monte‑Carlo ε‑prediction loss for a batch of ground‑truth RGB images.

    returns: tensor[batch]  (already on the same device as images)
    """
    device = images.device
    # 1. encode RGB -> latent z0
    vae = pipeline.vae
    z0  = vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215  # SD scaling

    # 2. pick random timestep for each sample
    scheduler = pipeline.scheduler
    bsz       = z0.size(0)
    timesteps = torch.randint(
        0, scheduler.num_train_timesteps, (bsz,), device=device
    ).long()

    # 3. add noise
    noise = torch.randn_like(z0)
    zt    = scheduler.add_noise(z0, noise, timesteps)

    # 4. tokenize prompt once (works because UNet weights are frozen)
    tok = pipeline.tokenizer(
        prompt_text,
        padding="max_length",
        truncation=True,                        
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)

    
    c = pipeline.text_encoder(**tok).last_hidden_state

    # 5. predict noise
    eps_hat = pipeline.unet(zt, timesteps, c).sample

    # 6. per‑sample MSE (*num_latent_channels*)
    loss = (eps_hat - noise).pow(2).mean(dim=(1,2,3))
    return loss


class StoryGenSystem(nn.Module):
    def __init__(self):
        super().__init__()
        # --- load huge frozen models (text & img) --------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        PAD_ID = self.tokenizer.pad_token_id    # now an int
        self.pad_id = PAD_ID
        self.text_model = AutoModelForCausalLM.from_pretrained(cfg.text_model_name)
        pipe_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.diffusion = StableDiffusionPipeline.from_pretrained(
            cfg.diffusion_model_name,
            torch_dtype=pipe_dtype,
        )

        self.text_model.requires_grad_(False).to("cpu").eval()

        for p in self.diffusion.unet.parameters():
            p.requires_grad_(False)
        for p in self.diffusion.vae.parameters():
            p.requires_grad_(False)
        for p in self.diffusion.text_encoder.parameters():
            p.requires_grad_(False)
        
        self.diffusion.to("cpu", dtype=torch.float32)
        
        # --- CLIP alignment --------------------------------------------------
        self.clip = CLIPModel.from_pretrained(cfg.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)
        # --- storyboard modules ---------------------------------------------
        self.vqvae = VQVAEStoryboard(self.tokenizer)
        self.prior = PriorTransformer(cfg.codebook_size, cfg.latent_len)
        self.trainable = nn.ModuleDict(dict(vqvae=self.vqvae, prior=self.prior))

    def elbo_step(self, batch: Dict[str, torch.Tensor]):
        # === Encode story into storyboard tokens ---------------------------
        codes, vq_loss = self.vqvae(batch["input_ids"], batch["attention_mask"])
        codes = pool_codes_to_len(codes, cfg.latent_len)
        codes_str = [" ".join(map(str, seq)) for seq in codes.cpu().tolist()]
        inputs = self.tokenizer(
            codes_str,                       # now List[str]
            return_tensors="pt",
            padding=True,
        ).to(batch["input_ids"].device)
        # Teacher‑forcing LM likelihood -------------------------------------
        with torch.no_grad():
            lm_out = self.text_model(
                input_ids=batch["input_ids"].to("cpu"),
                attention_mask=batch["attention_mask"].to("cpu"),
                labels=batch["input_ids"].to("cpu"),
            )
        nll_story = lm_out.loss.to(codes.device)   # move scalar back to GPU/accel device
        # Prior KL (codes→prior) -------------------------------------------
        logits_prior = self.prior(codes)
        prior_nll = F.cross_entropy(
            logits_prior.view(-1, cfg.codebook_size),
            codes[:, 1:].contiguous().view(-1),
        )

        image_term = diffusion_nll(
            self.diffusion,
            batch["pixels"].to(self.diffusion.device),   # RGB in [0,1]
            prompt_text=[self.tokenizer.decode(ids) for ids in batch["input_ids"]]
        ).mean()
        elbo = nll_story + image_term + vq_loss + cfg.beta_kl * prior_nll
        return elbo


# ---------------------------------------------------------------------------
# 6.  Training script --------------------------------------------------------
# ---------------------------------------------------------------------------
def story_collate(batch, tokenizer):
    # collect variable‑length tensors → list
    ids  = [item["input_ids"]      for item in batch]
    mask = [item["attention_mask"] for item in batch]

    # pad to longest in the batch
    PAD = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = pad_sequence(ids, batch_first=True, padding_value=PAD)
    attention_mask = pad_sequence(mask, batch_first=True, padding_value=0)

    # stack images (all same H×W already)
    pixels = torch.stack([item["pixels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixels": pixels,
    }

def train():
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with=["tensorboard"],
        project_dir=cfg.output_dir,
    )
    system = StoryGenSystem()
    tokenizer = system.tokenizer
    train_ds = StoryImageDataset("train", tokenizer, system.clip_processor)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,                # safer on macOS + MPS
        collate_fn=lambda x: story_collate(x, tokenizer),
    )

    optim = torch.optim.AdamW(
        system.trainable.parameters(),    # only ~55 M params
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    lr_sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.total_steps
    )

    trainable, optim, train_loader, lr_sched = accelerator.prepare(
        system.trainable, optim, train_loader, lr_sched
    )

    system.vqvae = trainable["vqvae"]
    system.prior = trainable["prior"]

    system.train()
    for step, batch in tqdm(enumerate(train_loader)):
        loss = system.elbo_step(batch)
        accelerator.backward(loss)
        optim.step()
        lr_sched.step()
        optim.zero_grad()

        if step % cfg.log_every == 0 and accelerator.is_main_process:
            accelerator.print(f"step {step}: loss = {loss.item():.4f}")

        if step % cfg.ckpt_every == 0 and step > 0 and accelerator.is_main_process:
            save_path = Path(cfg.output_dir) / f"ckpt_{step}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(save_path)

        if step >= cfg.total_steps:
            break


# ---------------------------------------------------------------------------
# 7.  Inference (generation) -------------------------------------------------
# ---------------------------------------------------------------------------

def generate(prompt, ckpt_path, seed=42):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. build system (loads SD in fp32 on CPU if no CUDA)
    system = StoryGenSystem()
    system.trainable.to(device)   # only the small trainable parts move

    # 2. load weights saved by accelerate
    if ckpt_path is not None:
        Accelerator().load_state(ckpt_path, load_model=True, model=system.trainable)

    system.eval()

    # 1. Sample storyboard codes from prior -------------------------------
    prompt_ids = system.tokenizer(prompt, return_tensors="pt")
    z = torch.zeros((1, cfg.latent_len), dtype=torch.long, device=device)
    for t in range(cfg.latent_len):
        logits = system.prior(z)[:, t - 1, :] if t > 0 else torch.zeros(1, cfg.codebook_size, device=device)
        probs = logits.softmax(-1)
        z[:, t] = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # 2. Generate story ----------------------------------------------------
    STYLE_PREFIX = (
        "Write a vivid science‑fiction short story for kids. "
        "Focus on wonder, sensory details, and positive emotion.\n\n"
    )
    story_prompt = STYLE_PREFIX + prompt
    gen_ids = system.text_model.generate(
        **system.tokenizer(story_prompt, return_tensors="pt"),
        max_length=512,
        do_sample=True,
        top_p=0.95,
        temperature=1.1,
        repetition_penalty=1.05,
    )
    story = system.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # 3. Generate image via diffusion -------------------------------------
    sd = system.diffusion.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        image = sd(prompt=story, num_inference_steps=50).images[0]
    image.save("generated.png")
    return story, "generated.png"


# ---------------------------------------------------------------------------
# 8.  Entry‑point ------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    gen = sub.add_parser("generate")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    if args.cmd == "train":
        train()
    else:
        story, fp = generate(args.prompt, args.ckpt)
        print("Story:\n", story)
        print(f"Image saved to {fp}")
