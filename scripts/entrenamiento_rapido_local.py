"""Entrenamiento LoRA minimo de Stable Diffusion v1.4 en local (GPU de 6 GB).

Uso:
    python scripts/entrenamiento_rapido_local.py            # entrena y guarda la LoRA
    python scripts/entrenamiento_rapido_local.py --subir    # sube la LoRA guardada a Hugging Face
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import time

import torch

# ==========================================================
# PARAMETROS
# ==========================================================
pretrained_model_name = "CompVis/stable-diffusion-v1-4"

dataset_name = "gigant/oldbookillustrations"
dataset_split = "train"
image_column = "1600px"
caption_column = "info_alt"
num_images = 100

resolution = 512

adapter_name = "oldbook"
lora_rank = 6
lora_alpha = 6

learning_rate = 1e-4
num_epochs = 1
batch_size = 1

seed = 42

lora_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "oldbook_lora_rapida")

hf_repo_name = "oldbook-lora-sd14"
hf_repo_private = True

device = "cuda"
weight_dtype = torch.float16


def subir():
    from huggingface_hub import HfApi

    api = HfApi()
    hf_user = api.whoami()["name"]
    hf_repo_id = f"{hf_user}/{hf_repo_name}"
    print("Repositorio de destino:", hf_repo_id, "(privado)" if hf_repo_private else "(publico)")

    archivos = sorted(os.listdir(lora_dir))
    print("Archivos a subir:", archivos)
    assert archivos, f"{lora_dir} esta vacio"

    api.create_repo(repo_id=hf_repo_id, repo_type="model", private=hf_repo_private, exist_ok=True)
    api.upload_folder(
        folder_path=lora_dir,
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Añado LoRA oldbook (Stable Diffusion v1.4, entrenamiento rapido)",
    )
    print("LoRA subida a:", f"https://huggingface.co/{hf_repo_id}")


def entrenar():
    import torch.nn.functional as F
    from datasets import load_dataset
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.training_utils import cast_training_params
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import ImageFile
    from torchvision import transforms
    from transformers import CLIPTextModel, CLIPTokenizer

    # Permite a Pillow cargar imagenes incompletas:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    assert torch.cuda.is_available(), "No hay GPU CUDA disponible"
    torch.manual_seed(seed)
    print("GPU:", torch.cuda.get_device_name(0))
    libre, total = torch.cuda.mem_get_info()
    print(f"VRAM libre/total: {libre / 1e9:.2f} / {total / 1e9:.2f} GB")

    # ---------------------------------------------------------- dataset
    dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"Dataset original: {len(dataset)} imagenes")
    dataset = dataset.filter(
        lambda caption: isinstance(caption, str) and caption.strip() != "",
        input_columns=[caption_column],
    )
    dataset = dataset.shuffle(seed=seed).select(range(num_images))
    print(f"Dataset de entrenamiento: {len(dataset)} imagenes")

    image_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # ---------------------------------------------------------- latentes y embeddings (se calculan una vez)
    # Con 1 epoca cada imagen se usa una sola vez, asi que precalcular es equivalente a hacerlo en el bucle,
    # y permite liberar el VAE y el text encoder de la GPU antes de entrenar la UNet.
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name, subfolder="vae", torch_dtype=weight_dtype
    ).to(device).eval()
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    cache = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(dataset)):
            ejemplo = dataset[i]
            pixel_values = image_transforms(ejemplo[image_column].convert("RGB")).unsqueeze(0).to(device, weight_dtype)
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            ids = tokenizer(
                ejemplo[caption_column], padding="max_length", truncation=True,
                max_length=tokenizer.model_max_length, return_tensors="pt",
            ).input_ids.to(device)
            embeddings = text_encoder(ids)[0]
            cache.append((latents.cpu(), embeddings.cpu()))
    print(f"Latentes y embeddings listos ({len(cache)}) en {time.time() - t0:.0f} s")

    del vae, text_encoder, dataset
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------- UNet + LoRA
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name, subfolder="unet", torch_dtype=weight_dtype
    ).to(device)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config, adapter_name=adapter_name)
    cast_training_params(unet, dtype=torch.float32)  # la LoRA se entrena en fp32; la base queda en fp16

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"Parametros entrenables (LoRA): {n_lora:,} de {n_total:,} ({100 * n_lora / n_total:.2f}%)")

    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda")

    # ---------------------------------------------------------- entrenamiento
    unet.train()
    pasos = 0
    total_pasos = num_epochs * (len(cache) // batch_size)
    t0 = time.time()
    for epoch in range(num_epochs):
        orden = torch.randperm(len(cache)).tolist()
        for inicio in range(0, len(orden) - batch_size + 1, batch_size):
            lote = [cache[j] for j in orden[inicio:inicio + batch_size]]
            latents = torch.cat([x[0] for x in lote]).to(device, torch.float32)
            encoder_hidden_states = torch.cat([x[1] for x in lote]).to(device)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(weight_dtype)

            with torch.autocast("cuda", dtype=weight_dtype):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            pasos += 1
            if pasos % 10 == 0 or pasos == total_pasos:
                print(
                    f"Epoca {epoch} | paso {pasos}/{total_pasos} | loss {loss.item():.4f} | "
                    f"VRAM pico {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | {time.time() - t0:.0f} s",
                    flush=True,
                )
    print(f"Entrenamiento terminado en {time.time() - t0:.0f} s")

    # ---------------------------------------------------------- guardado (solo LoRA + configuracion)
    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet, adapter_name=adapter_name))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=lora_dir,
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
        unet_lora_adapter_metadata=unet.peft_config[adapter_name].to_dict(),
    )
    lora_config.save_pretrained(lora_dir)

    for nombre in sorted(os.listdir(lora_dir)):
        mb = os.path.getsize(os.path.join(lora_dir, nombre)) / 1e6
        print(f"{nombre}: {mb:.1f} MB")
        assert mb < 100, f"{nombre} es demasiado grande: no deberia contener la UNet completa"
    print("LoRA guardada en:", lora_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subir", action="store_true", help="sube a Hugging Face la LoRA ya guardada")
    args = parser.parse_args()
    if args.subir:
        subir()
    else:
        entrenar()
