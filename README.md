# Fine-tuning de Stable Diffusion v1.4 con LoRA: ilustraciones de libros antiguos

Práctica del Máster de Inteligencia Artificial de PontIA.

## Objetivo

Adaptar Stable Diffusion v1.4 al estilo de las ilustraciones de libros antiguos mediante LoRA (Low-Rank Adaptation), entrenando únicamente una pequeña fracción de los parámetros del modelo y comparando la generación antes y después del entrenamiento.

## Los dos entrenamientos

Este proyecto contiene **dos entrenamientos distintos**. Conviene no confundirlos:

| | 1. Entrenamiento principal | 2. Entrenamiento exprés |
|---|---|---|
| Código | Notebook `notebooks/finetuning_oldbook_lora.ipynb` | Script `scripts/entrenamiento_rapido_local.py` |
| Imágenes | 800 | 100 |
| Épocas | 5 | 1 |
| Estado | Completado | Completado (realizado posteriormente) |
| Imágenes comparativas antes/después | Sí, en el notebook | No |
| Pesos LoRA | **No se pudieron conservar** | **Disponibles** en Hugging Face |
| Motivo | Los pesos se perdieron durante la exportación | Se hizo para obtener pesos LoRA funcionales y publicarlos |

> **Los pesos publicados en Hugging Face ([`ni7in/oldbook-lora-sd14`](https://huggingface.co/ni7in/oldbook-lora-sd14)) corresponden al entrenamiento exprés (100 imágenes, 1 época), NO al notebook del entrenamiento principal.**

## Modelo base

- [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4), resolución 512 × 512.
- Componentes: tokenizador y text encoder CLIP, VAE, UNet (~860 M de parámetros) y scheduler DDPM.
- VAE y text encoder permanecen congelados. Solo se entrenan las matrices LoRA de la UNet.

## Dataset

- [`gigant/oldbookillustrations`](https://huggingface.co/datasets/gigant/oldbookillustrations), split `train` (4154 imágenes).
- Columnas utilizadas: `1600px` (imagen) e `info_alt` (descripción).
- Se descartan las filas con descripción vacía o nula.
- Las imágenes no son cuadradas: se reescalan a 512 px en el lado corto (`Resize`) y se recortan al centro (`CenterCrop`), manteniendo las proporciones.
- Ambos entrenamientos usan el mismo procedimiento de selección: filtrado de descripciones, barajado con semilla 42 y primeras *N* imágenes (800 y 100 respectivamente).

## LoRA

Configuración común a los dos entrenamientos:

| Parámetro | Valor |
|---|---|
| Rango (`r`) / `lora_alpha` | 6 / 6 |
| Inicialización | `gaussian` |
| Módulos objetivo | `to_q`, `to_k`, `to_v`, `to_out.0` (atención de la UNet) |
| Nombre del adaptador | `oldbook` |
| Learning rate | 1e-4 (AdamW) |
| Parámetros entrenables | 1 195 776 de 860 716 740 (0,14 %) |

## 1. Entrenamiento principal (notebook, 800 imágenes, 5 épocas)

Notebook: [`notebooks/finetuning_oldbook_lora.ipynb`](notebooks/finetuning_oldbook_lora.ipynb), adaptado del notebook del profesor.

### Configuración

| Parámetro | Valor |
|---|---|
| Imágenes | 800 (selección aleatoria, semilla 42) |
| Épocas | 5 |
| Tamaño de lote | 3 (267 lotes por época) |
| Hardware | GPU T4 de Google Colab |
| Duración | ~18,5 min por época |

### Metodología

1. Generación con el modelo base para fijar el comportamiento de partida (4 prompts, semilla 42).
2. Carga y filtrado del dataset, y transformación de las imágenes.
3. Carga por separado de tokenizador, scheduler, text encoder, VAE y UNet.
4. Inyección de la LoRA en la UNet y optimizador AdamW solo sobre sus parámetros.
5. Bucle de entrenamiento: codificación de la imagen con el VAE, adición de ruido en un timestep aleatorio, predicción del ruido con la UNet y pérdida MSE.
6. Generación con el modelo base + LoRA, con los mismos prompts y la misma semilla, para comparar.
7. Exportación del adaptador entrenado.

Todos los parámetros se modifican en una única celda del notebook.

### Resultados

Pérdida (MSE) del último lote de cada época. Es un valor puntual y ruidoso, solo orientativo:

| Época | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Pérdida | 0,235 | 0,0618 | 0,117 | 0,117 | 0,0246 |

Las imágenes comparativas entre el modelo base y el modelo con LoRA están en las secciones 1 y 6 del notebook.

**Pesos:** el entrenamiento terminó, pero los pesos de la LoRA se perdieron durante la exportación y no están disponibles. Solo se conservan las salidas del notebook (pérdidas e imágenes generadas). Estos pesos **no** están publicados en ningún sitio.

## 2. Entrenamiento exprés (script, 100 imágenes, 1 época)

Script: [`scripts/entrenamiento_rapido_local.py`](scripts/entrenamiento_rapido_local.py). Se realizó **después** del entrenamiento principal, con el único fin de obtener pesos LoRA funcionales y publicarlos en Hugging Face.

### Configuración

| Parámetro | Valor |
|---|---|
| Imágenes | 100 (misma selección aleatoria, semilla 42) |
| Épocas | 1 (100 pasos) |
| Tamaño de lote | 1 |
| Resolución | 512 × 512 |
| Hardware | GPU RTX 2060 de 6 GB (WSL) |
| Optimizaciones de memoria | UNet, VAE y text encoder en FP16 (LoRA en FP32 con `GradScaler`), gradient checkpointing |
| Duración | 45 s de entrenamiento (+18 s de precálculo de latentes) |
| VRAM pico | 2,03 GB |

Diferencias respecto al notebook, por motivos de memoria y rapidez:

- No se genera ninguna imagen antes del entrenamiento.
- Con 1 época cada imagen se usa una sola vez, así que latentes y embeddings de texto se calculan de antemano, una sola vez. Esto permite liberar el VAE y el text encoder de la GPU antes de entrenar la UNet.
- Se usa PyTorch directamente (`autocast` y `GradScaler`) en lugar de `Accelerator`.

### Resultados

Pérdida (MSE) del lote de cada décimo paso. Son valores puntuales y muy ruidosos, no indican convergencia:

| Paso | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Pérdida | 0,0116 | 0,0757 | 0,1047 | 0,1690 | 0,0518 | 0,1243 | 0,0450 | 0,4267 | 0,1313 | 0,1679 |

Verificación de los pesos guardados antes de publicarlos:

- `pytorch_lora_weights.safetensors` (4,8 MB) y `adapter_config.json` se leen correctamente; los 1 195 776 parámetros son finitos y las 128 matrices B no son nulas (parten de cero, así que se han entrenado).
- La LoRA se carga sin avisos sobre Stable Diffusion v1.4 y cambia la salida de la UNet respecto al modelo base.
- Se generó una única imagen de prueba (10 pasos, semilla 42) con la LoRA activa. Es una comprobación de funcionamiento; no hay evaluación cuantitativa ni comparación visual sistemática con el modelo base.

El entrenamiento es muy corto, por lo que el efecto sobre el estilo es limitado.

## Hugging Face

Repositorio público: **https://huggingface.co/ni7in/oldbook-lora-sd14**

- Contiene los pesos LoRA (`pytorch_lora_weights.safetensors`), su configuración (`adapter_config.json`) y una descripción breve del entrenamiento.
- **Corresponden al entrenamiento exprés (100 imágenes, 1 época), no al notebook del entrenamiento principal (800 imágenes, 5 épocas), cuyos pesos no están disponibles.**

Uso:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("ni7in/oldbook-lora-sd14")

image = pipe("an old book illustration of a sheep in a meadow", num_inference_steps=30).images[0]
```

## Estructura del proyecto

```
.
├── README.md
├── .gitignore
├── notebooks/
│   ├── 2.finetuning_stable_diffusion_lora.ipynb   # notebook original del profesor (dataset de Pokémon)
│   └── finetuning_oldbook_lora.ipynb              # entrenamiento principal (800 imágenes, 5 épocas)
└── scripts/
    └── entrenamiento_rapido_local.py              # entrenamiento exprés (100 imágenes, 1 época)
```

## Ejecución

### Entrenamiento principal en Google Colab

1. Abre [Google Colab](https://colab.research.google.com/) y sube `notebooks/finetuning_oldbook_lora.ipynb` desde *Archivo → Subir cuaderno*.
2. Selecciona una GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución* (T4 o superior).
3. Ajusta, si hace falta, la celda de parámetros.
4. Ejecuta las celdas en orden. La primera instala `diffusers`, `datasets`, `accelerate` y `peft`, y desinstala `torchao` por incompatibilidad con `peft`.
5. La descarga del dataset ocupa unos 5 GB.

Opcionalmente, configura un `HF_TOKEN` para evitar el aviso de peticiones sin autenticar a Hugging Face.

### Entrenamiento exprés en local

Requiere una GPU NVIDIA con CUDA (probado en una RTX 2060 de 6 GB).

```bash
pip install torch diffusers transformers accelerate peft datasets huggingface_hub
python scripts/entrenamiento_rapido_local.py           # entrena y guarda la LoRA en outputs/oldbook_lora_rapida/
python scripts/entrenamiento_rapido_local.py --subir   # sube la LoRA guardada a Hugging Face (requiere sesión iniciada)
```

Los parámetros se modifican en la cabecera del script. Con `--subir`, el repositorio se crea como privado salvo que se cambie `hf_repo_private`.
