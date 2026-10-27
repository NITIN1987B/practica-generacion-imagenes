# Fine-tuning de Stable Diffusion v1.4 con LoRA: ilustraciones de libros antiguos

Práctica del Máster de Inteligencia Artificial de PontIA.

## Objetivo

Adaptar Stable Diffusion v1.4 al estilo de las ilustraciones de libros antiguos mediante LoRA (Low-Rank Adaptation), entrenando únicamente una pequeña fracción de los parámetros del modelo y comparando la generación antes y después del entrenamiento.

## Modelo base

- [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4), resolución 512 × 512.
- Componentes: tokenizador y text encoder CLIP, VAE, UNet (~860 M de parámetros) y scheduler DDPM.
- VAE y text encoder permanecen congelados. Solo se entrenan las matrices LoRA de la UNet.

## Dataset

- [`gigant/oldbookillustrations`](https://huggingface.co/datasets/gigant/oldbookillustrations), split `train` (4154 imágenes).
- Columnas utilizadas: `1600px` (imagen) e `info_alt` (descripción).
- Se descartan las filas con descripción vacía o nula.
- Las imágenes no son cuadradas: se reescalan a 512 px en el lado corto (`Resize`) y se recortan al centro (`CenterCrop`), manteniendo las proporciones.

## LoRA

| Parámetro | Valor |
|---|---|
| Rango (`r`) / `lora_alpha` | 6 / 6 |
| Inicialización | `gaussian` |
| Módulos objetivo | `to_q`, `to_k`, `to_v`, `to_out.0` (atención de la UNet) |
| Nombre del adaptador | `oldbook` |
| Parámetros entrenables | 1 195 776 de 860 716 740 (0,14 %) |

## Metodología

1. Generación con el modelo base para fijar el comportamiento de partida (4 prompts, semilla 42).
2. Carga y filtrado del dataset, y transformación de las imágenes.
3. Carga por separado de tokenizador, scheduler, text encoder, VAE y UNet.
4. Inyección de la LoRA en la UNet y optimizador AdamW solo sobre sus parámetros.
5. Bucle de entrenamiento: codificación de la imagen con el VAE, adición de ruido en un timestep aleatorio, predicción del ruido con la UNet y pérdida MSE.
6. Generación con el modelo base + LoRA, con los mismos prompts y la misma semilla, para comparar.
7. Exportación del adaptador entrenado.

Todos los parámetros se modifican en una única celda del notebook.

## Entrenamientos

| | Entrenamiento original | Entrenamiento reducido |
|---|---|---|
| Estado | Completado | Pendiente de finalizar |
| Imágenes | 800 (selección aleatoria, semilla 42) | Por definir |
| Épocas | 5 | Por definir |
| Tamaño de lote | 3 (267 lotes por época) | Por definir |
| Learning rate | 1e-4 | Por definir |
| Hardware | GPU T4 de Google Colab | Por definir |
| Duración | ~18,5 min por época | Por definir |

## Resultados

### Entrenamiento original (800 imágenes, 5 épocas)

Pérdida (MSE) del último lote de cada época. Es un valor puntual y ruidoso, solo orientativo:

| Época | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Pérdida | 0,235 | 0,0618 | 0,117 | 0,117 | 0,0246 |

La comparación visual entre el modelo base y el modelo con LoRA está en las secciones 1 y 6 del notebook.

### Entrenamiento reducido

Pendiente de finalizar.

## Estructura del proyecto

```
.
├── README.md
├── .gitignore
└── notebooks/
    ├── 2.finetuning_stable_diffusion_lora.ipynb   # notebook original del profesor (dataset de Pokémon)
    └── finetuning_oldbook_lora.ipynb              # adaptación a gigant/oldbookillustrations
```

## Ejecución en Google Colab

1. Abre [Google Colab](https://colab.research.google.com/) y sube `notebooks/finetuning_oldbook_lora.ipynb` desde *Archivo → Subir cuaderno*.
2. Selecciona una GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución* (T4 o superior).
3. Ajusta, si hace falta, la celda de parámetros.
4. Ejecuta las celdas en orden. La primera instala `diffusers`, `datasets`, `accelerate` y `peft`, y desinstala `torchao` por incompatibilidad con `peft`.
5. La descarga del dataset ocupa unos 5 GB.
6. Las dos últimas celdas guardan el adaptador en `/content/oldbook_lora` y descargan un `.zip`.

Opcionalmente, configura un `HF_TOKEN` para evitar el aviso de peticiones sin autenticar a Hugging Face.

## Hugging Face

Enlace al adaptador LoRA: _pendiente de publicar_.
