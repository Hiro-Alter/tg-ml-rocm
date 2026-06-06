# Experiments

## 2026-06-06 - Tuning parcial ResNet18 preliminar

- Config: `configs/tuning_resnet18.yaml`.
- Modelo: ResNet18, `full_finetuning`, pesos ImageNet.
- Dataset efectivo por trial: 10,000 train y 2,000 val
  (`max_train_samples_per_class=1000`, `max_val_samples_per_class=200`).
- Optimizador: AdamW. Scheduler: ReduceLROnPlateau. Monitor: `val_loss`.

| Trial | LR | Weight decay | Mejor epoca | Val loss | Val acc | Val F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.001 | 0.0001 | 7 | 0.0278 | 0.9910 | 0.9910 |
| 2 | 0.001 | 0.00001 | 6 | 0.0258 | 0.9925 | 0.9925 |

Observaciones:

- Estos resultados quedan como preliminares historicos: el subset fue
  deterministico por orden de archivos, no muestreado aleatoriamente.
- Trial 3 (`lr=0.0003`, `weight_decay=0.0001`) fue interrumpido manualmente
  durante la ejecucion; no tiene `metrics.json` final.
- Rendimiento observado con `batch_size=32`: ~94-95 s/epoca para 12,000
  imagenes efectivas, ~126 img/s.
- `rocm-smi` durante entrenamiento reporto GPU use 100%, sclk ~3200 MHz,
  potencia 133-163 W y VRAM ~13%.

## Protocolo corregido para siguiente tuning

- Subset estratificado aleatorio por clase con `experiment.seed=42`.
- `max_train_samples_per_class=1000` y `max_val_samples_per_class=200`.
- `batch_size=32` se mantiene para tuning por mejor throughput observado.
- Los trials del protocolo corregido deben reemplazar los resultados
  preliminares para seleccion de hiperparametros.
- Usar `--rerun-existing` o una nueva carpeta de salida para no saltar los
  trials 1-2 que ya tienen `metrics.json` del protocolo preliminar.

## 2026-06-06 - Probe batch size ResNet18

- Config temporal: `runs/probe_configs/resnet18_batch128_epoch1.yaml`.
- Modelo: ResNet18, `full_finetuning`, 1 epoca, `lr=0.001`,
  `weight_decay=0.0001`.
- Dataset efectivo: 10,000 train y 2,000 val con subset estratificado
  aleatorio reproducible.

| Batch size | Segundos/epoca | Img/s sobre 12,000 imagenes | Val loss | Val acc | Val F1 macro |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 96.16 | 124.79 | 0.1501 | 0.9555 | 0.9556 |
| 64 | 139.97 | 85.74 | 0.1225 | 0.9585 | 0.9581 |
| 128 | 198.09 | 60.58 | 0.0716 | 0.9770 | 0.9770 |

Observacion: batch 32 mantiene el mejor throughput. Las metricas de una sola
epoca no se usan para seleccion final de modelo; este probe solo compara
rendimiento de ejecucion.
