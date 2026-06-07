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

- Config temporal local, eliminada en la depuracion final:
  `runs/probe_configs/resnet18_batch128_epoch1.yaml`.
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

## 2026-06-06 - Diagnostico baseline ROCm/DataLoader

- Config: `configs/perf_resnet18_rocm.yaml`.
- Variante: `baseline` FP32, `cudnn.benchmark=True`, sin MIOpen search ni AMP.
- Dataset efectivo: 10,000 train y 2,000 val.
- Tiempo usado para comparar: promedio de epocas 2-3.

| Batch | Workers | Seg/epoca | Train img/s | Epoch img/s | GPU avg % | VRAM max MB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0 | 108.76 | 99.51 | 110.33 | 90.07 | 2132.71 |
| 32 | 2 | 93.13 | 114.70 | 128.85 | 96.03 | 2132.61 |
| 32 | 4 | 93.04 | 114.85 | 128.98 | 96.24 | 2132.60 |
| 32 | 6 | 93.26 | 114.61 | 128.67 | 96.55 | 2132.60 |
| 32 | 8 | 93.40 | 114.48 | 128.48 | 95.61 | 2132.63 |
| 64 | 0 | 109.02 | 99.30 | 110.07 | 87.86 | 3086.70 |
| 64 | 2 | 93.95 | 113.92 | 127.73 | 96.17 | 3086.59 |
| 64 | 4 | 94.12 | 113.72 | 127.49 | 96.46 | 3127.68 |
| 64 | 6 | 94.10 | 113.81 | 127.52 | 96.83 | 3115.19 |
| 64 | 8 | 94.28 | 113.63 | 127.28 | 96.02 | 3115.18 |
| 128 | 0 | 108.55 | 99.74 | 110.55 | 91.37 | 5075.21 |
| 128 | 2 | 96.69 | 110.73 | 124.10 | 96.57 | 5075.88 |
| 128 | 4 | 96.73 | 110.76 | 124.06 | 96.71 | 5075.89 |
| 128 | 6 | 96.65 | 111.04 | 124.16 | 96.53 | 5075.94 |
| 128 | 8 | 97.18 | 110.35 | 123.48 | 95.95 | 5075.90 |

Observaciones:

- Mejor baseline: batch 32, `num_workers=4` con 93.04 s/epoca y 128.98
  img/s sobre train+val.
- `num_workers=2/4/6` son practicamente equivalentes; `num_workers=0`
  penaliza fuerte el pipeline.
- Batch 64 queda muy cerca de batch 32, pero usa mas VRAM y no mejora tiempo.
- Batch 128 aumenta VRAM a ~5 GB y empeora throughput frente a batch 32/64.

## 2026-06-06 - Diagnostico MIOpen search

- Config: `configs/perf_resnet18_rocm.yaml`.
- Variante: `miopen_search` con `MIOPEN_FIND_MODE=NORMAL` y
  `MIOPEN_FIND_ENFORCE=SEARCH`, FP32, sin AMP.
- Dataset efectivo: 10,000 train y 2,000 val.
- Tiempo usado para comparar: promedio de epocas 2-3.

| Variante | Batch | Workers | Seg/epoca | Train img/s | Epoch img/s | GPU avg % | VRAM max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 32 | 4 | 93.04 | 114.85 | 128.98 | 96.24 | 2132.60 |
| miopen_search | 32 | 6 | 93.17 | 114.70 | 128.79 | 96.94 | 2210.91 |
| miopen_search | 32 | 2 | 93.22 | 114.61 | 128.73 | 97.63 | 2210.91 |
| miopen_search | 32 | 4 | 93.25 | 114.58 | 128.68 | 97.56 | 2210.91 |
| miopen_search | 64 | 2 | 94.01 | 113.86 | 127.64 | 96.95 | 3164.89 |
| miopen_search | 128 | 6 | 96.60 | 111.00 | 124.23 | 97.10 | 5145.36 |

Observacion: `miopen_search` no mejora el mejor baseline; la diferencia entre
93.04 s y 93.17 s esta dentro de ruido operativo, con algo mas de VRAM. No
conviene activarlo como default con estos datos.

## Decision Fase 4.5

- Se detiene el diagnostico antes de AMP FP16/MIOpen+AMP FP16 porque los
  resultados baseline vs MIOpen ya no justifican mas trials costosos.
- Config elegida para continuar Fase 4:
  `baseline` FP32, sin variables MIOpen, sin AMP, batch 32,
  `num_workers=4`, `pin_memory=true`, `persistent_workers=true`,
  `prefetch_factor=2` y `experiment.deterministic=false`.
- Motivo: mejor tiempo observado en epocas 2-3 fue baseline batch 32 workers 4
  con 93.04 s/epoca y 128.98 img/s; MIOpen quedo practicamente igual o peor.

## 2026-06-06 - Tuning ResNet18 dos etapas

- Config: `configs/tuning_resnet18.yaml`.
- Modelo: ResNet18, pesos ImageNet.
- Protocolo por trial: `classifier_only` 3 epocas con `lr=learning_rate`,
  luego `partial_finetuning` 5 epocas con `lr=learning_rate*0.1`.
- En `partial_finetuning` se descongela `layer4` + `fc`.
- Dataset efectivo: 10,000 train y 2,000 val con subset estratificado
  aleatorio reproducible.
- Tuning sin uso de test.
- Seleccion: menor `val_loss` en la etapa final.

| Trial | LR base | LR parcial | Weight decay | Mejor epoca | Val loss | Val acc | Val F1 macro |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.001 | 0.0001 | 0.0001 | 6 | 0.0604 | 0.9825 | 0.9826 |
| 2 | 0.001 | 0.0001 | 0.00001 | 6 | 0.0604 | 0.9825 | 0.9826 |
| 3 | 0.0003 | 0.00003 | 0.0001 | 8 | 0.0670 | 0.9815 | 0.9816 |
| 4 | 0.0003 | 0.00003 | 0.00001 | 8 | 0.0670 | 0.9820 | 0.9821 |
| 5 | 0.0001 | 0.00001 | 0.0001 | 8 | 0.1110 | 0.9750 | 0.9751 |
| 6 | 0.0001 | 0.00001 | 0.00001 | 8 | 0.1110 | 0.9750 | 0.9751 |

Resultado elegido:

- Mejor trial por `val_loss`: trial 1.
- Hiperparametros seleccionados para entrenamiento final:
  `learning_rate=0.001`, `partial_finetuning lr=0.0001`,
  `weight_decay=0.0001`, AdamW, ReduceLROnPlateau.
- Trial 1 y trial 2 quedaron practicamente empatados; se conserva trial 1 por
  tener el menor `val_loss` y mayor regularizacion por weight decay.

## 2026-06-07 - Entrenamiento final ResNet18 dos etapas

- Config: `configs/resnet18_finetuning.yaml`.
- Modelo: ResNet18, pesos ImageNet.
- Dataset efectivo: 231,900 train y 30,000 val.
- Protocolo: `classifier_only` 5 epocas con `lr=0.001`, luego
  `partial_finetuning` hasta 25 epocas con `lr=0.0001`.
- En `partial_finetuning` se descongela `layer4` + `fc`.
- Optimizador: AdamW. Scheduler: ReduceLROnPlateau.
- Early stopping: `val_loss`, paciencia 7.

Resultado:

| Mejor epoca | Final epoca | Val loss | Val acc | Val F1 macro | Early stopping |
| ---: | ---: | ---: | ---: | ---: | --- |
| 15 | 22 | 0.0103 | 0.9975 | 0.9975 | si, en `partial_finetuning` |

Observaciones:

- El mejor checkpoint quedo en `checkpoints/resnet18_two_stage/best_model.pth`.
- La validacion final en epoca 22 mantuvo `val_accuracy=0.9975` y
  `val_f1_macro=0.9975`, con `val_loss=0.0116`.
- Falta evaluar este checkpoint sobre `data/test`; las metricas anteriores son
  de validacion, no de test.

## 2026-06-07 - Entrenamiento final MobileNetV3 Small dos etapas

- Config: `configs/mobilenetv3_finetuning.yaml`.
- Modelo: MobileNetV3 Small, pesos ImageNet.
- Dataset efectivo: 231,900 train y 30,000 val.
- Protocolo: `classifier_only` 5 epocas con `lr=0.001`, luego
  `partial_finetuning` hasta 25 epocas con `lr=0.0001`.
- En `partial_finetuning` se descongela `classifier` + ultimos 3 bloques de
  `features`.
- Hiperparametros reutilizados del tuning ResNet18; no se hizo tuning separado
  para MobileNetV3 Small.
- Optimizador: AdamW. Scheduler: ReduceLROnPlateau.
- Early stopping: `val_loss`, paciencia 7.

Resultado:

| Mejor epoca | Final epoca | Val loss | Val acc | Val F1 macro | Early stopping |
| ---: | ---: | ---: | ---: | ---: | --- |
| 12 | 19 | 0.0156 | 0.9956 | 0.9956 | si, en `partial_finetuning` |

Observaciones:

- El mejor checkpoint quedo en
  `checkpoints/mobilenetv3_two_stage/best_model.pth`.
- La validacion final en epoca 19 tuvo `val_accuracy=0.9961` y
  `val_f1_macro=0.9961`, con `val_loss=0.0162`.
- Falta evaluar este checkpoint sobre `data/test`; las metricas anteriores son
  de validacion, no de test.

## 2026-06-07 - Evaluacion final sobre test

- Dataset: `data/test`.
- ResNet18 checkpoint:
  `checkpoints/resnet18_two_stage/best_model.pth`.
- MobileNetV3 Small checkpoint:
  `checkpoints/mobilenetv3_two_stage/best_model.pth`.

| Modelo | Test loss | Test acc | Test F1 macro | Parametros | Inferencia ms/img |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 0.0139 | 0.9971 | 0.9971 | 11,181,642 | 2.8845 |
| MobileNetV3 Small | 0.0215 | 0.9945 | 0.9945 | 1,528,106 | 0.4298 |

Observaciones:

- ResNet18 obtuvo la mejor exactitud y F1 macro en test.
- MobileNetV3 Small tuvo menor rendimiento, pero fue aproximadamente 6.7 veces
  mas rapido por imagen en inferencia y usa cerca de 13.7% de los parametros de
  ResNet18.
- Salidas ResNet18:
  `runs/resnet18_two_stage/test_evaluation/metrics.json`,
  `confusion_matrix.csv`, `confusion_matrix.png`, `per_class_metrics.csv`.
- Salidas MobileNetV3 Small:
  `runs/mobilenetv3_two_stage/test_evaluation/metrics.json`,
  `confusion_matrix.csv`, `confusion_matrix.png`, `per_class_metrics.csv`.
