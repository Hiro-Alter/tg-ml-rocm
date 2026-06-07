# Project State

## Fase actual

Fase 4 completada: ResNet18 y MobileNetV3 Small entrenados y evaluados sobre
test.

## Implementado

- Constantes inmutables de clases, tamano 224 e ImageNet mean/std.
- Factoria de modelos para ResNet18 y MobileNetV3 Small con tres modos:
  `classifier_only`, `partial_finetuning` y `full_finetuning`.
- `partial_finetuning` descongela `layer4` + `fc` en ResNet18, y
  `classifier` + los ultimos 3 bloques de `features` en MobileNetV3 Small.
- Transformaciones base para ImageFolder con salida de 3 canales y
  normalizacion ImageNet.
- Carga minima de configuraciones YAML.
- Configs iniciales para baseline, fine-tuning y tuning ResNet18.
- Script `scripts/check_rocm.py`.
- Script `scripts/train.py` para entrenamiento desde YAML.
- Guardado de `training_history.csv`, `metrics.json`, `best_model.pth` y
  `last_model.pth`.
- Early stopping por `val_loss`, `val_accuracy` o `val_f1_macro`.
- Schedulers `ReduceLROnPlateau` y `CosineAnnealingLR`.
- Script `scripts/evaluate.py` para metricas finales y matriz de confusion.
- Script `scripts/plot_history.py` para graficas desde `training_history.csv`.
- Script `scripts/export_model.py` para `.pth`, TorchScript y ONNX.
- Script `scripts/infer_image.py` para inferencia de una imagen.
- Script `scripts/tune.py` para grid limitado de hiperparametros delegando cada
  trial a `scripts/train.py`.
- `scripts/train.py` soporta limites opcionales estratificados aleatorios por
  clase usando `experiment.seed`:
  `data.max_samples_per_class`, `data.max_train_samples_per_class` y
  `data.max_val_samples_per_class`.
- `scripts/train.py` registra `train_seconds`, `val_seconds` y
  `epoch_seconds`, y permite configurar `data.pin_memory`,
  `data.persistent_workers`, `data.prefetch_factor` y
  `experiment.deterministic`.
- `scripts/train.py` soporta AMP opcional con
  `training.mixed_precision.enabled`, `dtype` y `grad_scaler`.
- `scripts/train.py` soporta `training.stages` para entrenar protocolos
  secuenciales sobre el mismo modelo.
- `scripts/tune.py` genera trials ResNet18 de dos etapas:
  `classifier_only` y luego `partial_finetuning`.
- PyTorch ROCm instalado en `.venv`:
  `torch==2.11.0+rocm7.2`, `torchvision==0.26.0+rocm7.2` y
  `torchaudio==2.11.0+rocm7.2`.
- Config diagnostica `configs/perf_resnet18_rocm.yaml`.
- Script `scripts/perf_diagnostics.py` para matriz corta de rendimiento con
  monitoreo CPU/RAM/GPU/VRAM/disco.
- Documento formal `PERFORMANCE_DIAGNOSTIC_PHASE_4_5.md` con descripcion,
  resultados y conclusion de la fase 4.5.
- Documento `FINAL_TRAINING_ANALYSIS.md` con analisis del entrenamiento final
  de ResNet18 y MobileNetV3 Small.
- Documento `FINAL_TEST_EVALUATION_ANALYSIS.md` con analisis de resultados
  finales sobre test y comparacion ResNet18 vs MobileNetV3 Small.

## Comandos funcionales

```bash
python scripts/check_rocm.py
python scripts/train.py --config configs/resnet18_finetuning.yaml
python scripts/train.py --config configs/mobilenetv3_finetuning.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data data/test
python scripts/plot_history.py --history runs/experiment/training_history.csv
python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format onnx
python scripts/infer_image.py --checkpoint checkpoints/best_model.pth --image example.jpg
python scripts/tune.py --config configs/tuning_resnet18.yaml --dry-run --max-trials 1
python scripts/tune.py --config configs/tuning_resnet18.yaml
python scripts/perf_diagnostics.py --config configs/perf_resnet18_rocm.yaml --dry-run
```

## Notas

- Se generaron symlinks locales `data/{train,val,test}/<clase>` desde
  `dataset/hagridv2_ROI_224_processed` usando
  `dataset/annotations_ROI_224`; `data/` sigue ignorado por Git.
- Tuning parcial ResNet18 completado para 2 trials sobre subset balanceado
  deterministico por orden de archivos:
  10,000 train / 2,000 val.
- Mejor trial parcial: `lr=0.001`, `weight_decay=0.00001`,
  `val_loss=0.0258`, `val_accuracy=0.9925`, `val_f1_macro=0.9925`.
- Trial 3 fue interrumpido manualmente; no tiene metricas finales.
- Resultados parciales marcados como preliminares; el protocolo corregido usa
  muestreo estratificado aleatorio reproducible con semilla 42.
- Probes de 1 epoca para batch size:
  batch 32 = 96.16 s / 124.79 img/s; batch 64 = 139.97 s / 85.74 img/s;
  batch 128 = 198.09 s / 60.58 img/s.
- `configs/tuning_resnet18.yaml` conserva `batch_size=32` para continuar
  tuning por mejor throughput observado.
- `configs/tuning_resnet18.yaml` ya no usa `full_finetuning` ni referencia
  `data.test_dir`; cada trial ejecuta:
  `classifier_only` 3 epocas con `lr=learning_rate`, luego
  `partial_finetuning` 5 epocas con `lr=learning_rate*0.1`.
- En protocolos de varias etapas, `best_model.pth` y la metrica de seleccion
  se actualizan solo durante la ultima etapa para evitar seleccionar un trial
  por la etapa `classifier_only`.
- `configs/resnet18_finetuning.yaml` y `configs/mobilenetv3_finetuning.yaml`
  quedaron como configs finales de dos etapas sobre dataset completo; sus LR
  ya usan los mejores hiperparametros del tuning ResNet18.
- Tuning ResNet18 dos etapas completado para 6 trials. Mejor trial:
  `learning_rate=0.001`, `partial_finetuning lr=0.0001`,
  `weight_decay=0.0001`, `val_loss=0.0604`, `val_accuracy=0.9825`,
  `val_f1_macro=0.9826`.
- Entrenamiento final ResNet18 dos etapas completado sobre dataset completo:
  231,900 train / 30,000 val. Mejor checkpoint en epoca 15:
  `val_loss=0.0103`, `val_accuracy=0.9975`, `val_f1_macro=0.9975`.
  Early stopping en `partial_finetuning` al llegar a epoca global 22.
- Entrenamiento final MobileNetV3 Small dos etapas completado sobre dataset
  completo: 231,900 train / 30,000 val. Mejor checkpoint en epoca 12:
  `val_loss=0.0156`, `val_accuracy=0.9956`, `val_f1_macro=0.9956`.
  Early stopping en `partial_finetuning` al llegar a epoca global 19.
- Evaluacion test completada:
  ResNet18 `test_loss=0.0139`, `test_accuracy=0.9971`,
  `test_f1_macro=0.9971`, `avg_inference_ms_per_image=2.8845`.
  MobileNetV3 Small `test_loss=0.0215`, `test_accuracy=0.9945`,
  `test_f1_macro=0.9945`, `avg_inference_ms_per_image=0.4298`.
- `scripts/tune.py --dry-run --max-trials 1` fue verificado; solo genero
  archivos ignorados bajo `runs/`.
- `scripts/check_rocm.py` verificado correctamente fuera del sandbox:
  PyTorch ve 1 dispositivo `AMD Radeon RX 9060 XT` y la prueba de tensor en
  `cuda:0` pasa.
- Dentro del sandbox restringido no se expone `/dev/kfd`; para validar GPU se
  requiere ejecutar con acceso real al dispositivo.
- `runs/perf_resnet18_rocm/perf_summary.csv` fue generado en modo `--dry-run`
  con 60 trials planificados: batch size 32/64/128, num_workers 0/2/4/6/8 y
  variantes `baseline`, `miopen_search`, `amp_fp16`, `miopen_amp_fp16`.
- Diagnostico baseline FP32 ejecutado con acceso real a ROCm para 15 trials.
  Mejor resultado por promedio de epocas 2-3:
  batch 32, `num_workers=4`, 93.04 s/epoca, 128.98 img/s train+val,
  GPU avg 96.24%, VRAM max ~2133 MB.
- Diagnostico `miopen_search` ejecutado para 15 trials. Mejor resultado:
  batch 32, `num_workers=6`, 93.17 s/epoca, 128.79 img/s train+val,
  GPU avg 96.94%, VRAM max ~2211 MB. No mejora el baseline.
- Se descartan mas trials AMP/MIOpen+AMP en esta fase: el autotuning MIOpen no
  produjo mejora y la decision operativa ya es estable.
- Politica predeterminada para retomar tuning ResNet18:
  `baseline` FP32, sin variables MIOpen, sin AMP, `batch_size=32`,
  `num_workers=4`, `pin_memory=true`, `persistent_workers=true`,
  `prefetch_factor=2`, `experiment.deterministic=false`.
- `cudnn.benchmark=True` se activa cuando `experiment.deterministic=false`;
  para ROCm la prueba principal de convoluciones sigue siendo MIOpen
  (`MIOPEN_FIND_MODE=NORMAL`).
- `data/`, `dataset/`, `runs/` y `checkpoints/` permanecen ignorados por Git.

## Retomar proxima sesion

1. Leer `AGENTS.md`, `PROJECT_STATE.md` y `TODO.md`.
2. No usar como definitivos los trials 1-2 existentes de
   `runs/tuning_resnet18`: fueron preliminares con subset deterministico.
3. La fase 4.5 concluyo con esta politica predeterminada:
   `baseline` FP32, batch 32, `num_workers=4`, sin MIOpen y sin AMP.
4. Fase 4 ya tiene evaluaciones test guardadas en:

```bash
runs/resnet18_two_stage/test_evaluation/
runs/mobilenetv3_two_stage/test_evaluation/
```

5. La ejecucion con GPU requiere acceso real a ROCm `/dev/kfd`; dentro del
   sandbox PyTorch no ve la GPU.
