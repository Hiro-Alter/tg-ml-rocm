# TODO

## Fase 1 - Base reproducible

- [x] Definir orden fijo de clases.
- [x] Definir transformaciones 224x224 con normalizacion ImageNet.
- [x] Crear factoria de ResNet18 y MobileNetV3 Small.
- [x] Crear YAML base de experimentos.
- [x] Crear verificacion de PyTorch/ROCm.

## Fase 2 - Entrenamiento

- [x] Implementar `scripts/train.py`.
- [x] Guardar `best_model.pth`, `last_model.pth`, `metrics.json` y
  `training_history.csv`.
- [x] Implementar early stopping y schedulers.

## Fase 3 - Evaluacion, graficas y exportacion

- [x] Implementar `scripts/evaluate.py`.
- [x] Implementar `scripts/plot_history.py`.
- [x] Implementar `scripts/export_model.py`.
- [x] Implementar `scripts/infer_image.py`.
- [x] Instalar PyTorch ROCm en `.venv`.
- [x] Verificar PyTorch ROCm con `scripts/check_rocm.py`.

## Fase 4 - Tuning y comparacion

- [x] Implementar `scripts/tune.py`.
- [x] Documentar resultados reales parciales en `EXPERIMENTS.md`.
- [x] Hacer el subset de tuning estratificado aleatorio y reproducible.
- [x] Subir `batch_size` de tuning a 128.
- [x] Probar throughput del tuning ResNet18 con `batch_size=128`.
- [x] Probar throughput del tuning ResNet18 con `batch_size=32` y `64`.
- [x] Restaurar `batch_size=32` por mejor throughput observado.
- [x] Agregar `partial_finetuning` para ResNet18 y MobileNetV3 Small.
- [x] Cambiar tuning ResNet18 a protocolo `classifier_only` +
  `partial_finetuning`.
- [x] Evitar uso de test en config de tuning.
- [x] Preparar configs finales de dos etapas para ResNet18 y MobileNetV3 Small.
- [x] Ejecutar tuning ResNet18 corregido con `--rerun-existing`.
- [x] Completar tuning limitado sobre ResNet18 con protocolo de dos etapas.
- [x] Actualizar configs finales con los mejores hiperparametros ResNet18.
- [x] Entrenar ResNet18 final en dataset completo.
- [x] Crear analisis formal del entrenamiento final ResNet18.
- [x] Entrenar MobileNetV3 Small final con los hiperparametros ResNet18.
- [x] Actualizar analisis formal con MobileNetV3 Small.
- [x] Evaluar ambos modelos sobre test.
- [x] Documentar comparacion final en `EXPERIMENTS.md`.
- [x] Crear analisis formal de resultados test para tesis.

## Fase 4.5 - Diagnostico de rendimiento ROCm/DataLoader

- [x] Consultar soporte/limitaciones RX 9060 XT 16GB + ROCm/PyTorch.
- [x] Crear config separada `configs/perf_resnet18_rocm.yaml`.
- [x] Instrumentar tiempos `train_seconds`, `val_seconds` y `epoch_seconds`.
- [x] Preparar matriz batch size 32/64/128 y num_workers 0/2/4/6/8.
- [x] Preparar variantes `baseline`, `miopen_search`, `amp_fp16` y
  `miopen_amp_fp16`.
- [x] Revisar y activar `cudnn.benchmark=True` para diagnostico no
  deterministico.
- [x] Agregar AMP FP16 opcional como variable de diagnostico.
- [x] Preparar monitoreo CPU/RAM/GPU/VRAM/disco por trial.
- [x] Ejecutar diagnostico baseline FP32 con acceso real a ROCm.
- [x] Ejecutar diagnostico MIOpen search.
- [x] Descartar AMP FP16 y MIOpen+AMP FP16 por no justificar mas trials tras
  MIOpen sin mejora.
- [x] Analizar epocas 2-3 y elegir batch/workers/variante.
- [x] Registrar resultados baseline y MIOpen reales en `EXPERIMENTS.md`.
- [x] Dejar `configs/tuning_resnet18.yaml` con la politica elegida.
- [x] Crear resumen formal para tesis de la fase 4.5.
