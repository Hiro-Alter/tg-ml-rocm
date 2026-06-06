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
- [ ] Repetir tuning ResNet18 corregido con `--rerun-existing`.
- [ ] Completar tuning limitado sobre ResNet18.
- [ ] Aplicar la mejor politica a MobileNetV3 Small.
- [ ] Documentar comparacion final en `EXPERIMENTS.md`.
