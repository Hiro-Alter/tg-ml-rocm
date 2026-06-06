# Changelog

## Fase 1

- Se agrego la estructura base del pipeline de entrenamiento.
- Se agregaron configuraciones YAML iniciales.
- Se agrego el script de verificacion PyTorch/ROCm.

## Fase 2

- Se agrego `scripts/train.py` para entrenamiento desde configuraciones YAML.
- Se agregaron metricas macro por epoca, historial CSV y metricas JSON.
- Se agrego guardado de `best_model.pth` y `last_model.pth`.
- Se agrego early stopping y soporte para `ReduceLROnPlateau` y
  `CosineAnnealingLR`.

## Fase 3

- Se agrego evaluacion de checkpoints con metricas finales y matriz de
  confusion.
- Se agrego generacion de graficas desde historial CSV.
- Se agrego exportacion a `.pth`, TorchScript y ONNX.
- Se agrego inferencia para una imagen individual.
- Se completo la instalacion de PyTorch ROCm 7.2 en `.venv`.
- Se verifico PyTorch ROCm con una RX 9060 XT visible y asignacion de tensor en
  `cuda:0`.
