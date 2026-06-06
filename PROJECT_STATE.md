# Project State

## Fase actual

Fase 3 completada. Entorno listo para iniciar Fase 4.

## Implementado

- Constantes inmutables de clases, tamano 224 e ImageNet mean/std.
- Factoria de modelos para ResNet18 y MobileNetV3 Small con dos modos:
  `classifier_only` y `full_finetuning`.
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
- PyTorch ROCm instalado en `.venv`:
  `torch==2.11.0+rocm7.2`, `torchvision==0.26.0+rocm7.2` y
  `torchaudio==2.11.0+rocm7.2`.

## Comandos funcionales

```bash
python scripts/check_rocm.py
python scripts/train.py --config configs/resnet18_finetuning.yaml
python scripts/train.py --config configs/mobilenetv3_finetuning.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data data/test
python scripts/plot_history.py --history runs/experiment/training_history.csv
python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format onnx
python scripts/infer_image.py --checkpoint checkpoints/best_model.pth --image example.jpg
```

## Notas

- No se entrenaron modelos en esta sesion.
- No se generaron checkpoints, metricas reales ni graficas porque no hay
  entrenamiento/checkpoint ejecutado todavia.
- `scripts/check_rocm.py` verificado correctamente fuera del sandbox:
  PyTorch ve 1 dispositivo `AMD Radeon RX 9060 XT` y la prueba de tensor en
  `cuda:0` pasa.
- Dentro del sandbox restringido no se expone `/dev/kfd`; para validar GPU se
  requiere ejecutar con acceso real al dispositivo.
- `data/`, `dataset/`, `runs/` y `checkpoints/` permanecen ignorados por Git.
