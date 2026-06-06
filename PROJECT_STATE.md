# Project State

## Fase actual

Fase 4B parcial: tuning ResNet18 iniciado, cortado tras diagnostico de
rendimiento.

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
- Script `scripts/tune.py` para grid limitado de hiperparametros delegando cada
  trial a `scripts/train.py`.
- `scripts/train.py` soporta limites opcionales estratificados aleatorios por
  clase usando `experiment.seed`:
  `data.max_samples_per_class`, `data.max_train_samples_per_class` y
  `data.max_val_samples_per_class`.
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
python scripts/tune.py --config configs/tuning_resnet18.yaml --dry-run --max-trials 1
python scripts/tune.py --config configs/tuning_resnet18.yaml
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
- `scripts/tune.py --dry-run --max-trials 1` fue verificado; solo genero
  archivos ignorados bajo `runs/`.
- `scripts/check_rocm.py` verificado correctamente fuera del sandbox:
  PyTorch ve 1 dispositivo `AMD Radeon RX 9060 XT` y la prueba de tensor en
  `cuda:0` pasa.
- Dentro del sandbox restringido no se expone `/dev/kfd`; para validar GPU se
  requiere ejecutar con acceso real al dispositivo.
- `data/`, `dataset/`, `runs/` y `checkpoints/` permanecen ignorados por Git.

## Retomar proxima sesion

1. Leer `AGENTS.md`, `PROJECT_STATE.md` y `TODO.md`.
2. No usar como definitivos los trials 1-2 existentes de
   `runs/tuning_resnet18`: fueron preliminares con subset deterministico.
3. Reanudar tuning ResNet18 con el protocolo corregido:

```bash
source .venv/bin/activate
python scripts/check_rocm.py
python scripts/tune.py --config configs/tuning_resnet18.yaml --rerun-existing --stop-on-failure
```

4. La ejecucion debe hacerse con acceso real a ROCm `/dev/kfd`; dentro del
   sandbox PyTorch no ve la GPU.
5. Al terminar el grid, registrar en `EXPERIMENTS.md` los trials corregidos,
   elegir la mejor politica y luego aplicarla a MobileNetV3 Small.
