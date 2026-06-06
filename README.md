# Hand Gesture Training with PyTorch ROCm

Pipeline para entrenar modelos de reconocimiento de gestos estaticos de mano
para direccionamiento de un dron virtual.

## Estado

Fase 3 implementada: entrenamiento, evaluacion, graficas, exportacion e
inferencia individual desde linea de comandos.

## Clases

El orden de clases es fijo:

```text
0 dislike
1 fist
2 like
3 palm
4 peace
5 peace_inverted
6 rock
7 stop
8 two_up
9 two_up_inverted
```

## Datos

No se versionan datasets, imagenes ni checkpoints. Por defecto las
configuraciones esperan carpetas tipo ImageFolder:

```text
data/train/<clase>/*.jpg
data/val/<clase>/*.jpg
data/test/<clase>/*.jpg
```

Cada split debe contener las 10 carpetas con los nombres exactos indicados.

## Dependencias

Entorno local sugerido:

```bash
python3 -m venv .venv
# Si falta python3.12-venv:
# python3 -m virtualenv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
```

En esta sesion se creo `.venv` con `virtualenv` porque el sistema no tiene
`python3.12-venv`. PyTorch ROCm 7.2 quedo instalado y verificado con
`scripts/check_rocm.py`.

## Comandos objetivo

Disponibles:

```bash
python scripts/check_rocm.py
python scripts/train.py --config configs/resnet18_finetuning.yaml
python scripts/train.py --config configs/mobilenetv3_finetuning.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data data/test
python scripts/plot_history.py --history runs/experiment/training_history.csv
python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format onnx
python scripts/infer_image.py --checkpoint checkpoints/best_model.pth --image example.jpg
python scripts/tune.py --config configs/tuning_resnet18.yaml
```

## Protocolo experimental breve

1. Usar pesos preentrenados en ImageNet para ResNet18 y MobileNetV3 Small.
2. Entrenar en modo `classifier_only` y `full_finetuning`.
3. Ajustar de forma limitada `learning_rate` y `weight_decay`, principalmente
   sobre ResNet18.
4. Aplicar la mejor politica a MobileNetV3 Small para comparacion final.
5. Reportar solo metricas reales obtenidas desde evaluacion reproducible.

## Salidas de entrenamiento

`scripts/train.py` guarda:

- `runs/<experimento>/training_history.csv`
- `runs/<experimento>/metrics.json`
- `checkpoints/<experimento>/best_model.pth`
- `checkpoints/<experimento>/last_model.pth`

## Salidas de evaluacion y exportacion

`scripts/evaluate.py` guarda en `evaluation/`:

- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`

`scripts/plot_history.py` guarda en `plots/`:

- `loss.png`
- `accuracy.png`
- `f1_macro.png`

`scripts/export_model.py` guarda en `exports/`:

- `model.pth`
- `model_torchscript.pt`
- `model.onnx`
