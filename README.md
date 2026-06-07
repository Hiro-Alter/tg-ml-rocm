# Reconocimiento de gestos de mano con PyTorch ROCm

Proyecto de entrenamiento, evaluacion y exportacion de modelos de vision por
computador para clasificar gestos estaticos de mano. El caso de uso objetivo es
el direccionamiento de un dron virtual a partir de gestos reconocidos en imagen.

El repositorio incluye el pipeline reproducible de entrenamiento, los resultados
finales, las graficas principales para presentacion, y modelos exportados listos
para inferencia en otros proyectos.

## Resumen

- Tarea: clasificacion multiclase de 10 gestos estaticos de mano.
- Modelos evaluados: ResNet18 y MobileNetV3 Small con pesos ImageNet.
- Framework: PyTorch sobre ROCm.
- Dataset local: estructura tipo ImageFolder con splits `train`, `val` y
  `test`.
- Entrada del modelo: imagen RGB de `224x224`, normalizada con ImageNet
  mean/std.
- Salida del modelo: logits para las 10 clases.
- Mejor modelo por exactitud test: ResNet18.
- Mejor modelo por tiempo de inferencia: MobileNetV3 Small.

## Clases

El orden de clases es fijo y debe mantenerse igual en entrenamiento,
evaluacion e inferencia:

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

## Estructura Principal

```text
configs/                     Configuraciones YAML finales y diagnosticas
figures/final_results/       Graficas finales para informe/presentacion
models/                      Modelos exportados para uso externo
runs/                        Resultados finales versionados
scripts/                     Scripts de entrenamiento, evaluacion y exportacion
src/hand_gesture_rocm/       Codigo reutilizable del pipeline
```

Los directorios `data/`, `dataset/`, `checkpoints/` y la mayoria de `runs/`
son locales e ignorados por Git. Los resultados finales y modelos exportados
fueron agregados explicitamente al repositorio.

## Datos

Las configuraciones esperan un dataset tipo ImageFolder:

```text
data/train/<clase>/*.jpg
data/val/<clase>/*.jpg
data/test/<clase>/*.jpg
```

Cada split debe contener las 10 carpetas con los nombres exactos de clase. En
este proyecto se usaron symlinks locales desde el dataset procesado hacia
`data/train`, `data/val` y `data/test`.

## Preprocesamiento

El preprocesamiento usado durante entrenamiento, evaluacion e inferencia es:

```text
Resize: 224x224
Color: RGB
Tensor: [batch, 3, 224, 224]
Rango antes de normalizar: 0.0..1.0
Mean ImageNet: [0.485, 0.456, 0.406]
Std ImageNet:  [0.229, 0.224, 0.225]
```

El modelo no incluye el preprocesamiento dentro del archivo exportado. Cualquier
proyecto externo debe aplicar estos pasos antes de ejecutar inferencia.

## Modelos

Se trabajaron dos arquitecturas:

- ResNet18: mayor exactitud final en test.
- MobileNetV3 Small: menor tiempo medio de inferencia.

Ambas arquitecturas usan una cabeza de clasificacion ajustada a 10 clases.

## Protocolo Experimental

El entrenamiento final siguio un protocolo de dos etapas:

1. `classifier_only`: se entrena solo el clasificador final.
2. `partial_finetuning`: se descongela una parte final del backbone y el
   clasificador.

En ResNet18, `partial_finetuning` descongela `layer4` y `fc`. En MobileNetV3
Small, descongela `classifier` y los ultimos tres bloques de `features`.

La seleccion de hiperparametros se hizo con tuning limitado sobre ResNet18,
usando subset estratificado aleatorio reproducible. El mejor trial fue:

```text
learning_rate = 0.001
weight_decay = 0.0001
partial_finetuning lr = 0.0001
selection_metric = val_loss
best_val_loss = 0.0604
best_val_accuracy = 0.9825
best_val_f1_macro = 0.9826
```

Esos hiperparametros se usaron luego en el entrenamiento final sobre el dataset
completo.

## Resultados Finales

### Validacion

ResNet18:

```text
Mejor epoca: 15
val_loss: 0.0103
val_accuracy: 0.9975
val_f1_macro: 0.9975
```

MobileNetV3 Small:

```text
Mejor epoca: 12
val_loss: 0.0156
val_accuracy: 0.9956
val_f1_macro: 0.9956
```

### Test

```text
ResNet18
test_loss: 0.0139
test_accuracy: 0.9971
test_f1_macro: 0.9971
avg_inference_ms_per_image: 2.8845

MobileNetV3 Small
test_loss: 0.0215
test_accuracy: 0.9945
test_f1_macro: 0.9945
avg_inference_ms_per_image: 0.4298
```

Interpretacion principal:

- ResNet18 obtuvo el mejor desempeno predictivo.
- MobileNetV3 Small fue mas rapido para inferencia.
- La diferencia de exactitud fue pequena, por lo que MobileNetV3 Small es una
  opcion fuerte si la prioridad es latencia.

## Diagnostico ROCm/DataLoader

Se ejecuto una fase diagnostica para elegir una politica operativa estable en
ROCm. La configuracion final recomendada para este entorno fue:

```text
variant: baseline FP32
batch_size: 32
num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2
AMP: false
MIOpen search: no usado como politica final
```

Resultado destacado:

```text
batch_size = 32
num_workers = 4
epoch_seconds_mean = 93.04
timed_epoch_img_s_mean = 128.98
GPU avg = 96.24%
VRAM max ~= 2133 MB
```

## Graficas Finales

Las graficas versionadas estan en:

```text
figures/final_results/
```

Incluyen:

- `01_entrenamiento_resnet18.png`
- `01_entrenamiento_mobilenetv3_small.png`
- `02_comparacion_test.png`
- `03_matriz_confusion_resnet18.png`
- `03_matriz_confusion_mobilenetv3_small.png`
- `04_tiempo_inferencia.png`
- `05_tuning_resnet18.png`
- `06_diagnostico_rocm_dataloader.png`

La grafica de tuning es importante porque muestra el trial seleccionado con el
subset estratificado y justifica los hiperparametros usados en el entrenamiento
final.

## Modelos Exportados

Los modelos finales listos para usar en otros proyectos estan en:

```text
models/resnet18/
models/mobilenetv3_small/
```

Cada carpeta contiene:

```text
model_torchscript.pt      Modelo portable para PyTorch/TorchScript
model.onnx                Modelo portable para ONNX Runtime u otros runtimes
model_state_dict.pth      Pesos PyTorch con metadata minima
metadata.json             Metadatos de arquitectura, clases y preprocesamiento
labels.txt                Clases en orden de salida
```

Tambien existe:

```text
models/manifest.json
```

Ese archivo resume las rutas de artefactos, el formato de entrada y la forma de
interpretar la salida.

## Uso en Otro Proyecto

La forma recomendada si el otro proyecto usa PyTorch es cargar TorchScript:

```python
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

model_dir = Path("models/resnet18")

labels = (model_dir / "labels.txt").read_text().splitlines()
model = torch.jit.load(str(model_dir / "model_torchscript.pt"), map_location="cpu")
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

image = Image.open("imagen.jpg").convert("RGB")
x = preprocess(image).unsqueeze(0)

with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)[0]
    class_id = int(torch.argmax(probabilities))
    class_name = labels[class_id]
    confidence = float(probabilities[class_id])

print(class_name, confidence)
```

Para proyectos que no usan PyTorch, usar `model.onnx` con ONNX Runtime. La
entrada y salida son las mismas: tensor `[batch, 3, 224, 224]` y logits de 10
clases.

## Entorno

Entorno local sugerido:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
```

En este entorno se uso PyTorch ROCm 7.2. La validacion de GPU requiere acceso
real al dispositivo ROCm; dentro de algunos sandboxes no se expone `/dev/kfd`.

## Comandos Principales

Verificar ROCm:

```bash
python scripts/check_rocm.py
```

Entrenar modelos finales:

```bash
python scripts/train.py --config configs/resnet18_finetuning.yaml
python scripts/train.py --config configs/mobilenetv3_finetuning.yaml
```

Ejecutar tuning ResNet18:

```bash
python scripts/tune.py --config configs/tuning_resnet18.yaml
```

Evaluar un checkpoint:

```bash
python scripts/evaluate.py --checkpoint checkpoints/resnet18_two_stage/best_model.pth --data data/test
```

Generar graficas finales:

```bash
python scripts/plot_final_results.py
```

Exportar modelos portables:

```bash
python scripts/export_portable_models.py
```

Inferencia individual desde el repo:

```bash
python scripts/infer_image.py --checkpoint checkpoints/resnet18_two_stage/best_model.pth --image example.jpg
```

## Archivos Clave

```text
configs/resnet18_finetuning.yaml
configs/mobilenetv3_finetuning.yaml
configs/tuning_resnet18.yaml
configs/perf_resnet18_rocm.yaml
scripts/train.py
scripts/evaluate.py
scripts/tune.py
scripts/perf_diagnostics.py
scripts/plot_final_results.py
scripts/export_portable_models.py
scripts/infer_image.py
src/hand_gesture_rocm/
```

## Notas de Reproducibilidad

- No se versiona el dataset completo.
- Los resultados finales de `runs/` fueron versionados para soportar las
  graficas finales.
- Los modelos exportados en `models/` fueron versionados para facilitar uso en
  otros proyectos.
- Los checkpoints locales bajo `checkpoints/` se conservan para retomar trabajo,
  pero no todos estan versionados.
- `runs/tuning_resnet18` y `checkpoints/tuning_resnet18` se conservan porque
  respaldan la seleccion de hiperparametros usada en el entrenamiento final.
