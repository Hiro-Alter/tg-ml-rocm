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

## Fase 4

- Se agrego `scripts/tune.py` para ejecutar un grid limitado desde YAML.
- Se agrego `--dry-run`, resumen CSV/JSON y deteccion de mejor trial por
  metrica de seleccion.
- Se agregaron limites opcionales de muestras por clase en `scripts/train.py`.
- Se ajusto `configs/tuning_resnet18.yaml` para tuning sobre subset balanceado.
- Se documento tuning parcial ResNet18 y diagnostico de rendimiento GPU.
- Se corrigio la seleccion del subset para usar muestreo estratificado
  aleatorio reproducible con `experiment.seed`.
- Se probo `batch_size=128`; al empeorar throughput, se conservo
  `batch_size=32` para tuning.
- Se agregaron probes comparativos de batch 32, 64 y 128 en `EXPERIMENTS.md`.

## Fase 4.5

- Se agrego `configs/perf_resnet18_rocm.yaml` para diagnostico aislado de
  rendimiento.
- Se agrego `scripts/perf_diagnostics.py` para probar batch size 32/64/128,
  num_workers 0/2/4/6/8 y variantes ROCm con monitoreo de recursos.
- Se amplio `scripts/train.py` con tiempos separados train/val por epoca y
  opciones configurables de DataLoader/determinismo.
- Se agrego AMP FP16 opcional y variantes diagnosticas `amp_fp16` y
  `miopen_amp_fp16`.
- Se ajusto `scripts/perf_diagnostics.py` para acumular resumenes de variantes
  sin sobrescribir resultados previos.
- Se cerro la Fase 4.5 eligiendo baseline FP32, batch 32 y `num_workers=4`
  como politica predeterminada para retomar tuning ResNet18.
- Se agrego `PERFORMANCE_DIAGNOSTIC_PHASE_4_5.md` como resumen formal para
  incluir en la tesis.

## Fase 4 - Ajuste de protocolo

- Se agrego `partial_finetuning` como modo de entrenamiento.
- Se actualizo `scripts/train.py` para soportar protocolos secuenciales con
  `training.stages`.
- Se actualizo `scripts/tune.py` para que cada trial ResNet18 ejecute
  `classifier_only` seguido de `partial_finetuning`.
- En protocolos de varias etapas, la seleccion de mejor checkpoint se limita a
  la ultima etapa.
- Se removio `full_finetuning` del flujo de tuning ResNet18.
- Se prepararon configs finales de dos etapas para ResNet18 y MobileNetV3
  Small.
- Se ejecuto el tuning ResNet18 de dos etapas y se selecciono
  `learning_rate=0.001`, `weight_decay=0.0001`.
- Se actualizaron las configs finales de ResNet18 y MobileNetV3 Small con los
  hiperparametros seleccionados.
- Se entreno ResNet18 final de dos etapas en el dataset completo; mejor
  checkpoint en epoca 15 con `val_accuracy=0.9975` y `val_f1_macro=0.9975`.
- Se agrego `FINAL_TRAINING_ANALYSIS.md` para documentar el analisis del
  entrenamiento final de ResNet18 y reservar espacio para MobileNetV3 Small.
- Se entreno MobileNetV3 Small final de dos etapas en el dataset completo;
  mejor checkpoint en epoca 12 con `val_accuracy=0.9956` y
  `val_f1_macro=0.9956`.
- Se completo `FINAL_TRAINING_ANALYSIS.md` con el analisis de MobileNetV3
  Small y comparacion de validacion frente a ResNet18.
