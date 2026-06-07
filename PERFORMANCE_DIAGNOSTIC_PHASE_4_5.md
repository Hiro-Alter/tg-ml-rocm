# Diagnostico de rendimiento ROCm/DataLoader

## Contexto

Durante la fase de ajuste del modelo ResNet18 se observo un rendimiento de
entrenamiento menor al esperado para una GPU AMD Radeon RX 9060 XT de 16 GB.
Antes de continuar con el tuning de hiperparametros del modelo, se definio una
fase intermedia orientada a identificar si el cuello de botella provenia del
tamano de lote, la carga de datos, la configuracion de ROCm/MIOpen o el uso de
precision mixta.

El objetivo de esta fase no fue mejorar la metrica de clasificacion, sino
establecer una configuracion de entrenamiento mas eficiente y reproducible para
continuar los experimentos posteriores.

## Metodologia

Se uso una configuracion diagnostica separada para no comprometer los avances
previos del proyecto. El entrenamiento se realizo con ResNet18 en modo
`full_finetuning`, pesos iniciales de ImageNet y un subconjunto balanceado del
dataset:

- 10,000 imagenes de entrenamiento.
- 2,000 imagenes de validacion.
- Imagenes de entrada de 224 x 224 pixeles.
- Optimizador AdamW.
- `learning_rate=0.001`.
- `weight_decay=0.00001`.
- Scheduler `ReduceLROnPlateau`.
- 3 epocas por prueba.

Para evitar que la primera epoca sesgara la medicion por calentamiento del
sistema, carga inicial de kernels y estabilizacion del pipeline, la comparacion
de tiempos se hizo usando unicamente el promedio de las epocas 2 y 3.

Las variables evaluadas fueron:

- `batch_size`: 32, 64 y 128.
- `num_workers`: 0, 2, 4, 6 y 8.
- Variante baseline FP32.
- Variante MIOpen search con `MIOPEN_FIND_MODE=NORMAL` y
  `MIOPEN_FIND_ENFORCE=SEARCH`.

Adicionalmente, se preparo soporte para AMP FP16, pero no se continuo con esos
trials debido a que la comparacion baseline vs MIOpen ya mostro una conclusion
estable sobre el comportamiento del sistema.

Durante cada prueba se monitoreo:

- Uso de GPU.
- Uso de VRAM.
- Uso de CPU.
- Uso de RAM.
- Lectura/escritura de disco.
- Tiempo de entrenamiento, validacion y epoca completa.

## Resultados principales

La mejor configuracion baseline fue:

| Variante | Batch size | Num workers | Segundos/epoca | Img/s train+val | GPU promedio | VRAM maxima |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline FP32 | 32 | 4 | 93.04 | 128.98 | 96.24% | 2132.60 MB |

El mejor resultado con MIOpen search fue:

| Variante | Batch size | Num workers | Segundos/epoca | Img/s train+val | GPU promedio | VRAM maxima |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIOpen search FP32 | 32 | 6 | 93.17 | 128.79 | 96.94% | 2210.91 MB |

La comparacion muestra que MIOpen search no produjo una mejora real frente al
baseline. La diferencia entre 93.04 s y 93.17 s por epoca esta dentro del ruido
operativo normal de este tipo de medicion, y la variante MIOpen uso ligeramente
mas memoria de video.

Tambien se observo que:

- `num_workers=0` penaliza notablemente el rendimiento.
- `num_workers=2`, `4` y `6` producen resultados muy similares.
- `num_workers=4` fue elegido por ser el mejor resultado medido y mantener una
  configuracion conservadora.
- `batch_size=64` queda cerca del rendimiento de batch 32, pero no lo mejora y
  requiere mas VRAM.
- `batch_size=128` empeora el throughput y aumenta considerablemente el uso de
  VRAM.

## Configuracion seleccionada

Con base en los resultados, se dejo como configuracion predeterminada para
continuar el tuning de ResNet18:

```yaml
experiment:
  deterministic: false

data:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

training:
  mixed_precision:
    enabled: false
```

La configuracion seleccionada corresponde a entrenamiento baseline FP32, sin
variables de entorno MIOpen adicionales y sin AMP FP16.

## Justificacion de la decision

La decision de mantener entrenamiento baseline FP32 se basa en que el intento de
forzar busqueda de algoritmos mediante MIOpen no redujo los tiempos de epoca. En
consecuencia, continuar ampliando la matriz de pruebas hacia AMP FP16 y
MIOpen+AMP FP16 implicaba un costo computacional adicional sin una senal previa
de mejora clara en el backend.

El principal ajuste efectivo fue el uso de multiples workers en el DataLoader.
Esto indica que parte del problema original estaba asociado al pipeline de carga
de datos y no exclusivamente a la capacidad de computo de la GPU.

Por tanto, se concluye que la configuracion mas adecuada para continuar los
experimentos del proyecto es:

- Batch size 32.
- `num_workers=4`.
- Entrenamiento FP32.
- Sin MIOpen search como configuracion por defecto.
- Sin AMP FP16 en esta etapa.
- `pin_memory=true`.
- `persistent_workers=true`.
- `prefetch_factor=2`.

Esta configuracion prioriza el mejor tiempo medido, estabilidad del pipeline y
menor consumo de VRAM frente a alternativas que no demostraron mejora.

## Alcance pendiente

Con la fase de diagnostico cerrada, el siguiente paso del proyecto es retomar el
tuning de hiperparametros del modelo ResNet18 usando la configuracion
seleccionada. En esa fase se evaluaran nuevamente los hiperparametros de
entrenamiento definidos en `configs/tuning_resnet18.yaml`, principalmente
`learning_rate` y `weight_decay`.
