# Analisis del entrenamiento final de modelos

## Proposito

Este documento resume el entrenamiento final de los modelos seleccionados para
el proyecto. En esta version se documentan ResNet18 y MobileNetV3 Small con
protocolo de dos etapas.

Las metricas descritas aqui corresponden a validacion. La evaluacion sobre el
conjunto de test se documentara en otro capitulo.

## Protocolo experimental

El entrenamiento final de ResNet18 se realizo con los hiperparametros obtenidos
durante el tuning sobre subconjunto estratificado:

- `learning_rate=0.001` para la etapa `classifier_only`.
- `learning_rate=0.0001` para la etapa `partial_finetuning`.
- `weight_decay=0.0001`.
- Optimizador AdamW.
- Scheduler `ReduceLROnPlateau`.
- `batch_size=32`.
- `num_workers=4`.
- Entrenamiento FP32, sin AMP.

El dataset usado para el entrenamiento final fue el dataset completo de
entrenamiento y validacion:

- 231,900 imagenes de entrenamiento.
- 30,000 imagenes de validacion.

El protocolo tuvo dos etapas:

1. `classifier_only`: se congelo el backbone y se entreno solo la capa de
   clasificacion.
2. `partial_finetuning`: se descongelo `layer4` y `fc`, manteniendo congeladas
   las capas previas del backbone.

## Resultados ResNet18

El entrenamiento final de ResNet18 termino por early stopping durante la etapa
`partial_finetuning`.

| Modelo | Mejor epoca | Epoca final | Val loss | Val acc | Val F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 15 | 22 | 0.0103 | 0.9975 | 0.9975 |

El checkpoint seleccionado fue:

```text
checkpoints/resnet18_two_stage/best_model.pth
```

## Evolucion por etapas

Durante la etapa `classifier_only`, el modelo solo ajusto la nueva capa de
clasificacion. En esta etapa el backbone preentrenado actuo como extractor fijo
de caracteristicas. Las metricas evolucionaron de forma estable:

| Epoca | Etapa | Train loss | Train acc | Val loss | Val acc | Val F1 macro |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | classifier_only | 0.6363 | 0.7830 | 0.4733 | 0.8361 | 0.8371 |
| 2 | classifier_only | 0.5209 | 0.8174 | 0.4640 | 0.8353 | 0.8336 |
| 3 | classifier_only | 0.5047 | 0.8220 | 0.4615 | 0.8362 | 0.8352 |
| 4 | classifier_only | 0.4963 | 0.8244 | 0.4431 | 0.8445 | 0.8456 |
| 5 | classifier_only | 0.4940 | 0.8261 | 0.4539 | 0.8408 | 0.8400 |

La etapa `classifier_only` alcanzo aproximadamente 84% de accuracy de
validacion. Esto indica que las caracteristicas preentrenadas de ImageNet son
utiles para la tarea, pero no suficientes para alcanzar el rendimiento final
requerido sin adaptar capas del backbone.

Al iniciar `partial_finetuning`, el rendimiento mejoro de forma marcada:

| Epoca global | LR | Train loss | Train acc | Val loss | Val acc | Val F1 macro |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.0001 | 0.0575 | 0.9821 | 0.0222 | 0.9929 | 0.9929 |
| 7 | 0.0001 | 0.0171 | 0.9947 | 0.0112 | 0.9965 | 0.9965 |
| 8 | 0.0001 | 0.0095 | 0.9969 | 0.0116 | 0.9962 | 0.9962 |
| 9 | 0.0001 | 0.0058 | 0.9981 | 0.0112 | 0.9963 | 0.9963 |
| 10 | 0.0001 | 0.0044 | 0.9986 | 0.0134 | 0.9962 | 0.9962 |
| 14 | 0.00001 | 0.0008 | 0.9998 | 0.0112 | 0.9971 | 0.9971 |
| 15 | 0.00001 | 0.0002 | 1.0000 | 0.0103 | 0.9975 | 0.9975 |
| 20 | 0.000001 | 0.0000 | 1.0000 | 0.0114 | 0.9978 | 0.9978 |
| 22 | 0.000001 | 0.0000 | 1.0000 | 0.0116 | 0.9975 | 0.9975 |

La mejor perdida de validacion se obtuvo en la epoca global 15. Aunque la
accuracy y el F1 macro tuvieron pequenas mejoras posteriores, el criterio de
seleccion fue `val_loss`, por lo que se selecciono la epoca 15 como mejor
checkpoint.

## Interpretacion del cambio en `train_loss`

En este entrenamiento no se observo un aumento perjudicial de `train_loss` al
pasar a `partial_finetuning`; ocurrio lo contrario: el `train_loss` bajo de
0.4940 en la ultima epoca `classifier_only` a 0.0575 en la primera epoca
`partial_finetuning`.

Este cambio abrupto es esperable por la naturaleza del protocolo. Durante
`classifier_only`, el modelo solo puede modificar la capa final, por lo que su
capacidad de adaptacion esta limitada. Al iniciar `partial_finetuning`, se
habilita el entrenamiento de `layer4`, que contiene representaciones mas
especializadas del backbone. Esto aumenta la capacidad del modelo para adaptar
las caracteristicas visuales a las clases del dataset, y por eso la perdida de
entrenamiento cae rapidamente.

La reduccion fuerte de `train_loss` debe interpretarse como una senal de que la
segunda etapa esta adaptando efectivamente el modelo a la tarea. Sin embargo,
tambien genera una brecha creciente entre entrenamiento y validacion: hacia el
final, el `train_loss` se aproxima a cero y el `train_accuracy` llega a 1.0,
mientras que `val_loss` deja de mejorar despues de la epoca 15.

Esa brecha no invalida el entrenamiento, pero indica que despues de cierto punto
el modelo empieza a memorizar mejor el conjunto de entrenamiento sin traducirlo
en una mejora proporcional de la perdida de validacion. Por esa razon fue
importante conservar early stopping y seleccionar el checkpoint por `val_loss`.

## Interpretacion de validacion

La validacion mejoro de manera sustancial al pasar de `classifier_only` a
`partial_finetuning`.

- Ultima epoca `classifier_only`: `val_accuracy=0.8408`,
  `val_f1_macro=0.8400`.
- Primera epoca `partial_finetuning`: `val_accuracy=0.9929`,
  `val_f1_macro=0.9929`.
- Mejor checkpoint: `val_accuracy=0.9975`, `val_f1_macro=0.9975`.

Esto muestra que congelar completamente el backbone es insuficiente para
obtener el mejor rendimiento, pero que no fue necesario descongelar todo el
modelo. La adaptacion parcial de las capas finales fue suficiente para alcanzar
metricas de validacion muy altas, manteniendo un entrenamiento mas controlado
que un `full_finetuning`.

## Early stopping

El entrenamiento se detuvo en la epoca global 22 por early stopping en la etapa
`partial_finetuning`. La mejor perdida de validacion ocurrio en la epoca 15. A
partir de ahi, el modelo mantuvo accuracy y F1 altos, pero la perdida de
validacion no siguio mejorando de forma sostenida.

Esto justifica conservar el checkpoint de la epoca 15 en lugar del ultimo
checkpoint. El ultimo modelo no era necesariamente peor en accuracy, pero el
criterio de seleccion definido para el experimento fue minimizar `val_loss`.

## Conclusion ResNet18

El entrenamiento final de ResNet18 confirma que el protocolo de dos etapas es
adecuado para este problema:

- La etapa `classifier_only` sirve como adaptacion inicial de la cabeza de
  clasificacion.
- La etapa `partial_finetuning` aporta la mejora principal al adaptar las capas
  finales del backbone.
- El mejor checkpoint alcanza `val_accuracy=0.9975` y `val_f1_macro=0.9975`.
- Early stopping evita conservar un checkpoint posterior con menor capacidad de
  generalizacion segun `val_loss`.

La evaluacion sobre test queda fuera del alcance de este documento y sera
tratada en otro capitulo.

## Resultados MobileNetV3 Small

MobileNetV3 Small se entreno con el mismo protocolo general y con los
hiperparametros seleccionados a partir del tuning de ResNet18. No se realizo un
tuning independiente para este modelo.

En la segunda etapa se descongelaron el `classifier` y los ultimos 3 bloques de
`features`.

| Modelo | Mejor epoca | Epoca final | Val loss | Val acc | Val F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3 Small | 12 | 19 | 0.0156 | 0.9956 | 0.9956 |

El checkpoint seleccionado fue:

```text
checkpoints/mobilenetv3_two_stage/best_model.pth
```

## Evolucion MobileNetV3 Small

Durante `classifier_only`, MobileNetV3 Small entreno solo el clasificador. La
validacion fue superior a la obtenida por ResNet18 en la misma etapa, alcanzando
aproximadamente 90% de accuracy de validacion al cierre de la etapa.

| Epoca | Etapa | Train loss | Train acc | Val loss | Val acc | Val F1 macro |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | classifier_only | 0.4567 | 0.8351 | 0.3701 | 0.8640 | 0.8626 |
| 2 | classifier_only | 0.3602 | 0.8687 | 0.2878 | 0.8946 | 0.8945 |
| 3 | classifier_only | 0.3319 | 0.8790 | 0.2799 | 0.8956 | 0.8947 |
| 4 | classifier_only | 0.3126 | 0.8848 | 0.2625 | 0.9029 | 0.9027 |
| 5 | classifier_only | 0.3032 | 0.8892 | 0.2750 | 0.8990 | 0.8980 |

Al iniciar `partial_finetuning`, el modelo tuvo una mejora fuerte, aunque menos
abrupta que ResNet18 porque la etapa `classifier_only` ya habia alcanzado una
validacion mas alta.

| Epoca global | LR | Train loss | Val loss | Val acc | Val F1 macro |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.0001 | 0.0714 | 0.0309 | 0.9893 | 0.9893 |
| 7 | 0.0001 | 0.0304 | 0.0209 | 0.9927 | 0.9927 |
| 8 | 0.0001 | 0.0202 | 0.0189 | 0.9938 | 0.9938 |
| 9 | 0.0001 | 0.0146 | 0.0164 | 0.9947 | 0.9947 |
| 12 | 0.0001 | 0.0070 | 0.0156 | 0.9956 | 0.9956 |
| 17 | 0.00001 | 0.0021 | 0.0165 | 0.9959 | 0.9959 |
| 19 | 0.00001 | 0.0010 | 0.0162 | 0.9961 | 0.9961 |

La mejor perdida de validacion ocurrio en la epoca global 12. Despues de esa
epoca, la accuracy y el F1 macro continuaron con pequenas mejoras, pero
`val_loss` no mejoro de forma sostenida. Por coherencia con el criterio del
experimento, se conserva como mejor checkpoint la epoca 12.

## Comparacion de validacion

| Modelo | Mejor epoca | Val loss | Val acc | Val F1 macro |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 | 15 | 0.0103 | 0.9975 | 0.9975 |
| MobileNetV3 Small | 12 | 0.0156 | 0.9956 | 0.9956 |

Ambos modelos alcanzaron metricas de validacion altas con el protocolo de dos
etapas. ResNet18 obtuvo la menor perdida de validacion y el mayor F1 macro,
mientras que MobileNetV3 Small alcanzo un rendimiento cercano con un modelo
considerablemente mas pequeno.

La evaluacion sobre test queda fuera del alcance de este documento y sera
tratada en otro capitulo.
