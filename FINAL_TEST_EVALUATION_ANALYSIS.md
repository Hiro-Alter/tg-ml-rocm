# Analisis de evaluacion final sobre test

## Proposito

Este documento presenta el analisis de los resultados finales obtenidos sobre
el conjunto de test. A diferencia del analisis de entrenamiento y validacion,
esta evaluacion se realiza sobre datos no usados para ajustar hiperparametros,
entrenar pesos ni seleccionar checkpoints.

El objetivo es comparar el comportamiento final de ResNet18 y MobileNetV3
Small bajo el protocolo de dos etapas definido en el proyecto.

## Configuracion de evaluacion

Los modelos evaluados fueron los mejores checkpoints seleccionados por
`val_loss` durante el entrenamiento final:

- ResNet18:
  `checkpoints/resnet18_two_stage/best_model.pth`.
- MobileNetV3 Small:
  `checkpoints/mobilenetv3_two_stage/best_model.pth`.

La evaluacion se realizo sobre:

- Dataset: `data/test`.
- Total de imagenes: 50,000.
- Clases: 10.
- Soporte por clase: 5,000 imagenes.

Las salidas generadas se almacenaron en:

- `runs/resnet18_two_stage/test_evaluation/`.
- `runs/mobilenetv3_two_stage/test_evaluation/`.

Cada carpeta contiene metricas globales, metricas por clase, matriz de
confusion en CSV y matriz de confusion en imagen.

## Resultados globales

| Modelo | Test loss | Accuracy | Precision macro | Recall macro | F1 macro | Parametros | Inferencia ms/img |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 0.0139 | 0.9971 | 0.9971 | 0.9971 | 0.9971 | 11,181,642 | 2.8845 |
| MobileNetV3 Small | 0.0215 | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 1,528,106 | 0.4298 |

ResNet18 obtuvo el mejor rendimiento global. Su accuracy y F1 macro fueron
aproximadamente 0.26 puntos porcentuales superiores a los de MobileNetV3 Small.
Tambien obtuvo una menor perdida de test, lo que indica predicciones mas
confiables bajo el criterio de entropia cruzada.

MobileNetV3 Small, por su parte, obtuvo un rendimiento ligeramente inferior,
pero con una ventaja clara en eficiencia. Su tiempo promedio de inferencia fue
aproximadamente 6.7 veces menor que el de ResNet18. Ademas, tiene cerca del
13.7% de los parametros de ResNet18.

## Analisis por clase: ResNet18

ResNet18 mostro un rendimiento muy alto en todas las clases. Sus clases con
mayor F1 fueron:

| Clase | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| dislike | 0.9996 | 0.9998 | 0.9997 |
| fist | 0.9996 | 0.9996 | 0.9996 |
| like | 0.9986 | 0.9996 | 0.9991 |

Las clases con menor F1 fueron:

| Clase | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| peace | 0.9952 | 0.9938 | 0.9945 |
| two_up | 0.9950 | 0.9954 | 0.9952 |
| peace_inverted | 0.9976 | 0.9948 | 0.9962 |
| palm | 0.9968 | 0.9960 | 0.9964 |

Aunque estas son las clases mas dificiles para ResNet18, sus valores siguen
siendo altos. Esto sugiere que el modelo generaliza correctamente, pero que las
confusiones restantes se concentran en gestos visualmente cercanos.

Las principales confusiones de ResNet18 fueron:

| Clase real | Prediccion | Casos |
| --- | --- | ---: |
| palm | stop | 18 |
| peace | two_up | 17 |
| peace_inverted | two_up_inverted | 16 |
| stop | palm | 13 |
| two_up | peace | 12 |
| peace | rock | 9 |

Estas confusiones son coherentes con la similitud visual entre pares de gestos.
Por ejemplo, `palm` y `stop` pueden compartir una estructura de mano abierta,
mientras que `peace` y `two_up` dependen de diferencias finas en orientacion o
posicion relativa de los dedos.

## Analisis por clase: MobileNetV3 Small

MobileNetV3 Small tambien alcanzo metricas altas en todas las clases, aunque
con una reduccion moderada frente a ResNet18.

Sus clases con mayor F1 fueron:

| Clase | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| dislike | 0.9984 | 0.9996 | 0.9990 |
| fist | 0.9990 | 0.9990 | 0.9990 |
| like | 0.9984 | 0.9980 | 0.9982 |

Las clases con menor F1 fueron:

| Clase | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| peace | 0.9896 | 0.9872 | 0.9884 |
| two_up | 0.9873 | 0.9920 | 0.9896 |
| peace_inverted | 0.9938 | 0.9918 | 0.9928 |
| two_up_inverted | 0.9934 | 0.9924 | 0.9929 |

El patron de error es similar al de ResNet18, pero con mayor numero de
confusiones entre gestos relacionados con dedos extendidos. Esto es esperable
en un modelo mas compacto, ya que su menor capacidad puede limitar la
representacion de detalles finos en gestos visualmente parecidos.

Las principales confusiones de MobileNetV3 Small fueron:

| Clase real | Prediccion | Casos |
| --- | --- | ---: |
| peace | two_up | 40 |
| peace_inverted | two_up_inverted | 25 |
| two_up | peace | 20 |
| stop | palm | 17 |
| rock | peace | 17 |
| two_up_inverted | peace_inverted | 16 |
| palm | stop | 16 |

La mayor confusion `peace -> two_up` muestra que MobileNetV3 Small tiene mas
dificultad para separar gestos con estructura de dedos similar. Aun asi, el
error absoluto sigue siendo bajo en relacion con el soporte de 5,000 imagenes
por clase.

## Comparacion de errores

Ambos modelos concentran sus errores en pares de clases visualmente cercanos:

- `peace` y `two_up`.
- `peace_inverted` y `two_up_inverted`.
- `palm` y `stop`.
- `peace` y `rock`.

ResNet18 reduce mejor estas confusiones, especialmente en `peace -> two_up`,
donde registra 17 errores frente a 40 de MobileNetV3 Small. Esta diferencia
explica parte de la brecha en F1 macro.

Sin embargo, MobileNetV3 Small mantiene una exactitud de 0.9945, lo cual indica
que la perdida de rendimiento es pequena en terminos absolutos. Por tanto, la
eleccion entre ambos modelos depende del escenario de uso:

- Si se prioriza la maxima precision, ResNet18 es la mejor opcion.
- Si se prioriza velocidad de inferencia y menor tamano, MobileNetV3 Small es
  una alternativa competitiva.

## Analisis de eficiencia

La diferencia de eficiencia entre los modelos es clara:

| Modelo | Parametros | Inferencia ms/img | Velocidad relativa |
| --- | ---: | ---: | ---: |
| ResNet18 | 11,181,642 | 2.8845 | 1.0x |
| MobileNetV3 Small | 1,528,106 | 0.4298 | 6.7x mas rapido |

MobileNetV3 Small reduce fuertemente el costo computacional. Esta caracteristica
lo hace atractivo para escenarios donde la inferencia deba ejecutarse en tiempo
real, con recursos limitados o con restricciones de latencia.

ResNet18, en cambio, conserva ventaja cuando la prioridad es maximizar la
calidad predictiva. Su mayor capacidad permite reducir errores en las clases
mas ambiguas del conjunto.

## Conclusion

La evaluacion final confirma que ambos modelos generalizan bien al conjunto de
test. ResNet18 obtuvo el mejor rendimiento absoluto, con `accuracy=0.9971` y
`F1 macro=0.9971`. MobileNetV3 Small obtuvo `accuracy=0.9945` y
`F1 macro=0.9945`, con una velocidad de inferencia aproximadamente 6.7 veces
mayor.

En consecuencia, ResNet18 es el modelo recomendado cuando el objetivo principal
es maximizar la precision. MobileNetV3 Small es la alternativa recomendada
cuando se requiere un modelo mas liviano y rapido, aceptando una perdida
moderada de rendimiento.

Estos resultados cierran la comparacion final sobre test y permiten seleccionar
el modelo segun los requisitos practicos del sistema.
