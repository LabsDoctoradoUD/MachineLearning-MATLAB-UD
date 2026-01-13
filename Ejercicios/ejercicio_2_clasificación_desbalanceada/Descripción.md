# Ejercicio 2 — Clasificación Desbalanceada

En este ejercicio se aborda un problema de clasificación binaria
con un conjunto de datos real y desbalanceado, utilizando la herramienta
gráfica **Classification Learner** de MATLAB.

El objetivo principal es analizar las limitaciones de la métrica
**accuracy** y comprender la importancia de otras métricas de evaluación
en contextos reales.

---

## Dataset

**Bank Marketing Dataset**  
Fuente: UCI Machine Learning Repository

- Observaciones: ~41.000
- Variables: 20 (numéricas y categóricas)
- Variable objetivo: `y`  
  - `yes` → el cliente acepta el producto  
  - `no` → el cliente no acepta el producto  

El dataset presenta un **fuerte desbalance** entre clases, lo cual es
característico de muchos problemas reales en la industria.

---

## Objetivos del ejercicio

- Comprender el impacto del desbalance de clases
- Identificar las limitaciones de la accuracy
- Analizar matrices de confusión
- Comparar métricas como Precision y Recall
- Evaluar distintos modelos de clasificación

---

## Actividades

1. Cargar el dataset y analizar la distribución de clases
2. Preparar los datos para entrenamiento
3. Entrenar modelos usando Classification Learner
4. Comparar métricas de evaluación
5. Analizar falsos positivos y falsos negativos
6. Discutir el impacto del error según el contexto del problema

---

## Modelos sugeridos

- K-Nearest Neighbors (KNN)
- Árboles de decisión
- Modelos Ensemble

---

## Discusión guiada

- ¿Es la accuracy una métrica confiable en este problema?
- ¿Qué tipo de error es más costoso?
- ¿Cómo cambia el modelo al priorizar Recall sobre Accuracy?
- ¿Qué modelo sería preferible en un contexto real?

---

## Nota

Para fines docentes, durante el taller se utiliza una versión reducida
del dataset con el fin de acelerar el entrenamiento y facilitar la
interacción con la interfaz gráfica.

---

## Resultado esperado

Al finalizar el ejercicio, el participante será capaz de interpretar
correctamente los resultados de un modelo de clasificación desbalanceada
y justificar la elección de métricas y modelos según el problema.
