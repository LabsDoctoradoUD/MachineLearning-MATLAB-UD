# MachineLearning-MATLAB-UD
Taller práctico de Machine Learning con MATLAB – Clasificación y Regresión

Universidad Distrital Francisco José de Caldas  
Facultad de Ingeniería  

---

## 📌 Descripción general

Este repositorio contiene el material práctico del **Taller de Machine Learning con MATLAB**, orientado a estudiantes de doctorado, maestría, pregrado avanzado y comunidad académica interesada en **aprendizaje automático aplicado**.

El taller se centra en el uso de **Classification Learner** y **Regression Learner** del *Statistics and Machine Learning Toolbox*, combinando fundamentos teóricos con ejercicios prácticos sobre **datasets reales**.

El enfoque del taller es **aprender a tomar decisiones con datos**, no únicamente entrenar modelos.

---

## 🎯 Objetivos del taller

Al finalizar el taller, el participante será capaz de:

- Comprender el flujo completo de un proyecto de Machine Learning.
- Diferenciar problemas de **clasificación** y **regresión**.
- Preparar y explorar conjuntos de datos reales.
- Entrenar y comparar múltiples modelos en MATLAB.
- Ajustar hiperparámetros y analizar su impacto.
- Evaluar modelos usando métricas apropiadas.
- Interpretar resultados y justificar la selección de un modelo.

---

## 🧠 Metodología de trabajo

El taller sigue una metodología **teórico–práctica**, estructurada en tres fases:

1. **Fase 1 – Fundamentos teóricos**  
   Conceptos clave de Machine Learning, tipos de problemas, métricas y flujo de trabajo.

2. **Fase 2 – Práctica guiada en MATLAB**  
   Ejercicios paso a paso usando herramientas gráficas (Apps) y código básico.

3. **Fase 3 – Análisis y discusión**  
   Comparación de modelos, interpretación de resultados y mini–proyecto final.

Este repositorio cubre principalmente la **Fase 2** y sirve de apoyo para la **Fase 3**.

---

## 🛠️ Requisitos técnicos

Antes de iniciar, asegúrate de contar con:

- MATLAB R2022b o superior (recomendado)
- Statistics and Machine Learning Toolbox
- Conocimientos básicos de MATLAB (scripts, variables, tablas)

---

## 📂 Estructura del repositorio

```text
ML_MATLAB_Taller/
│
├── README.md
│
├── docs/
│   ├── Guia_Estudiante.pdf
│   ├── Guia_Instructor.pdf
│   └── Presentacion_Taller.pdf
│
├── exercises/
│   ├── 00_setup/
│   │   └── setup_environment.m
│   │
│   ├── 01_exploracion_clasificacion/
│   │   ├── ejercicio_1_exploracion_fraude.m
│   │   └── ejercicio_1_exploracion.md
│   │
│   ├── 02_clasificacion_modelos/
│   │   ├── ejercicio_2_clasificacion_toolbox.m
│   │   └── ejercicio_2_clasificacion.md
│   │
│   ├── 03_exploracion_regresion/
│   │   ├── ejercicio_3_exploracion_energia.m
│   │   └── ejercicio_3_exploracion.md
│   │
│   ├── 04_regresion_modelos/
│   │   ├── ejercicio_4_regresion_toolbox.m
│   │   └── ejercicio_4_regresion.md
│   │
│   └── 05_discusion_resultados/
│       └── ejercicio_5_analisis_comparativo.md
│
└── data/
    ├── creditcard.csv
    └── Energy_Efficiency.xlsx
