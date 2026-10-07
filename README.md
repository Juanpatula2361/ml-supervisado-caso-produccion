# Machine Learning Supervisado – Caso de estudio: Predicción de defectos en producción

**Universidad de Cundinamarca, sede Fusagasugá – Especialización en Inteligencia Artificial**
Curso: Introducción a Machine Learning (2026-2) · Docente: Ing. Wilson Rojas Reales, PhD(c)
Estudiante: Juan Pablo Cristancho Tula, Ing. Electrónico
Actividad: R1-A2-S8 Informe de trabajo – ML Supervisado

## Descripción

La empresa manufacturera XYZ Inc. quiere predecir si un lote de producción saldrá **defectuoso (1)** o **no defectuoso (0)** a partir de cinco variables del proceso: temperatura, presión, velocidad, horas de operación y vibración. Se implementan y comparan seis algoritmos de clasificación supervisada con Scikit-learn, Pandas, NumPy y Matplotlib.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `01_Regresion_Logistica_Caso_Produccion.ipynb` | Fundamentos de la regresión logística: función sigmoide, modelo simple y multivariable, coeficientes y *odds ratio*, probabilidades, matriz de confusión, curva ROC y ajuste del umbral de decisión para priorizar el recall. |
| `02_Comparacion_Clasificadores_Caso_Produccion.ipynb` | Script de la guía modificado: Regresión Logística, Árbol de Decisión, Random Forest, KNN, SVM y Naive Bayes evaluados con 15 métricas, validación cruzada repetida, *Leave-One-Out*, prueba de robustez con ruido y selección del mejor modelo. |
| `C1_CasoProduccion.py` | Script original suministrado en la guía práctica. |
| `figuras/` | Gráficas generadas por los notebooks. |
| `resultados/` | Tablas de métricas en formato CSV. |

## Cómo ejecutar

1. Abrir el notebook en Google Colab: **Archivo → Abrir cuaderno → GitHub** y pegar la URL de este repositorio.
2. Ejecutar todas las celdas (**Entorno de ejecución → Ejecutar todas**). No requiere archivos externos: el dataset está incluido en el código.

Dependencias: `scikit-learn`, `pandas`, `numpy`, `matplotlib` (preinstaladas en Colab).

## Resultados principales

| Modelo | F1 (CV 5×10) | Recall (CV 5×10) | Recall con ruido (k = 0.75) |
|---|---|---|---|
| Regresión Logística | 1.000 | 1.000 | 0.920 |
| Naive Bayes | 0.996 | 1.000 | 0.906 |
| SVM (lineal) | 0.992 | 1.000 | 0.917 |
| KNN (k = 3) | 0.992 | 1.000 | 0.908 |
| Random Forest | 0.989 | 0.983 | 0.907 |
| Árbol de Decisión | 0.970 | 0.967 | 0.859 |

**Mejor modelo: Regresión Logística**, por su recall perfecto y estable, su mejor comportamiento con ruido y sus probabilidades interpretables.

> Limitación: el dataset es simulado y tiene 20 registros, por lo que los resultados deben validarse con datos reales de planta.
