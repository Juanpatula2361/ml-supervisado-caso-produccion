# ============================================================
# COMPARACIÓN DE ALGORITMOS DE CLASIFICACIÓN
# Caso de estudio: Predicción de lotes defectuosos en producción
# ============================================================

import pandas as pd
import numpy as np

# Modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Herramientas de evaluación
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. CREACIÓN DEL DATASET
# ============================================================

# Dataset simulado de producción
# defectuoso:
# 0 = lote correcto
# 1 = lote defectuoso

data = {
    "temperatura": [
        70, 72, 75, 80, 85, 90, 95, 100, 78, 82,
        88, 92, 97, 76, 73, 89, 94, 99, 81, 86
    ],
    "presion": [
        30, 32, 34, 36, 40, 42, 45, 48, 35, 37,
        41, 43, 46, 33, 31, 42, 44, 47, 36, 39
    ],
    "velocidad": [
        120, 118, 115, 110, 105, 100, 95, 90, 112, 108,
        102, 98, 93, 116, 119, 101, 96, 91, 109, 104
    ],
    "horas_operacion": [
        2, 3, 4, 6, 8, 10, 12, 14, 5, 7,
        9, 11, 13, 4, 3, 10, 12, 15, 6, 8
    ],
    "vibracion": [
        1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 1.8, 2.2,
        2.8, 3.2, 3.7, 1.5, 1.3, 3.1, 3.6, 4.2, 2.1, 2.6
    ],
    "defectuoso": [
        0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
        1, 1, 1, 0, 0, 1, 1, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

print("Dataset inicial:")
print(df.head())

# ============================================================
# 2. DEFINICIÓN DE VARIABLES
# ============================================================

# X contiene las variables predictoras
X = df[[
    "temperatura",
    "presion",
    "velocidad",
    "horas_operacion",
    "vibracion"
]]

# y contiene la variable objetivo
y = df["defectuoso"]

# ============================================================
# 3. DIVISIÓN DE DATOS
# ============================================================

# Se separan los datos en entrenamiento y prueba.
# 70% entrenamiento, 30% prueba.
# stratify=y mantiene la proporción de clases en ambos conjuntos.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. DEFINICIÓN DE MODELOS
# ============================================================

# Se crea un diccionario con los modelos a comparar

modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel="linear"),
    "Naive Bayes": GaussianNB()
}

# ============================================================
# 5. ENTRENAMIENTO, PREDICCIÓN Y EVALUACIÓN
# ============================================================

resultados = []

for nombre, modelo in modelos.items():

    print("\n" + "="*70)
    print("MODELO:", nombre)
    print("="*70)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Realizar predicciones con el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # Calcular métricas principales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Validación cruzada con 5 particiones
    cv_scores = cross_val_score(
        modelo,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_promedio = cv_scores.mean()

    # Guardar resultados en una lista
    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Validación cruzada": cv_promedio
    })

    # Mostrar matriz de confusión
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Mostrar reporte completo
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Mostrar validación cruzada
    print("Resultados validación cruzada:", cv_scores)
    print("Promedio validación cruzada:", cv_promedio)

# ============================================================
# 6. TABLA COMPARATIVA FINAL
# ============================================================

df_resultados = pd.DataFrame(resultados)

print("\n" + "="*70)
print("TABLA COMPARATIVA DE MODELOS")
print("="*70)

print(df_resultados.sort_values(by="F1-score", ascending=False))

# ============================================================
# 7. SELECCIÓN DEL MEJOR MODELO
# ============================================================

mejor_modelo = df_resultados.sort_values(
    by="F1-score",
    ascending=False
).iloc[0]

print("\nMejor modelo según F1-score:")
print(mejor_modelo)