# Pipeline Multi-Agente — Rendimiento Estudiantil

Sistema de análisis de rendimiento estudiantil basado en un pipeline de tres agentes de Machine Learning.

## Descripción
Este proyecto implementa un pipeline multi-agente que analiza un dataset de 10,000 estudiantes para predecir su éxito laboral y analizar el impacto de los hábitos de sueño en su rendimiento académico.

## Estructura del Proyecto

```
Boot-con-Mistral.ia/
├── Agentes/
│   ├── tres_agentes.py
│   └── imagenes/
│       ├── matriz_confusion.png
│       ├── curva_roc.png
│       └── real_vs_predicho.png
├── README.md
└── Boot_Mistral.ipynb
```

## Pipeline

```
Dataset CSV (10,000 estudiantes)
        ↓
Agente 1 — Normalizador
        ↓ dataset limpio
Agente 2 — Entrenador
        ↓ métricas + modelos
Agente 3 — Comunicador
        ↓ reporte en lenguaje natural
```

## Agentes

### Agente 1 — Normalizador
Prepara el dataset para el entrenamiento:
- Limpieza e imputación de valores nulos
- Codificación de variables categóricas (`placement_status` → 0/1)
- Escalado de features con `StandardScaler`
- Exporta `dataset_limpio.csv`

### Agente 2 — Entrenador
Entrena y evalúa dos modelos:
- División 80% entrenamiento / 20% prueba
- Entrena **Modelo 1** (Regresión Logística) y **Modelo 2** (Regresión Lineal)
- Genera matriz de confusión y curva ROC
- Exporta `modelo_entrenado.pkl`

### Agente 3 — Comunicador
Genera reporte en lenguaje natural con los resultados de ambos modelos.

## Dataset
----------------------------------------------------------------------------------
|       Columna           | Tipo   | Descripción                                 |
|-------------------------|--------|---------------------------------------------|
| `study_hours`           | int    | Horas de estudio                            |
| `attendance`            | int    | Porcentaje de asistencia                    |
| `sleep_hours`           | int    | Horas de sueño                              |
| `internet_usage`        | int    | Uso de internet                             |
| `assignments_completed` | int    | Tareas completadas                          |
| `previous_score`        | int    | Puntaje previo                              |
| `exam_score`            | float  | Puntaje del examen                          |
| `placement_status`      | object | **Variable objetivo** (Placed / Not Placed) |
----------------------------------------------------------------------------------

## Resultados

### Modelo 1 — ¿Consiguen trabajo los estudiantes?
**Algoritmo:** Regresión Logística
--------------------------
|  Métrica  |    Valor   |
|-----------|------------|
| Accuracy  | 99.75%     |
| Precisión | 99% - 100% |
| Recall    | 99% - 100% |
| AUC-ROC   | 1.00       |
--------------------------

**Distribución:**
- Estudiantes colocados (Placed): 1,655 (82.8%)
- Estudiantes no colocados (Not Placed): 345 (17.2%)

**Conclusión:** El modelo predice con 99.75% de precisión si un estudiante conseguirá trabajo. Las variables más influyentes son el puntaje del examen y el puntaje previo.

#### Matriz de Confusión
![Matriz de Confusión](Agentes/imagenes/matriz_confusion.png)

#### Curva ROC
![Curva ROC](Agentes/imagenes/curva_roc.png)

### Modelo 2 — ¿Dormir afecta el rendimiento?
**Algoritmo:** Regresión Lineal

--------------------
| Métrica | Valor  |
|---------|--------|
| R2      | 5.85%  |
| MSE     | 0.9681 |
--------------------

**Conclusión:** Las horas de sueño tienen una relación muy baja (5.85%) con el rendimiento académico. Dormir más o menos no predice significativamente el éxito laboral. Factores externos no capturados en el dataset (estrés, trabajo part-time, hábitos) probablemente influyen más.

#### Real vs Predicho
![Real vs Predicho](Agentes/imagenes/real_vs_predicho.png)

## Tecnologías

- Python 3
- Google Colab
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib

## Cómo ejecutar

1. Abrir `tres_agentes.py` en Google Colab
2. Subir el dataset CSV cuando se solicite
3. Ejecutar las celdas en orden
4. Los resultados se guardan como `dataset_limpio.csv`, `modelo_entrenado.pkl`, `matriz_confusion.png`, `curva_roc.png` y `real_vs_predicho.png`

## Autora

**Milagros Acosta** — [@miliacoss0](https://github.com/miliacoss0)

---

---

# Boot-con-Mistral.ia 

Proyecto de inteligencia artificial que utiliza **Mistral AI** para crear un agente y chatbot capaz de responder preguntas sobre un dataset de ventas utilizando **RAG (Retrieval-Augmented Generation)**.

## Descripción

Este proyecto implementa un pipeline completo que incluye:
- Limpieza y normalización del dataset de ventas
- Validación de datos
- Agente pandas con LangChain para análisis numérico
- Chatbot RAG con corpus, embeddings, retriever y prefix para preguntas en lenguaje natural

## Tecnologías utilizadas

- **Python**
- **Pandas / NumPy** — manipulación y normalización del dataset
- **Scikit-learn** — MinMaxScaler, SimpleImputer
- **LangChain** — framework para construir el pipeline RAG y el agente
- **Mistral AI** — modelo de lenguaje (`mistral-small-latest`)
- **ChromaDB** — base de datos vectorial para almacenar embeddings
- **Sentence Transformers** — embeddings multilingües (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Google Colab** — entorno de ejecución

## Dataset

Se utiliza el dataset `sales_data_sample.csv` con información de ventas internacionales (~2823 filas, 25 columnas originales).

## Estructura del notebook

### Limpieza y normalización del dataset
- Eliminación de columnas irrelevantes (identificadores, direcciones, datos de contacto)
- Imputación de valores nulos con el valor más frecuente
- Normalización de columnas numéricas con MinMaxScaler (valores entre 0 y 1)
- Validación del dataset: nulos, duplicados, rangos de valores

### Agente Pandas
- Creación de un agente con LangChain + Mistral que ejecuta código pandas automáticamente
- Permite hacer preguntas sobre cálculos y estadísticas del dataset

### RAG con LangChain y Mistral
- **Corpus:** conversión de cada fila del dataset en un documento de texto
- **Embeddings:** vectorización de los documentos con sentence-transformers multilingüe
- **Retriever:** búsqueda semántica de los 10 documentos más relevantes por pregunta
- **Prefix:** instrucciones iniciales que definen el rol y comportamiento del agente
- **Chatbot interactivo:** responde preguntas en español sobre el dataset

## Cómo correr el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/miliacoss0/Boot-con-Mistral.ia.git
```

### 2. Abrir el notebook en Google Colab
Ir a [colab.research.google.com](https://colab.research.google.com) → Archivo → Abrir notebook → GitHub → buscar `miliacoss0/Boot-con-Mistral.ia`

### 3. Configurar la API Key de Mistral
- Crear una cuenta gratuita en [console.mistral.ai](https://console.mistral.ai)
- Generar una API Key
- Guardarla en los Secrets de Colab con el nombre `MISTRAL_API_KEY`

### 4. Subir el dataset
Subir el archivo `sales_data_sample.csv` cuando el notebook lo solicite.

### 5. Ejecutar las celdas en orden
Ejecutar todas las celdas de arriba hacia abajo.

## Ejemplos de preguntas al chatbot

**Para el agente pandas (cálculos):**
- `¿Cuántas filas tiene el dataset?`
- `¿Cuál es el promedio de ventas por país?`
- `¿Cuántos pedidos hay por línea de producto?`

**Para el chatbot RAG (preguntas generales):**
- `¿Qué países aparecen en los datos?`
- `¿Cuál es el deal size más común?`
- `¿Qué territorio tiene más ventas?`
- `¿Cuál es el producto más vendido?`

## Diferencia entre el agente y el RAG

--------------------------------------------------------------------------------
|                |        Agente Pandas       |      Chatbot RAG               |
|----------------|----------------------------|--------------------------------|
| Cómo responde  | Ejecuta código pandas real | Busca en el corpus y responde  |
| Mejor para     |   Cálculos y estadísticas  | Preguntas generales y patrones |
| Usa embeddings | No                         | Sí                             |
--------------------------------------------------------------------------------

## Autora

**Milagros Acosta** — [@miliacoss0](https://github.com/miliacoss0)