# Fase 3 - Predicción del acceso a internet en hogares paraguayos

Esta carpeta se incorpora en la raíz del repositorio del proyecto, junto a `fase1/` y `fase2/`.

## Contenido de la entrega

- `data/`: dataset limpio de 17.092 hogares y nota de procedencia. Es el mismo contenido que el CSV de `fase2/`; se conserva aquí para que esta fase sea reproducible de forma autónoma.
- `notebooks/Fase3_AccesoInternet_ejecutado.ipynb`: **única notebook incluida**; está ejecutada de principio a fin y conserva sus salidas, tablas y gráficos.
- `src/modelado_fase3.py`: funciones de carga, preprocesamiento, comparación de modelos y evaluación utilizadas por la notebook; también permite ejecutar el pipeline desde consola.
- `src/predecir_internet.py`: aplicación del modelo serializado a un CSV de hogares nuevos.
- `output/`: modelo `.joblib`, métricas, matrices, curvas, importancia por permutación, diagnósticos, ejemplos y exportación HTML de la notebook.
- `informe/`: informe de Fase 3, informe integrador de las tres fases y guía de código, en PDF y DOCX. También se incluyen los informes originales de Fase 1 y Fase 2 como antecedentes.
- `requirements.txt`: versiones de bibliotecas probadas para la ejecución entregada.
- `CHECKLIST_ENTREGA.md`: control final de la documentación y de los pendientes del grupo.

No se incluye una notebook adicional sin ejecutar. Los módulos auxiliares sí se entregan por separado en `src/`, como corresponde a la estructura del proyecto.

## Entorno e instalación

Entorno de referencia probado: Python 3.13.5. Se recomienda crear un entorno virtual dentro de esta carpeta para evitar conflictos con dependencias de otras fases.

### Windows PowerShell

```powershell
cd fase3
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
cd fase3
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Abrir y ejecutar la única notebook

Desde la raíz del repositorio o desde `fase3/`:

```bash
jupyter notebook fase3/notebooks/Fase3_AccesoInternet_ejecutado.ipynb
```

Si ya estás dentro de `fase3/`, utiliza:

```bash
jupyter notebook notebooks/Fase3_AccesoInternet_ejecutado.ipynb
```

La notebook ubica `fase3/data/`, importa las funciones desde `fase3/src/` y guarda los artefactos en `fase3/output/`. Para volver a ejecutar todas las celdas conservando la única notebook:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/Fase3_AccesoInternet_ejecutado.ipynb --inplace --ExecutePreprocessor.timeout=900
```

Para regenerar la exportación HTML:

```bash
python -m jupyter nbconvert --to html notebooks/Fase3_AccesoInternet_ejecutado.ipynb --output Fase3_AccesoInternet.html --output-dir output
```

## Ejecutar el pipeline desde consola (alternativa)

Desde `fase3/`:

```bash
python src/modelado_fase3.py
```

El script vuelve a entrenar y guardar el modelo, las métricas y los diagnósticos en `output/`.

## Aplicar el modelo a nuevos hogares

El CSV de entrada debe incluir los ocho predictores: `AREA_DESC`, `DPTO_NOMBRE`, `TOTAL`, `EDAD_JEFE`, `SEXO_JEFE_DESC`, `ANIOS_ESTUDIO_JEFE`, `CAT_OCUP_JEFE_DESC` e `IPCM`. No debe incluir identificadores ni necesita incluir `TIENE_INTERNET`.

```bash
python src/predecir_internet.py --input output/entrada_ejemplo.csv --output output/predicciones_nuevos_hogares.csv
```

El script agrega `PROBABILIDAD_ESTIMADA_INTERNET` y `CLASIFICACION_UMBRAL_0_5`. La probabilidad no es una garantía individual ni una estimación de cobertura poblacional.

## Diseño metodológico

- Unidad de análisis: hogar. Variable objetivo: `TIENE_INTERNET` (1 = sí; 0 = no).
- Predictores: `AREA_DESC`, `DPTO_NOMBRE`, `TOTAL`, `EDAD_JEFE`, `SEXO_JEFE_DESC`, `ANIOS_ESTUDIO_JEFE`, `CAT_OCUP_JEFE_DESC` e `IPCM`.
- Se excluyen identificadores, variables que revelan o están condicionadas por el objetivo, campos redundantes y `FACTOR` como predictor.
- Partición estratificada 60/20/20, semilla 42; validación cruzada estratificada de 3 pliegues y búsqueda en grilla solo sobre entrenamiento.
- Se compararon regresión logística, Random Forest e HistGradientBoosting. Selección por F1 de validación, con AUC-ROC como desempate; el test quedó reservado para la evaluación final.
- El preprocesamiento está dentro de los pipelines: imputación, one-hot encoding, escalado y transformación `log1p` de IPCM.
- Los valores extremos de IPCM se conservaron en el modelo principal; los informes documentan las limitaciones y el análisis de sensibilidad inferencial.

## Resultados de la ejecución incluida

Modelo seleccionado: **HistGradientBoosting**. En el conjunto de prueba: accuracy 0,8023; precision 0,8238; recall 0,8672; F1 0,8450 y AUC-ROC 0,8695. La línea base mayoritaria obtuvo F1 0,7664 y AUC-ROC 0,5000. La matriz de confusión (filas = clase real; columnas = predicción) fue `[[901, 394], [282, 1842]]`.

El análisis de errores encontró diferencias entre áreas: entre hogares conectados, error de 39,2 % en rurales frente a 4,8 % en urbanos; entre hogares sin conexión, 17,7 % en rurales frente a 59,7 % en urbanos. Son métricas internas y no ponderadas de esta partición.

## Límites de interpretación y uso

No se incorporaron ponderadores, estratos ni conglomerados del diseño muestral complejo de la EPHC. Por ello, las métricas no deben presentarse como estimaciones nacionales ponderadas. El estudio utiliza una sola edición anual, no es causal y no cuenta con validación externa o temporal. La disparidad de errores por área requiere auditoría antes de cualquier uso aplicado. No utilizar el clasificador para negar servicios, beneficios o derechos.




