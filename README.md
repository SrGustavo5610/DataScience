# Acceso a internet en los hogares paraguayos (EPHC Anual 2025)

Trabajo práctico de **Data Science (Electiva, Curso TQ)** — Facultad Politécnica, Universidad Nacional de Asunción.
Integrantes: Gustavo Fabián Carballo Rojas · Joel María Mendieta Candia

## Problema
¿En qué medida el acceso a internet de los hogares paraguayos varía según el área de residencia, el ingreso per cápita y la educación del jefe/a, y es posible predecir si un hogar cuenta con conexión a partir de esas características?

## Objetivos
- **General:** analizar los factores socioeconómicos, educativos y geográficos asociados al acceso a internet y evaluar en qué medida permiten predecir la conexión.
- **Específicos:** (1) caracterizar el acceso por área, departamento, educación e ingreso; (2) contrastar diferencias entre grupos con pruebas estadísticas; (3) identificar las variables más asociadas; (4) entrenar y comparar modelos de clasificación con partición reproducible y línea base.

## Datos y fuente
- INE Paraguay, Encuesta Permanente de Hogares Continua (EPHC) Anual 2025: https://www.ine.gov.py/microdatos/ (carpeta `EPHC_2025_ANUAL`). Documentación: https://anda.ine.gov.py
- Archivos originales: `REG01`, `REG02`, `INGREFAM` y el diccionario de variables (Fase 1, carpeta `data/`).
- Dataset limpio: 17.092 hogares × 21 variables. Unidad de análisis: el hogar. Objetivo: `TIENE_INTERNET` (1 = sí, 0 = no).
- Datos públicos y anonimizados; uso exclusivamente académico y análisis agregado.

## Estructura del repositorio
| Carpeta | Contenido | Ejecutar |
|---|---|---|
| `Fase1/` | Limpieza, integración y análisis exploratorio | `notebooks/Fase1_AccesoInternet.ipynb` |
| `Fase2/` | Análisis inferencial (5 contrastes, Bonferroni) | `Fase2_AccesoInternet.ipynb` |
| `Fase3/` | Modelado predictivo, modelo `.joblib` y script de inferencia | `notebooks/Fase3_AccesoInternet_ejecutado.ipynb` o `python src/modelado_fase3.py` |

Orden de ejecución: Fase 1 → Fase 2 → Fase 3. La Fase 3 incluye su propia copia del dataset limpio para poder ejecutarse de forma autónoma.

## Tecnologías y reproducibilidad
Python 3.13 (probado), pandas, numpy, matplotlib, seaborn, scipy, statsmodels y scikit-learn. Semilla aleatoria fija: **42**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Síntesis de resultados
- Acceso observado (sin ponderar): 62,1 %. Urbano 79,6 % vs. rural 38,8 %; educación del jefe/a de 36,3 % a 89,2 %; decil 1 de ingreso 22,0 % vs. decil 10 91,5 %.
- Cinco contrastes significativos tras Bonferroni; efectos moderados para área (V = 0,416), educación (V = 0,403) e ingreso (r = 0,460); efecto despreciable para tamaño del hogar (r = 0,038).
- Modelo seleccionado: HistGradientBoosting. En prueba: F1 = 0,845, AUC-ROC = 0,869 (línea base: F1 = 0,766, AUC-ROC = 0,500).
- Limitaciones: métricas no ponderadas (sin diseño muestral complejo), un solo año, error desigual entre áreas (rural 39,2 % de error en hogares conectados vs. urbano 4,8 %). No usar el modelo para negar servicios o derechos.

## Uso de asistentes de IA
Claude y Gemini se usaron como apoyo en depuración de código, consulta del diccionario de variables y redacción. Las decisiones metodológicas y la interpretación son de los autores.
