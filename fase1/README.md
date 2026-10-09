# DataScience
Acceso a Internet en los Hogares Paraguayos — Proyecto Integrador de Data Science

Integrantes: Joel Maria Mendieta Candia · Gustavo Fabian Carballo Rojas Fase actual: Fase 1 — Comprensión, preparación y análisis exploratorio Fuente de datos: Encuesta Permanente de Hogares Continua (EPHC) 2025 — INE Paraguay

Pregunta de investigación

¿Qué factores del hogar (área de residencia, departamento, ingreso, educación y ocupación del jefe/a de hogar) se asocian con el acceso a internet en los hogares paraguayos en 2025?

Estructura del repositorio
TP_Data_Science/
├── data/                  # Datos originales de la EPHC 2025 (no modificar)
│   ├── REG01_EPHC_ANUAL_2025.csv
│   ├── REG02_EPHC_ANUAL_2025.csv
│   ├── INGREFAM_EPHC_ANUAL_2025.csv
│   └── diccionario_EPHC_ANUAL_2025.xls
├── notebooks/
│   ├── Fase1_AccesoInternet.ipynb
│   └── Fase1_AccesoInternet.html
├── output/                # Todo lo generado por el notebook
│   ├── dataset_limpio_internet_hogares.csv
│   ├── diccionario_datos_final.csv
│   ├── bitacora_limpieza.csv
│   └── fig1..fig8_*.png
├── informe/                # PDFs de entrega de la Fase 1
│   ├── Documento_comprension_problema.pdf
│   └── Informe_Fase1.pdf
├── requirements.txt
└── README.md
Cómo reproducir el análisis
Clonar el repositorio y ubicarse en la carpeta TP_Data_Science/.
Crear un entorno virtual e instalar dependencias:
bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
Abrir notebooks/Fase1_AccesoInternet.ipynb y ejecutar todas las celdas de principio a fin (Kernel → Restart & Run All). El notebook lee los archivos de data/ con rutas relativas y regenera automáticamente todo lo que hay en output/.
No se requiere ninguna intervención manual ni edición de los CSV en planillas de cálculo: toda transformación está en el código.
Fuente y licencia de los datos

Instituto Nacional de Estadística (INE), Paraguay — Encuesta Permanente de Hogares Continua (EPHC) 2025. Datos públicos de estadística oficial, entregados por la cátedra en septiembre de 2026, uso exclusivamente académico.

Uso de asistentes de IA

Se utilizaron Claude y Gemini como apoyo para explorar el diccionario de variables de la EPHC y corregir errores de código en la limpieza, unión de tablas y visualización tambien para la ayuda en la generación de este README. Ninguna decisión metodológica ni la interpretación de los resultados fue delegada a la herramienta (ver detalle en la portada del notebook).
