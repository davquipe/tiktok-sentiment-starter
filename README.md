# TikTok Sentiment Analysis — Starter (Ubuntu)

MVP de análisis de sentimientos en español. El proyecto **no extrae automáticamente datos de TikTok**. Lee un CSV autorizado o sintético, ejecuta localmente `pysentimiento/robertuito-sentiment-analysis` y exporta resultados reproducibles.

## Requisitos

- Ubuntu 22.04/24.04/26.04 o entorno Linux compatible
- Python **3.11 o 3.12 recomendado** (otras versiones recientes pueden funcionar con ruedas compatibles)
- Internet en el primer arranque para descargar el modelo
- CPU; **no requiere GPU**. El modelo ocupa cientos de MB y necesita memoria RAM.

## Instalación

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip
# Si tu distribución no tiene Python 3.11 como python3, usa python3.11 en los siguientes comandos.
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# PyTorch para CPU, evitando librerías CUDA innecesarias:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[nlp]'
```

## Verificar el pipeline de datos (sin descargar el modelo)

```bash
python -m unittest discover -s tests -v
```

## Ejecutar análisis de sentimientos

```bash
tiktok-sentiment --input data/raw/comments_sample.csv
# o: python -m tiktok_sentiment.cli --input data/raw/comments_sample.csv
```

**Resultados:**

- `outputs/predictions.csv`: comentario, clase y probabilidades POS/NEU/NEG.
- `outputs/summary.json`: totales y porcentajes globales y por video.

> La primera ejecución descarga el modelo desde Hugging Face. No hemos podido validar esa descarga y la inferencia real en este entorno offline; el pipeline de datos y reportes tiene pruebas offline incluidas.

## Formato de entrada

```csv
id,video_id,text,create_time,like_count
12345,789,"Excelente video",1784000000,4
12346,789,"No me convenció",1784001000,1
```

Obligatorias: `id`, `video_id`, `text`. Opcionales: `create_time`, `like_count`. Los IDs se manejan como **texto**, evitando pérdidas de precisión en identificadores de 64 bits.

Los valores de `data/raw/comments_sample.csv` son completamente ficticios y solo sirven para probar el flujo de trabajo. No son datos de usuarios reales.

## Límites metodológicos

- La salida indica **polaridad estimada**, no emociones detalladas. La confianza es la probabilidad asignada por el modelo, no una garantía de acierto.
- El modelo se entrenó en texto de redes sociales en español, particularmente tweets; no está calibrado específicamente para comentarios peruanos de TikTok.
- No eliminar emojis ni la negación al limpiar texto. Casos como sarcasmo, ironía, jergas y mezclas de idiomas requieren validación humana.
- Un porcentaje sobre los comentarios analizados **no representa** automáticamente la opinión de los usuarios de TikTok, una ciudad o un país.
- Para un estudio serio, crea una muestra etiquetada manualmente (p.ej., 200–500 comentarios según recursos) y evalúa macro-F1, matriz de confusión y errores por categoría.
- No exportes identificadores de personas o datos sensibles innecesarios. Obtén datos solo por vías autorizadas.

## Roadmap

1. Dataset + inferencia local (esta versión).
2. Dataset etiquetado y evaluación (macro-F1, matriz de confusión).
3. Dashboards (Streamlit o React/TypeScript + FastAPI).
4. Ingesta autorizada con paginación, deduplicación y almacenamiento Parquet.
5. Pipeline por lotes con Airflow; Spark cuando el tamaño lo justifique.

## Fase 2 — Evaluación objetiva (v0.2)

**Meta:** distinguir que el modelo ejecuta de que el modelo **clasifica bien** comentarios en español peruano. La evaluación se hace contra etiquetas humanas, no contra las probabilidades del mismo modelo.

### Camino rápido con los 18 comentarios ficticios

Si todavía no ejecutaste la inferencia, hazlo primero:

```bash
tiktok-sentiment --input data/raw/comments_sample.csv
```

Actualiza los comandos del proyecto luego de descomprimir la versión 0.2:

```bash
python -m pip install -e '.[nlp]'
python -m unittest discover -s tests -v
tiktok-evaluate \
  --predictions outputs/predictions.csv \
  --labels data/labeled/comments_labeled_sample.csv \
  --output outputs/evaluation
```

Los resultados quedan en:

- `outputs/evaluation/metrics.json`: accuracy, macro-F1 y precision/recall/F1 de cada clase.
- `outputs/evaluation/confusion_matrix.csv`: **filas = etiquetas humanas, columnas = predicciones**.
- `outputs/evaluation/errors.csv`: errores del modelo, ordenados por confianza descendente para revisión manual.

### Dataset propio con etiquetado ciego

```bash
tiktok-make-labels \
  --input data/raw/comments_sample.csv \
  --output data/labeled/to_label.csv
```

Abre `to_label.csv` en una hoja de cálculo y completa `gold_label` **sin consultar los resultados del modelo**. Opciones:

| gold_label | Cuándo utilizar |
|---|---|
| POS | Expresa una valoración principalmente favorable o satisfacción. |
| NEG | Expresa una valoración principalmente desfavorable o insatisfacción. |
| NEU | Preguntas o afirmaciones descriptivas sin valoración marcada. |
| SKIP | No se puede interpretar con suficiente certeza, depende de contexto ausente o contiene polaridad irreconciliablemente mixta. |
| vacío | No anotado aún; se excluye de la evaluación. |

La columna `notes` permite justificar decisiones de casos difíciles. No cambies `id`, `video_id` ni `text` después de generar las predicciones. Se validan IDs, texto y video para evitar comparar versiones distintas del dataset.

```bash
tiktok-evaluate --predictions outputs/predictions.csv \
  --labels data/labeled/to_label.csv \
  --output outputs/evaluation-personal
```

### Reglas metodológicas

- El dataset de 18 comentarios es **sintético y demostrativo**, con etiquetas asignadas manualmente. Sus métricas no demuestran eficacia en TikTok real.
- Para medir rendimiento serio en la población objetivo, recolecta legalmente comentarios auténticos y selecciona una **muestra no sesgada** de aproximadamente 200–500 como punto de partida, idealmente distribuida entre distintos videos/temas.
- Recomendada revisión independiente de dos personas en una submuestra; mide acuerdo y resuelve desacuerdos con reglas documentadas.
- Identifica modismos peruanos, ironía, sarcasmo, emojis y textos mixtos. Usa `SKIP` cuando falte contexto, no para ocultar ejemplos difíciles que el modelo clasifique mal.
- Macro-F1 promedia las tres clases y es útil cuando están desbalanceadas. Accuracy sola puede ocultar errores sobre una clase minoritaria.
- Evita incluir datos personales no necesarios, y no publiques comentarios identificables sin verificar los permisos y reglas aplicables.


## Fase 3 — Analizar comentarios reales de un video (v0.3)

No se incluyen comentarios reales, ni se extraen automáticamente desde TikTok. Primero obtiene legalmente una exportación CSV/JSON con permiso de uso. Puedes probar con [este exportador de Apify](https://apify.com/xmolodtsov/tiktok-comments-scraper), sujeto a sus condiciones, límites y posibles costos. **No compartas tokens ni credenciales.**

Video de prueba elegido: `https://www.tiktok.com/@keikofujimorih/video/7707793734842240277`.

1. Abre el proveedor, introduce el video en `postURLs` y limita inicialmente `commentsPerPost` a 100. Desactiva la captura de perfiles de personas si aparece la opción. Descarga el dataset como CSV o JSON. Comprueba las reglas de TikTok y las condiciones del proveedor antes de utilizarlo.
2. Coloca la exportación como `data/raw/apify_export.csv` (o `apify_export.json`).
3. Desde el directorio raíz con `.venv` activo:

```bash
python -m pip install -e '.[nlp]'
tiktok-import --input data/raw/apify_export.csv \
  --video-url 'https://www.tiktok.com/@keikofujimorih/video/7707793734842240277' \
  --output data/raw/video_7707793734842240277.csv
tiktok-sentiment --input data/raw/video_7707793734842240277.csv \
  --output outputs/video_7707793734842240277/predictions.csv \
  --summary outputs/video_7707793734842240277/summary.json
python -m json.tool outputs/video_7707793734842240277/summary.json
```

Si descargas JSON, cambia `apify_export.csv` por `apify_export.json`. El importador conserva solo ID de comentario, ID de video, texto, fecha y likes; elimina nombres, perfiles y avatares. Valida el ID del video, evita duplicados y omite filas de error.

**Interpretación:** los resultados describen solo los comentarios obtenidos, no la opinión pública ni la intención de voto. Las respuestas pueden estar filtradas o priorizadas por TikTok; una descarga limitada a 100 no constituye una muestra aleatoria. El sentimiento del texto también puede referirse a un tercero y no necesariamente a la persona que publicó el video.
