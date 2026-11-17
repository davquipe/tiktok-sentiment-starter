"""Inferencia usando un modelo entrenado en textos de redes sociales en español."""

from dataclasses import dataclass

MODEL_ID = "pysentimiento/robertuito-sentiment-analysis"
LABELS = ("NEG", "NEU", "POS")


@dataclass(frozen=True, slots=True)
class Sentiment:
    label: str
    confidence: float
    probabilities: dict[str, float]


def classify(texts: list[str], model_id: str = MODEL_ID, batch_size: int = 16) -> list[Sentiment]:
    """Descarga el modelo en el primer uso; ejecuta inferencia local en CPU."""
    if not texts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size debe ser positivo")
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError("Instala la dependencia NLP: pip install -e '.[nlp]' y PyTorch") from error

    classifier = pipeline(
        task="text-classification",
        model=model_id,
        tokenizer=model_id,
        device=-1,
    )
    raw_outputs = classifier(
        texts,
        top_k=None,
        truncation=True,
        max_length=128,
        batch_size=batch_size,
    )
    if len(raw_outputs) != len(texts):
        raise RuntimeError("La cantidad de predicciones no coincide con las entradas")

    sentiments: list[Sentiment] = []
    for scores in raw_outputs:
        probabilities = {str(result["label"]): float(result["score"]) for result in scores}
        if set(probabilities) != set(LABELS):
            raise ValueError(f"Etiquetas inesperadas del modelo: {sorted(probabilities)}")
        label = max(LABELS, key=lambda item: probabilities[item])
        sentiments.append(
            Sentiment(label=label, confidence=probabilities[label], probabilities=probabilities)
        )
    return sentiments
