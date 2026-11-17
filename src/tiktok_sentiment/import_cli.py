"""Comando para importar comentarios CSV/JSON exportados por un proveedor."""

import argparse
from pathlib import Path

from .import_apify import import_comments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte exportaciones CSV/JSON de Apify al formato del pipeline NLP"
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV o JSON descargado")
    parser.add_argument("--video-url", required=True, help="Enlace original del video")
    parser.add_argument("--output", required=True, type=Path, help="CSV compatible de salida")
    args = parser.parse_args()
    try:
        result = import_comments(args.input, args.output, args.video_url)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Video: {result['video_id']}")
    print(f"Comentarios importados: {result['imported']} (omitidos: {result['skipped']})")
    print(f"CSV compatible: {result['output']}")


if __name__ == "__main__":
    main()
