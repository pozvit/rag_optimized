"""
Служебный скрипт для быстрого просмотра содержимого JSONL-артефактов пайплайна.

Запуск:
    python src/check_results.py                          # чанки и эмбеддинги по умолчанию
    python src/check_results.py data/chunks/chunks.jsonl  # произвольный файл
    python src/check_results.py data/chunks/chunks.jsonl 5
"""
import json
import sys
from pathlib import Path

DEFAULT_FILES = [
    ("Чанки", "data/chunks/chunks.jsonl"),
    ("Эмбеддинги", "data/embeddings/embeddings.jsonl"),
]


def preview_jsonl(filepath: str, n: int = 3) -> None:
    """Печатает первые n записей JSONL-файла в читаемом виде."""
    path = Path(filepath)
    if not path.exists():
        print(f"Файл не найден: {filepath}")
        return
    with open(path, 'r', encoding='utf-8') as f:
        shown = 0
        for line in f:
            if shown >= n:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            shown += 1
            print(f"Запись {shown}:")
            print(f"  текст (первые 100 символов): {data['text'][:100]}...")
            print(f"  метаданные: {data.get('metadata', {})}")
            if 'embedding' in data:
                emb = data['embedding']
                print(f"  размерность embedding: {len(emb)}")
                print(f"  первые 5 значений: {emb[:5]}")
            print("-" * 40)
        if shown == 0:
            print(f"Файл пуст: {filepath}")


def main(argv: list) -> None:
    if argv:
        filepath = argv[0]
        n = int(argv[1]) if len(argv) > 1 else 3
        preview_jsonl(filepath, n)
        return
    for title, filepath in DEFAULT_FILES:
        print(f"=== {title} ===")
        preview_jsonl(filepath, 2)


if __name__ == "__main__":
    main(sys.argv[1:])
