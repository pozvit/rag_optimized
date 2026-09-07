import json
from pathlib import Path
from typing import List, Dict, Any

def load_chunks(filepath: str) -> List[Dict[str, Any]]:
    """
    Загружает чанки из JSONL-файла.
    Каждая запись должна содержать поля: text, metadata.
    """
    chunks = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if 'text' not in chunk or 'metadata' not in chunk:
                    print(f"Предупреждение: строка {line_num} не содержит 'text' или 'metadata'")
                    continue
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON в строке {line_num}: {e}")
                continue
    return chunks