import json
from pathlib import Path
from typing import List, Dict, Any

def load_prepared_documents(filepath: str) -> List[Dict[str, Any]]:
    """
    Загружает документы из JSONL-файла, подготовленного первым пайплайном.
    Каждая запись содержит поля: text, metadata (с document_id и др.)
    """
    documents = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                # Ожидаем структуру: {"text": "...", "metadata": {...}}
                if 'text' in doc and 'metadata' in doc:
                    documents.append(doc)
                else:
                    print(f"Пропущена запись без 'text' или 'metadata': {line[:100]}")
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON: {e} в строке: {line[:100]}")
                continue
    return documents