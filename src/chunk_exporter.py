import json
from pathlib import Path
from typing import List, Dict, Any

def export_chunks(chunks: List[Dict[str, Any]], output_file: str):
    """
    Сохраняет чанки в JSONL-файл.
    Каждая строка – JSON-объект с полями text и metadata.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')