import json
from pathlib import Path
from typing import List, Dict, Any


def export_embeddings(embedded_chunks: List[Dict[str, Any]], output_file: str,
                      model_name: str, include_text: bool = True) -> str:
    """
    Сохраняет чанки с embeddings в JSONL-файл.

    model_name   — имя модели из config/embeddings.yaml, записывается в metadata.
    include_text — управляется ключом export.include_text; при False текст чанка
                   в файл не попадает (этап индексации в этом случае работать не будет,
                   так как payload Qdrant строится из текста).
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not include_text:
        print("Предупреждение: export.include_text=false — текст чанков не сохраняется. "
              "Этап vector_pipeline.py потребует поле 'text' и завершится ошибкой.")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in embedded_chunks:
            record = dict(item)
            if 'metadata' in record:
                record['metadata'] = dict(record['metadata'])
                record['metadata']['embedding_model'] = model_name
                record['metadata']['embedding_dimensions'] = len(record['embedding'])
            if not include_text:
                record.pop('text', None)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return str(output_path)
