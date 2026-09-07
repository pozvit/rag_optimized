import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


@lru_cache(maxsize=8)
def _load_metadata_file(metadata_path: str) -> tuple:
    """
    Читает sidecar-файл с метаданными документов (data/raw/metadata.yaml).
    Возвращает хешируемый кортеж (defaults, documents) — так работает lru_cache
    и файл не перечитывается для каждого документа.
    Отсутствие файла не является ошибкой: тогда работают только базовые поля.
    """
    path = Path(metadata_path)
    if not path.exists():
        print(f"Предупреждение: файл метаданных {metadata_path} не найден, "
              f"используются значения по умолчанию.")
        return ((), ())

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    defaults = data.get('defaults', {}) or {}
    documents = data.get('documents', {}) or {}
    return (
        tuple(sorted(defaults.items())),
        tuple((k, tuple(sorted((v or {}).items()))) for k, v in sorted(documents.items())),
    )


def _metadata_for_file(metadata_path: str, file_name: str) -> Dict[str, Any]:
    defaults_t, documents_t = _load_metadata_file(metadata_path)
    defaults = dict(defaults_t)
    documents = {k: dict(v) for k, v in documents_t}

    doc_meta = dict(defaults)
    if file_name in documents:
        doc_meta.update(documents[file_name])
    else:
        if documents:
            print(f"Предупреждение: для файла '{file_name}' нет записи в metadata.yaml, "
                  f"применяются defaults.")
        doc_meta.setdefault('document_name', file_name)
    return doc_meta


def structure_document(raw_doc: Dict[str, Any], config: dict = None) -> Dict[str, Any]:
    """
    Преобразует загруженный документ в единый формат с метаданными.

    Помимо технических полей (source, file_type, document_id) подтягивает
    прикладные метаданные из data/raw/metadata.yaml: document_name, category, version.
    Эти поля требуются заданием: они проходят через чанкинг и эмбеддинги
    и попадают в payload Qdrant, поэтому видны в результатах retrieval и в ответе.
    """
    config = config or {}
    text = raw_doc['text']
    source = raw_doc['source']
    file_type = raw_doc['file_type']

    # document_id — хеш от содержимого и источника: стабилен между запусками,
    # пока сам файл не изменился.
    content = (text + source).encode('utf-8')
    doc_id = hashlib.sha256(content).hexdigest()[:16]

    metadata_path = config.get('paths', {}).get('metadata_file', 'data/raw/metadata.yaml')
    # source может содержать суффикс '#N' (json/csv) — для поиска в sidecar он не нужен
    file_name = Path(source.split('#')[0]).name
    doc_meta = _metadata_for_file(metadata_path, file_name)

    metadata = {
        "source": source,
        "file_name": file_name,
        "file_type": file_type,
        "document_id": doc_id,
        "section": "main",
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "document_name": doc_meta.get('document_name', file_name),
        "category": doc_meta.get('category', 'uncategorized'),
        "version": doc_meta.get('version', 'n/a'),
        "source_type": doc_meta.get('source_type', 'unknown'),
    }

    return {
        "text": text,
        "metadata": metadata
    }
