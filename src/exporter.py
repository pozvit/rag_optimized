import json
from pathlib import Path
from typing import List, Dict, Any

SUPPORTED_FORMATS = ('jsonl', 'json')

DEFAULT_FILENAMES = {
    'jsonl': 'prepared_documents.jsonl',
    'json': 'prepared_documents.json',
}


def export_documents(documents: List[Dict[str, Any]], output_dir: str, config: dict) -> str:
    """
    Сохраняет документы в формате, заданном в config/config.yaml (секция export).

    format: "jsonl" — по одному JSON-объекту в строке (используется следующими этапами);
            "json"  — один JSON-массив со всеми документами.

    Возвращает путь к сохранённому файлу.
    """
    fmt = str(config.get('format', 'jsonl')).strip().lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Недопустимый export.format='{config.get('format')}' в config/config.yaml. "
            f"Допустимые значения: {', '.join(SUPPORTED_FORMATS)}."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = config.get('filename') or DEFAULT_FILENAMES[fmt]
    filepath = output_path / filename

    expected_suffix = '.' + fmt
    if filepath.suffix.lower() != expected_suffix:
        print(f"Предупреждение: export.format='{fmt}', "
              f"а export.filename='{filename}' имеет расширение '{filepath.suffix}'.")

    with open(filepath, 'w', encoding='utf-8') as f:
        if fmt == 'jsonl':
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        else:
            json.dump(documents, f, ensure_ascii=False, indent=2)

    return str(filepath)
