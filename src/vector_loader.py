import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

REQUIRED_META = ['chunk_id', 'document_id', 'embedding_model', 'embedding_dimensions']


def load_embeddings(filepath: str, expected_vector_size: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Загружает данные из embeddings.jsonl.

    Проверяет: существование файла, корректность JSON, наличие text / embedding / metadata,
    наличие chunk_id, document_id, embedding_model, embedding_dimensions,
    непустой вектор, отсутствие NaN/Inf, совпадение фактической размерности
    с конфигом и с заявленной в metadata.

    Возвращает (records, stats), где stats — согласованная статистика по файлу:
        total_lines  = blank_lines + data_lines
        data_lines   = valid_records + error_records
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {filepath}. "
            f"Сначала выполните этап embeddings: python src/embed_pipeline.py"
        )

    records: List[Dict[str, Any]] = []
    stats = {
        'total_lines': 0,
        'blank_lines': 0,
        'data_lines': 0,
        'valid_records': 0,
        'error_records': 0,
    }
    error_details: List[str] = []

    def fail(line_num: int, message: str) -> None:
        text = f"строка {line_num}: {message}"
        print(f"Пропуск {text}")
        stats['error_records'] += 1
        error_details.append(text)

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, raw_line in enumerate(f, 1):
            stats['total_lines'] += 1
            line = raw_line.strip()
            if not line:
                # пустая строка учитывается отдельно и не считается ни валидной записью, ни ошибкой
                stats['blank_lines'] += 1
                continue
            stats['data_lines'] += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                fail(line_num, f"некорректный JSON ({e})")
                continue

            if not isinstance(data, dict):
                fail(line_num, "запись не является JSON-объектом")
                continue
            if 'text' not in data:
                fail(line_num, "отсутствует 'text'")
                continue
            if 'embedding' not in data:
                fail(line_num, "отсутствует 'embedding'")
                continue
            if 'metadata' not in data or not isinstance(data['metadata'], dict):
                fail(line_num, "отсутствует или некорректен 'metadata'")
                continue

            meta = data['metadata']
            missing = [f for f in REQUIRED_META if f not in meta or meta[f] in (None, '')]
            if missing:
                fail(line_num, f"отсутствуют поля в metadata: {missing}")
                continue

            emb = data['embedding']
            if not isinstance(emb, list) or not emb:
                fail(line_num, "embedding не является непустым списком")
                continue
            # bool — подкласс int, поэтому проверяем его отдельно
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in emb):
                fail(line_num, "embedding содержит нечисловые элементы")
                continue
            if not all(math.isfinite(float(x)) for x in emb):
                fail(line_num, "embedding содержит NaN, Infinity или -Infinity")
                continue
            if len(emb) != expected_vector_size:
                fail(line_num, f"размерность {len(emb)} не соответствует vector_size={expected_vector_size} из конфига")
                continue
            if meta['embedding_dimensions'] != len(emb):
                fail(line_num,
                     f"metadata.embedding_dimensions={meta['embedding_dimensions']} "
                     f"не совпадает с фактической размерностью {len(emb)}")
                continue

            records.append(data)
            stats['valid_records'] += 1

    # контроль согласованности счётчиков
    assert stats['total_lines'] == stats['blank_lines'] + stats['data_lines']
    assert stats['data_lines'] == stats['valid_records'] + stats['error_records']

    stats['error_details'] = error_details[:20]
    return records, stats
