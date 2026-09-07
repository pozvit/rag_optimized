from typing import List, Dict
from qdrant_client import QdrantClient

REQUIRED_PAYLOAD = ['chunk_id', 'document_id', 'text', 'source', 'section',
                    'embedding_model', 'embedding_dimensions']


def validate_index(client: QdrantClient, collection_name: str, input_records: List[Dict],
                   stats: Dict, config: dict) -> dict:
    """
    Проверяет индекс по фактическим данным Qdrant, а не по конфигу:
    количество точек, реальная размерность и метрика коллекции,
    содержимое payload у выборки точек и работоспособность search-запроса.
    """
    expected_count = len(input_records)
    info = client.get_collection(collection_name)
    actual_count = info.points_count
    vector_size = config['vector_store']['vector_size']
    distance = config['vector_store'].get('distance', 'Cosine')

    results = {
        # статистика по входному файлу (total_lines = blank_lines + data_lines,
        # data_lines = valid_records + error_records)
        'total_lines_in_file': stats['total_lines'],
        'blank_lines_in_file': stats['blank_lines'],
        'data_lines_in_file': stats['data_lines'],
        'valid_records_loaded': stats['valid_records'],
        'error_records_in_file': stats['error_records'],
        'counters_consistent': (
            stats['total_lines'] == stats['blank_lines'] + stats['data_lines']
            and stats['data_lines'] == stats['valid_records'] + stats['error_records']
        ),
        'all_input_records_valid': stats['error_records'] == 0,
        # сравнение с индексом
        'actual_points_in_qdrant': actual_count,
        'count_match': expected_count == actual_count,
        'count_match_vs_data_lines': stats['data_lines'] == actual_count,
        # параметры коллекции: ожидание из конфига против факта из Qdrant
        'vector_size_expected': vector_size,
        'vector_size_actual': info.config.params.vectors.size,
        'vector_size_match': info.config.params.vectors.size == vector_size,
        'distance_metric_expected': distance,
        'distance_metric_actual': info.config.params.vectors.distance.name,
        'distance_metric_match': info.config.params.vectors.distance.name == str(distance).upper(),
        'all_embeddings_have_same_dimension': True,
        # payload
        'payload_checked_points': 0,
        'payload_fields_present': True,
        'payload_required_fields': REQUIRED_PAYLOAD,
        # поиск
        'search_works': False,
        'search_returned_hits': 0,
        'errors': []
    }

    if stats['error_records']:
        results['errors'].append(
            f"Во входном файле отклонено записей: {stats['error_records']} "
            f"(см. error_details в manifest)"
        )
    if not results['counters_consistent']:
        results['errors'].append("Счётчики по входному файлу не сходятся")
    if not results['count_match']:
        results['errors'].append(
            f"Количество точек в Qdrant ({actual_count}) не совпадает с числом валидных записей ({expected_count})"
        )
    if not results['vector_size_match']:
        results['errors'].append(
            f"Размерность коллекции ({results['vector_size_actual']}) не совпадает с конфигом ({vector_size})"
        )
    if not results['distance_metric_match']:
        results['errors'].append(
            f"Метрика коллекции ({results['distance_metric_actual']}) не совпадает с конфигом ({distance})"
        )

    # размерности векторов во входном файле
    dims = {len(rec['embedding']) for rec in input_records}
    if len(dims) != 1:
        results['all_embeddings_have_same_dimension'] = False
        results['errors'].append(f"Разные размерности векторов во входном файле: {sorted(dims)}")
    elif next(iter(dims)) != vector_size:
        results['errors'].append(
            f"Размерность во входном файле ({next(iter(dims))}) не совпадает с конфигом ({vector_size})"
        )

    # payload реальных точек, поднятых из Qdrant по chunk_id
    sample_ids = [rec['metadata']['chunk_id'] for rec in input_records[:5]]
    if sample_ids:
        points = client.retrieve(collection_name=collection_name, ids=sample_ids, with_payload=True)
        results['payload_checked_points'] = len(points)
        if len(points) != len(sample_ids):
            results['payload_fields_present'] = False
            results['errors'].append(
                f"По chunk_id найдено {len(points)} точек из {len(sample_ids)} — нарушена связь chunk_id -> point"
            )
        for point in points:
            payload = point.payload or {}
            if not payload:
                results['payload_fields_present'] = False
                results['errors'].append(f"Точка {point.id} не имеет payload")
                continue
            missing = [f for f in REQUIRED_PAYLOAD if f not in payload]
            if missing:
                results['payload_fields_present'] = False
                results['errors'].append(f"Точка {point.id}: в payload нет полей {missing}")
            if str(payload.get('chunk_id')) != str(point.id):
                results['payload_fields_present'] = False
                results['errors'].append(
                    f"Точка {point.id}: payload.chunk_id={payload.get('chunk_id')} не совпадает с id точки"
                )
    else:
        results['errors'].append("Нет точек для проверки payload")

    # фактическая проверка возможности выполнить search-запрос
    if input_records:
        try:
            test_vector = input_records[0]['embedding']
            if hasattr(client, 'query_points'):
                hits = client.query_points(
                    collection_name=collection_name, query=test_vector,
                    limit=1, with_payload=True
                ).points
            else:
                hits = client.search(
                    collection_name=collection_name, query_vector=test_vector,
                    limit=1, with_payload=True
                )
            results['search_returned_hits'] = len(hits)
            results['search_works'] = len(hits) > 0
            if not hits:
                results['errors'].append("Поиск выполнен, но не вернул ни одного результата")
        except Exception as e:
            results['errors'].append(f"Ошибка при выполнении тестового поиска: {e}")

    results['index_is_valid'] = (
        results['count_match']
        and results['vector_size_match']
        and results['distance_metric_match']
        and results['payload_fields_present']
        and results['search_works']
        and results['all_input_records_valid']
        and not results['errors']
    )
    return results
