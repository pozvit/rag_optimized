import sys
import json
import yaml
from pathlib import Path

# запуск как `python src/vector_pipeline.py` из корня проекта
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vector_loader import load_embeddings
from vector_indexer import QdrantIndexer
from vector_validator import validate_index
from vector_searcher import run_search_tests
from vector_manifest import save_manifest


def run_vector_pipeline(config_path: str = "config/vector_store.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['_config_path'] = config_path

    input_file = config['paths']['input_jsonl']
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_size = config['vector_store']['vector_size']

    print("Загрузка эмбеддингов...")
    records, stats = load_embeddings(input_file, vector_size)
    print(f"   строк всего: {stats['total_lines']}, пустых: {stats['blank_lines']}, "
          f"валидных: {stats['valid_records']}, отклонено: {stats['error_records']}")
    if not records:
        raise SystemExit("Нет корректных записей для индексации — индексация прервана.")

    print("Инициализация Qdrant...")
    indexer = QdrantIndexer(config)
    indexer.create_collection()

    print("Загрузка векторов в коллекцию...")
    indexed_count = indexer.index_records(records)
    print(f"   загружено точек: {indexed_count}")

    print("Валидация индекса...")
    validation = validate_index(indexer.client, indexer.collection_name, records, stats, config)
    val_path = output_dir / 'validation.json'
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)
    print(f"   validation сохранён в {val_path}")

    print("Тестовый поиск...")
    search_config = config.get('search', {})
    model = None
    if not search_config.get('use_existing_vectors', True):
        model_name = search_config.get('model_name')
        if not model_name:
            raise ValueError("use_existing_vectors: false требует search.model_name в config/vector_store.yaml")
        # импорт ленивый: в режиме существующих векторов sentence-transformers не нужен
        from sentence_transformers import SentenceTransformer
        used_models = {r['metadata'].get('embedding_model') for r in records}
        if used_models and model_name not in used_models:
            print(f"   ВНИМАНИЕ: запросы кодируются моделью '{model_name}', "
                  f"а векторы в индексе получены моделью(ями) {sorted(used_models)}")
        print(f"   загрузка модели {model_name} для эмбеддинга запросов...")
        model = SentenceTransformer(model_name)
    search_results = run_search_tests(indexer.client, indexer.collection_name, records, config, model)
    search_path = output_dir / 'search_results.json'
    with open(search_path, 'w', encoding='utf-8') as f:
        json.dump(search_results, f, ensure_ascii=False, indent=2)
    print(f"   search_results сохранён в {search_path}")

    print("Сохранение manifest...")
    manifest_path = save_manifest(config, stats, indexed_count, validation, search_results, str(output_dir))
    print(f"   manifest сохранён в {manifest_path}")

    print("\nИтог:")
    print(f"   строк во входном файле : {stats['total_lines']} "
          f"(пустых {stats['blank_lines']}, данных {stats['data_lines']})")
    print(f"   валидных записей       : {stats['valid_records']}")
    print(f"   отклонено записей      : {stats['error_records']}")
    print(f"   загружено в Qdrant     : {indexed_count}")
    print(f"   размерность коллекции  : {validation['vector_size_actual']} "
          f"(метрика {validation['distance_metric_actual']})")
    print(f"   search_works           : {validation['search_works']}")
    print(f"   index_is_valid         : {validation['index_is_valid']}")
    if validation['errors']:
        print("   ошибки валидации:")
        for e in validation['errors']:
            print(f"     - {e}")
    return validation['index_is_valid']


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/vector_store.yaml"
    ok = run_vector_pipeline(cfg)
    sys.exit(0 if ok else 1)
