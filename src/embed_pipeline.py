import yaml
from embed_loader import load_chunks
from embedder import Embedder
from embed_exporter import export_embeddings

def run_embed_pipeline(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    input_file = config['paths']['input_jsonl']
    output_file = config['paths']['output_jsonl']
    model_name = config['embedding']['model_name']
    provider = str(config['embedding'].get('provider', 'sentence-transformers')).strip().lower()
    declared_dimensions = config['embedding'].get('dimensions')
    batch_size = config['embedding'].get('batch_size', 32)
    max_tokens = config['embedding'].get('max_tokens', None)
    include_text = config.get('export', {}).get('include_text', True)

    if provider != 'sentence-transformers':
        raise ValueError(
            f"Неподдерживаемый embedding.provider='{provider}' в config/embeddings.yaml. "
            f"Реализован только 'sentence-transformers'."
        )
    
    print(f"Загрузка чанков из {input_file}")
    chunks = load_chunks(input_file)
    print(f"   Загружено {len(chunks)} чанков.")
    
    if not chunks:
        print("Нет данных для обработки. Выход.")
        return
    
    passage_prefix = config['embedding'].get('passage_prefix', '')
    normalize = config['embedding'].get('normalize_embeddings', True)

    print(f"Инициализация модели {model_name}...")
    embedder = Embedder(model_name, batch_size, max_tokens,
                        passage_prefix=passage_prefix, normalize=normalize)
    print(f"   Размерность вектора: {embedder.dimensions}")

    # Заявленная в конфиге размерность должна совпадать с фактической размерностью модели,
    # иначе vector_store.yaml будет настроен на неверный vector_size.
    if declared_dimensions is not None and embedder.dimensions != declared_dimensions:
        raise ValueError(
            f"embedding.dimensions={declared_dimensions} в config/embeddings.yaml "
            f"не совпадает с фактической размерностью модели '{model_name}' "
            f"({embedder.dimensions}). Исправьте конфиг или смените модель."
        )
    
    print("Расчёт embeddings...")
    embedded_chunks = embedder.embed_chunks(chunks)
    
    print(f"Сохранение результатов в {output_file}")
    export_embeddings(embedded_chunks, output_file, model_name, include_text)
    
    print("Готово!")
    print("Статистика:")
    print(f"   - Всего чанков: {len(embedded_chunks)}")
    print(f"   - Размерность векторов: {len(embedded_chunks[0]['embedding'])}")
    print(f"   - Модель: {model_name}")

if __name__ == "__main__":
    run_embed_pipeline("config/embeddings.yaml")