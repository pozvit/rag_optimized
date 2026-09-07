import yaml
import uuid
from chunk_loader import load_prepared_documents
from chunker import chunk_document
from chunk_exporter import export_chunks

# Определяем пространство имён для стабильных UUID (можно использовать любой константный UUID)
NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID для DNS

def run_chunk_pipeline(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    input_file = config['paths']['input_jsonl']
    output_file = config['paths']['output_jsonl']
    chunk_config = config['chunking']
    strategy = chunk_config['strategy']
    chunk_size = chunk_config['chunk_size']
    chunk_overlap = chunk_config['chunk_overlap']
    split_long = chunk_config.get('split_long_sentences', True)
    
    print(f"Загрузка подготовленных документов из {input_file}")
    documents = load_prepared_documents(input_file)
    print(f"   Загружено {len(documents)} документов.")
    
    all_chunks = []
    total_chunks = 0
    
    for doc in documents:
        doc_id = doc['metadata'].get('document_id', 'unknown')
        text = doc['text']
        # Пробрасываем ВСЕ метаданные документа, а не фиксированный список полей:
        # прикладные поля (document_name, category, version) должны дойти
        # до payload Qdrant и попасть в ответ RAG. Чанк-специфичные ключи
        # ниже перезаписывают одноимённые поля документа.
        original_meta = dict(doc['metadata'])
        chunks_data = chunk_document(text, strategy, chunk_size, chunk_overlap, split_long)
        
        for chunk_info in chunks_data:
            chunk_text = chunk_info['text']
            chunk_idx = chunk_info['chunk_index']
            actual_size = chunk_info['chunk_size']
            
            # Генерируем стабильный UUID на основе document_id и индекса
            chunk_uuid = uuid.uuid5(NAMESPACE, f"{doc_id}_{chunk_idx}")
            chunk_id = str(chunk_uuid)  # строка в формате UUID
            
            chunk_meta = dict(original_meta)
            chunk_meta.update({
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "chunk_size": actual_size,
                "chunk_overlap": chunk_overlap,
                "strategy": strategy,
            })

            chunk_obj = {
                "text": chunk_text,
                "metadata": chunk_meta
            }
            all_chunks.append(chunk_obj)
            total_chunks += 1
    
    print(f"Создано {total_chunks} чанков.")
    print(f"Сохранение результатов в {output_file}")
    export_chunks(all_chunks, output_file)
    print("Готово!")

if __name__ == "__main__":
    run_chunk_pipeline("config/chunking.yaml")