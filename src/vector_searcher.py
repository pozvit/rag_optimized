from qdrant_client import QdrantClient
from typing import List, Dict, Any, Optional

# Пороги для вердикта о релевантности (cosine similarity, all-MiniLM-L6-v2)
HIGH_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.40


def _judge(score: Optional[float]) -> str:
    if score is None:
        return 'no_results'
    if score >= HIGH_THRESHOLD:
        return 'high'
    if score >= MEDIUM_THRESHOLD:
        return 'medium'
    return 'low'


def perform_search(client, collection_name, vector, top_k) -> List[Dict[str, Any]]:
    """
    Выполняет поиск, поддерживая новое (query_points, qdrant-client >= 1.10)
    и старое (search) API клиента.
    """
    if hasattr(client, 'query_points'):
        hits = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=top_k,
            with_payload=True
        ).points
    else:
        hits = client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True
        )
    return [
        {
            'id': hit.id,
            'score': hit.score,
            'chunk_id': (hit.payload or {}).get('chunk_id'),
            'document_id': (hit.payload or {}).get('document_id'),
            'payload': hit.payload
        }
        for hit in hits
    ]


def run_search_tests(client: QdrantClient, collection_name: str, records: List[Dict],
                     config: dict, model=None) -> List[Dict]:
    """
    Тестовый retrieval в двух режимах.

    use_existing_vectors: true  — берём вектор конкретного чанка, текст запроса и test_id
                                  относятся к этому же чанку (честная связка).
                                  Вердикт о релевантности выносится по лучшему результату
                                  БЕЗ учёта self-match: сам чанк всегда найдётся со score ~1.0
                                  и ничего не доказывает.
    use_existing_vectors: false — считаем embedding текстового запроса той же моделью.
    """
    top_k = config['search'].get('top_k', 5)
    queries = config['search'].get('test_queries', [])
    use_existing = config['search'].get('use_existing_vectors', True)
    search_results: List[Dict] = []

    if use_existing:
        print("Тестовый поиск по существующим векторам...")
        sample_records = records[:min(3, len(records))]
        for rec in sample_records:
            chunk_id = rec['metadata']['chunk_id']
            hits = perform_search(client, collection_name, rec['embedding'], top_k)

            self_match = next((h for h in hits if str(h['id']) == str(chunk_id)), None)
            others = [h for h in hits if str(h['id']) != str(chunk_id)]
            best_other = others[0]['score'] if others else None
            verdict = _judge(best_other)

            search_results.append({
                'test_id': chunk_id,
                'query_type': 'existing_vector',
                'source_chunk_id': chunk_id,
                'query_text': rec['text'],
                'self_match_found': self_match is not None,
                'self_match_score': self_match['score'] if self_match else None,
                'best_other_score': best_other,
                'relevance_judgment': verdict,
                'relevance_comment': (
                    "Исходный чанк найден первым (self-match), "
                    f"лучший сторонний результат score={best_other:.4f} — вердикт '{verdict}'."
                    if self_match and best_other is not None else
                    "Исходный чанк не найден по собственному вектору — индекс некорректен."
                    if not self_match else
                    "Кроме самого чанка ничего не найдено."
                ),
                'results': hits
            })
    else:
        if model is None:
            raise RuntimeError(
                "use_existing_vectors: false требует модели для эмбеддинга запросов. "
                "Проверьте search.model_name в config/vector_store.yaml."
            )
        print(f"Тестовый поиск по {len(queries)} текстовым запросам...")
        for idx, query in enumerate(queries, 1):
            query_emb = model.encode([query], convert_to_numpy=True)[0].tolist()
            hits = perform_search(client, collection_name, query_emb, top_k)
            best = hits[0]['score'] if hits else None
            verdict = _judge(best)
            search_results.append({
                'test_id': f"query_{idx}",
                'query_type': 'text_query',
                'source_chunk_id': None,
                'query_text': query,
                'top_chunk_id': hits[0]['chunk_id'] if hits else None,
                'best_other_score': best,
                'relevance_judgment': verdict,
                'relevance_comment': (
                    f"Лучший результат score={best:.4f} — вердикт '{verdict}'."
                    if best is not None else "Поиск не вернул результатов."
                ),
                'results': hits
            })
    return search_results
