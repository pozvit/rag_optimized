"""
Retrieval: вопрос пользователя -> top-k релевантных чанков из Qdrant.

top_k — параметр (по умолчанию берётся из config/rag.yaml, retrieval.top_k = 3),
результат содержит текст чанка и полные метаданные, чтобы вызывающий код
мог показать пользователю, на чём основан ответ.
"""

from typing import Any, Dict, List

from qdrant_client import QdrantClient

from embedder import Embedder


class Retriever:
    def __init__(self, config: Dict[str, Any]):
        retrieval_cfg = config.get('retrieval', {})
        paths_cfg = config.get('paths', {})

        self.collection_name = retrieval_cfg.get('collection_name', 'rag_documents')
        self.default_top_k = int(retrieval_cfg.get('top_k', 3))
        self.score_threshold = float(retrieval_cfg.get('score_threshold', 0.0))
        self.query_prefix = retrieval_cfg.get('query_prefix', '')
        self.model_name = retrieval_cfg['model_name']

        storage_path = paths_cfg.get('qdrant_storage', 'data/qdrant_storage')
        self.client = QdrantClient(path=storage_path)

        self._assert_collection_exists()

        self.embedder = Embedder(
            self.model_name,
            batch_size=1,
            max_tokens=retrieval_cfg.get('max_tokens', 512),
            passage_prefix=self.query_prefix,
            normalize=retrieval_cfg.get('normalize_embeddings', True),
        )
        self._assert_dimension_matches()

    def _assert_collection_exists(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            raise RuntimeError(
                f"Коллекция '{self.collection_name}' не найдена в Qdrant. "
                f"Сначала выполните индексацию: python src/vector_pipeline.py. "
                f"Найдено: {existing or 'ничего'}."
            )

    def _assert_dimension_matches(self) -> None:
        info = self.client.get_collection(self.collection_name)
        collection_dim = info.config.params.vectors.size
        if collection_dim != self.embedder.dimensions:
            raise ValueError(
                f"Размерность модели запроса '{self.model_name}' ({self.embedder.dimensions}) "
                f"не совпадает с размерностью коллекции '{self.collection_name}' ({collection_dim}). "
                f"Модель retrieval обязана совпадать с моделью этапа embeddings."
            )

    def embed_query(self, question: str) -> List[float]:
        return self.embedder.embed_texts([question])[0]

    def retrieve(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Возвращает список найденных чанков, отсортированных по убыванию score:
        [{'rank', 'score', 'chunk_id', 'document_id', 'text', 'metadata'}, ...]
        """
        k = int(top_k or self.default_top_k)
        if k <= 0:
            raise ValueError(f"top_k должен быть положительным, получено {k}")

        vector = self.embed_query(question)

        if hasattr(self.client, 'query_points'):
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=k,
                with_payload=True,
                score_threshold=self.score_threshold or None,
            ).points
        else:  # qdrant-client < 1.10
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=k,
                with_payload=True,
                score_threshold=self.score_threshold or None,
            )

        results = []
        for rank, hit in enumerate(hits, start=1):
            payload = dict(hit.payload or {})
            text = payload.pop('text', '')
            results.append({
                'rank': rank,
                'score': round(float(hit.score), 4),
                'chunk_id': payload.get('chunk_id'),
                'document_id': payload.get('document_id'),
                'text': text,
                'metadata': payload,
            })
        return results

    def close(self) -> None:
        # Qdrant в local mode держит блокировку на каталоге хранилища
        try:
            self.client.close()
        except Exception:
            pass
