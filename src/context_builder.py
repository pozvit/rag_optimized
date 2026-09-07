"""
Context builder: превращает найденные чанки в единый текстовый контекст для LLM.

Формат намеренно жёсткий и «нумерованный» — так модель может ссылаться
на конкретные фрагменты, а пользователь видит, из какого документа что взято:

    [Фрагмент 1]
    Источник: IBP — Интегрированное бизнес-планирование (версия 2025-06, категория products/integrated_planning)
    Файл: 04_ibp_integrated_business_planning.pdf | чанк 12 | score 0.8712
    Текст: ...
"""

from typing import Any, Dict, List, Tuple


def _truncate(text: str, limit: int) -> str:
    text = (text or '').strip()
    if limit and len(text) > limit:
        return text[:limit].rsplit(' ', 1)[0] + ' […]'
    return text


def build_context(chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Возвращает (context_text, used_chunks).

    used_chunks — это те фрагменты, что реально попали в контекст: если сработал
    лимит max_context_chars, часть чанков отбрасывается, и ответ не должен
    ссылаться на них как на использованные.
    """
    ctx_cfg = config.get('context', {})
    max_chars_per_chunk = int(ctx_cfg.get('max_chars_per_chunk', 1200))
    max_context_chars = int(ctx_cfg.get('max_context_chars', 6000))
    include_metadata = bool(ctx_cfg.get('include_metadata', True))

    if not chunks:
        return "", []

    blocks: List[str] = []
    used: List[Dict[str, Any]] = []
    total = 0

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get('metadata', {}) or {}
        text = _truncate(chunk.get('text', ''), max_chars_per_chunk)

        lines = [f"[Фрагмент {i}]"]
        if include_metadata:
            lines.append(
                f"Источник: {meta.get('document_name', 'неизвестно')} "
                f"(версия {meta.get('version', 'n/a')}, категория {meta.get('category', 'n/a')})"
            )
            lines.append(
                f"Файл: {meta.get('file_name', meta.get('source', 'n/a'))} | "
                f"чанк {meta.get('chunk_index', '?')} | score {chunk.get('score', '?')}"
            )
        lines.append(f"Текст: {text}")
        block = "\n".join(lines)

        if max_context_chars and total + len(block) > max_context_chars and used:
            break

        blocks.append(block)
        used.append({**chunk, 'context_position': i})
        total += len(block) + 2

    return "\n\n".join(blocks), used


def format_chunks_for_display(chunks: List[Dict[str, Any]], preview_chars: int = 300) -> str:
    """Человекочитаемый список использованных фрагментов — для CLI и отчёта."""
    if not chunks:
        return "Фрагменты не найдены."

    parts = []
    for chunk in chunks:
        meta = chunk.get('metadata', {}) or {}
        parts.append(
            f"[Фрагмент {chunk.get('context_position', chunk.get('rank'))}] "
            f"score={chunk.get('score')} | {meta.get('document_name', '—')} | "
            f"{meta.get('file_name', '—')} | chunk_index={meta.get('chunk_index', '?')} | "
            f"chunk_id={chunk.get('chunk_id')}\n"
            f"    {_truncate(chunk.get('text', ''), preview_chars)}"
        )
    return "\n".join(parts)
