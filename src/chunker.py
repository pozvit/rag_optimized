import re
from typing import List, Dict, Any


def split_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения (упрощённо)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]


def split_paragraphs(text: str) -> List[str]:
    """Разбивает текст на абзацы (по двойному переводу строки)."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p for p in paragraphs if p.strip()]


def split_long_sentence(sentence: str, chunk_size: int) -> List[str]:
    """
    Fallback для предложения длиннее chunk_size: режем по словам,
    не разрывая слова. Без этого chunk_size перестаёт быть реальным ограничением.
    """
    words = sentence.split()
    parts: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        add_len = len(word) + (1 if current else 0)
        if current and current_len + add_len > chunk_size:
            parts.append(' '.join(current))
            current, current_len = [word], len(word)
        else:
            current.append(word)
            current_len += add_len
    if current:
        parts.append(' '.join(current))
    return parts or [sentence]


def chunk_by_sentences(text: str, chunk_size: int, overlap: int,
                       split_long: bool = True) -> List[str]:
    """
    Разбивает текст на чанки по предложениям.
    chunk_size — целевой размер в символах, overlap — перекрытие в символах.
    split_long=True: предложение длиннее chunk_size дополнительно режется по словам.
    """
    raw_sentences = split_sentences(text)
    if not raw_sentences:
        return []

    # длинное предложение режем до (chunk_size - overlap), чтобы итоговый чанк
    # вместе с перекрытием предыдущего чанка тоже укладывался в chunk_size
    target = max(1, chunk_size - max(0, overlap))
    sentences: List[str] = []
    for sent in raw_sentences:
        if split_long and len(sent) > chunk_size:
            sentences.extend(split_long_sentence(sent, target))
        else:
            sentences.append(sent)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len <= chunk_size:
            current_chunk.append(sent)
            current_len += sent_len
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            overlap_sentences: List[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_sentences + [sent]
            current_len = overlap_len + sent_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def chunk_by_paragraphs(text: str, chunk_size: int, overlap: int,
                        split_long: bool = True) -> List[str]:
    """Разбивает по абзацам; длинный абзац дополнительно режется по предложениям."""
    chunks: List[str] = []
    for p in split_paragraphs(text):
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            chunks.extend(chunk_by_sentences(p, chunk_size, overlap, split_long))
    return chunks


def chunk_document(text: str, strategy: str, chunk_size: int, overlap: int,
                   split_long: bool = True) -> List[Dict[str, Any]]:
    """Разбивает текст документа на чанки согласно выбранной стратегии."""
    if strategy == 'sentence':
        chunk_texts = chunk_by_sentences(text, chunk_size, overlap, split_long)
    elif strategy == 'paragraph':
        chunk_texts = chunk_by_paragraphs(text, chunk_size, overlap, split_long)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}. Доступно: sentence, paragraph")

    result = []
    for idx, chunk_text in enumerate(t for t in chunk_texts if t.strip()):
        result.append({
            'text': chunk_text,
            'chunk_index': idx,
            'chunk_size': len(chunk_text)
        })
    return result
