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


# Заголовок раздела: номер вида «8.», «8.1.», «8.1.2.» в начале строки,
# затем пробел и слово с заглавной буквы. После нормализации PDF заголовки
# остаются на отдельных строках, поэтому якорь по началу строки надёжен.
HEADING_RE = re.compile(r'^(\d{1,2}(?:\.\d{1,2})*)\.\s+(\S.*)$', re.MULTILINE)


def split_sections(text: str) -> List[Dict[str, str]]:
    """
    Делит текст на разделы по строкам-заголовкам.

    Возвращает список {'heading': ..., 'body': ...}. Текст до первого заголовка
    попадает в раздел с пустым heading — обычно это титул документа.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [{'heading': '', 'body': text}]

    sections: List[Dict[str, str]] = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append({'heading': '', 'body': preamble})

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({
            'heading': m.group(0).strip(),
            'body': text[m.end():end].strip(),
        })
    return sections


def chunk_by_sections(text: str, chunk_size: int, overlap: int,
                      split_long: bool = True) -> List[str]:
    """
    Чанкинг по разделам документа.

    Решает задачу атрибуции: показатели, перечисленные внутри раздела, остаются
    в одном чанке с его заголовком. При нарезке предложениями заголовок мог
    оказаться в предыдущем чанке, и числа теряли привязку к кейсу.

    Правила:
    - раздел целиком помещается в chunk_size — становится одним чанком;
    - раздел длиннее — режется по предложениям, и заголовок ПОВТОРЯЕТСЯ
      в начале каждой части, чтобы привязка не терялась;
    - подряд идущие короткие разделы склеиваются, пока укладываются в chunk_size,
      иначе документ рассыпается на десятки мелких фрагментов.
    """
    chunks: List[str] = []
    pending: List[str] = []
    pending_len = 0

    def flush() -> None:
        nonlocal pending, pending_len
        if pending:
            chunks.append('\n'.join(pending))
            pending, pending_len = [], 0

    for section in split_sections(text):
        heading, body = section['heading'], section['body']
        full = f"{heading}\n{body}".strip() if heading else body
        if not full:
            continue

        if len(full) <= chunk_size:
            # короткий раздел: копим, пока влезает
            if pending_len + len(full) > chunk_size:
                flush()
            pending.append(full)
            pending_len += len(full) + 1
            continue

        # длинный раздел: сбрасываем накопленное и режем с повтором заголовка
        flush()
        prefix = f"{heading} " if heading else ""
        budget = max(1, chunk_size - len(prefix))
        for part in chunk_by_sentences(body, budget, overlap, split_long):
            chunks.append(f"{prefix}{part}".strip())

    flush()
    return chunks


def chunk_document(text: str, strategy: str, chunk_size: int, overlap: int,
                   split_long: bool = True) -> List[Dict[str, Any]]:
    """Разбивает текст документа на чанки согласно выбранной стратегии."""
    if strategy == 'sentence':
        chunk_texts = chunk_by_sentences(text, chunk_size, overlap, split_long)
    elif strategy == 'paragraph':
        chunk_texts = chunk_by_paragraphs(text, chunk_size, overlap, split_long)
    elif strategy == 'section':
        chunk_texts = chunk_by_sections(text, chunk_size, overlap, split_long)
    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. Доступно: sentence, paragraph, section")

    result = []
    for idx, chunk_text in enumerate(t for t in chunk_texts if t.strip()):
        result.append({
            'text': chunk_text,
            'chunk_index': idx,
            'chunk_size': len(chunk_text)
        })
    return result
