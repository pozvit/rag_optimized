import re


def _unwrap_pdf_lines(text: str) -> str:
    """
    Склеивает строки, разорванные вёрсткой PDF.

    Экстрактор PDF ставит перевод строки на каждой визуальной строке, из-за чего
    предложения рвутся посередине. Это мешает и чанкингу по предложениям,
    и читаемости контекста, который уходит в LLM.

    Строка НЕ склеивается со следующей, если:
      - заканчивается на . ! ? : ; — конец смысловой единицы;
      - следующая строка начинается с маркера списка, номера пункта или заголовка.
    Перенос со знаком «-» в конце строки склеивается без пробела (перенос слова).
    """
    lines = [line.strip() for line in text.splitlines()]
    starts_new_block = re.compile(r'^(#{1,6}\s|[•\-*—]\s|\d+(\.\d+)*[.)]\s|\[)')

    result = []
    buffer = ''
    for line in lines:
        if not line:
            if buffer:
                result.append(buffer)
                buffer = ''
            result.append('')
            continue

        if not buffer:
            buffer = line
            continue

        if buffer.endswith(('.', '!', '?', ':', ';')) or starts_new_block.match(line):
            result.append(buffer)
            buffer = line
        elif buffer.endswith('-'):
            buffer = buffer[:-1] + line
        else:
            buffer = f'{buffer} {line}'

    if buffer:
        result.append(buffer)
    return '\n'.join(result)


def clean_text(text: str, config: dict) -> str:
    """
    Очищает текст: удаляет управляющие символы, лишние пробелы, пустые строки,
    опционально склеивает строки, разорванные вёрсткой PDF.
    """
    # Удаление управляющих символов (кроме перевода строки и табуляции)
    if config.get('remove_control_characters', True):
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Замена множественных пробелов на один
    if config.get('remove_extra_spaces', True):
        text = re.sub(r'[ \t]+', ' ', text)

    # Склейка «рваных» строк из PDF — до удаления пустых строк,
    # потому что пустая строка здесь является границей абзаца
    if config.get('unwrap_pdf_line_breaks', False):
        text = _unwrap_pdf_lines(text)

    # Удаление пустых строк
    if config.get('remove_empty_lines', True):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)

    return text
