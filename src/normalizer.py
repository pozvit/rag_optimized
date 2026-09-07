import unicodedata
import re

def normalize_text(text: str, config: dict) -> str:
    """
    Нормализует Unicode, приводит концы строк к единому формату,
    опционально приводит к нижнему регистру.
    """
    # Unicode нормализация
    form = config.get('unicode_form', 'NFKC')
    text = unicodedata.normalize(form, text)
    
    # Нормализация концов строк
    if config.get('normalize_line_endings', True):
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)  # не более двух переносов подряд
    
    # Приведение к нижнему регистру (опционально)
    if config.get('lowercase', False):
        text = text.lower()
    
    return text