import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    print("PyPDF2 не установлен. Установите: pip install PyPDF2")

def load_documents(raw_dir: str) -> List[Dict[str, Any]]:
    documents = []
    raw_path = Path(raw_dir)
    
    for file_path in raw_path.iterdir():
        if not file_path.is_file():
            continue
        
        ext = file_path.suffix.lower()
        source = str(file_path)
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            documents.append({'text': text, 'source': source, 'file_type': 'txt'})
        
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict) and 'text' in item:
                        text = item['text']
                    elif isinstance(item, dict) and 'content' in item:
                        text = item['content']
                    else:
                        text = json.dumps(item, ensure_ascii=False)
                    documents.append({
                        'text': text,
                        'source': f"{source}#{idx}",
                        'file_type': 'json'
                    })
            elif isinstance(data, dict):
                if 'text' in data:
                    text = data['text']
                elif 'content' in data:
                    text = data['content']
                else:
                    text = json.dumps(data, ensure_ascii=False)
                documents.append({'text': text, 'source': source, 'file_type': 'json'})
            else:
                documents.append({'text': str(data), 'source': source, 'file_type': 'json'})
        
        elif ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parts = []
                    for k, v in row.items():
                        if isinstance(v, list):
                            v = ', '.join(str(item) for item in v)
                        if v and str(v).strip():
                            parts.append(f"{k}: {v}")
                    text = ". ".join(parts)
                    if text.strip():
                        documents.append({
                            'text': text,
                            'source': f"{source}#{reader.line_num}",
                            'file_type': 'csv'
                        })
        
        elif ext == '.pdf':
            if PyPDF2 is None:
                print(f"Пропуск {file_path}: PyPDF2 не установлен")
                continue
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
                    if text.strip():
                        documents.append({'text': text, 'source': source, 'file_type': 'pdf'})
                    else:
                        print(f"Предупреждение: не удалось извлечь текст из {file_path}")
            except Exception as e:
                print(f"Ошибка при чтении PDF {file_path}: {e}")
        
        elif ext == '.fb2':
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                # Пространство имён FB2
                ns = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
                # Извлекаем весь текст из тегов <p> внутри <body>
                paragraphs = []
                for p in root.findall('.//fb:p', ns):
                    # Объединяем текст внутри тега p (может быть несколько)
                    para_text = ''.join(p.itertext()).strip()
                    if para_text:
                        paragraphs.append(para_text)
                text = '\n\n'.join(paragraphs)  # разделяем абзацы двумя переводами строк
                if text.strip():
                    documents.append({'text': text, 'source': source, 'file_type': 'fb2'})
                else:
                    print(f"Предупреждение: не удалось извлечь текст из {file_path}")
            except Exception as e:
                print(f"Ошибка при чтении FB2 {file_path}: {e}")
    
    return documents