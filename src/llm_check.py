"""
Проверка подключения к LLM без retrieval и без Qdrant.

Нужен, чтобы отделить проблемы доступа (ключ, роль, идентификатор модели)
от проблем пайплайна. Запускается сразу после заполнения .env,
до построения индекса:

    python src/llm_check.py
    python src/llm_check.py --config config/rag.yaml

Ключ не печатается: в выводе видны только последние 4 символа.
"""

import argparse
import sys
from typing import Any, Dict

import yaml

from llm_client import LLMError, create_client

DEFAULT_CONFIG = "config/rag.yaml"

PING_SYSTEM = "Отвечай ровно одним словом, без пояснений и знаков препинания."
PING_USER = "Ответь словом: готово"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Проверка доступа к LLM")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="путь к config/rag.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    llm_config = config.get('llm', {})

    print(f"provider   : {llm_config.get('provider')}")
    client, init_error = create_client(llm_config)
    if client is None:
        print(f"КОНФИГУРАЦИЯ: {init_error}")
        return 2

    print(f"подключение: {client.describe()}")
    if not client.enabled:
        print("Генерация отключена (llm.provider: none) — проверять нечего.")
        return 0

    if client.provider == 'openai_compatible':
        print(f"эндпоинт   : {client.base_url}/chat/completions")
        print(f"схема авт. : {client.auth_scheme}")
    else:
        print(f"эндпоинт   : {client.base_url}")

    try:
        answer = client.complete(PING_SYSTEM, PING_USER)
    except LLMError as exc:
        print(f"\nОШИБКА: {exc}")
        return 1

    print(f"\nОтвет модели: {answer!r}")
    print("Подключение работает. Можно запускать пайплайн и src/rag_answer.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
