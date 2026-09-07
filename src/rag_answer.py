"""
Финальный этап RAG: вопрос -> retrieval -> контекст -> промпт -> LLM -> ответ.

CLI:
    python src/rag_answer.py "Что входит в цифровое ядро AIMS?"
    python src/rag_answer.py "Эффекты от IBP?" --top-k 5
    python src/rag_answer.py "Эффекты от IBP?" --no-llm      # только retrieval, без генерации
    python src/rag_answer.py "Эффекты от IBP?" --json        # машиночитаемый вывод

Ответ всегда сопровождается списком фрагментов, использованных как контекст.
"""

import argparse
import json
import sys
from typing import Any, Dict, List

import yaml

from context_builder import build_context, format_chunks_for_display
from llm_client import LLMClient, LLMError, create_client
from retriever import Retriever

DEFAULT_CONFIG = "config/rag.yaml"

USER_PROMPT_TEMPLATE = """Контекст из базы знаний:
-----
{context}
-----

Вопрос пользователя: {question}

Ответь на вопрос, используя только контекст выше."""


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_prompt(question: str, context: str, config: Dict[str, Any]) -> Dict[str, str]:
    system_prompt = config.get('prompt', {}).get('system', '').strip()
    if not system_prompt:
        raise ValueError("В config/rag.yaml не задан prompt.system")
    return {
        'system': system_prompt,
        'user': USER_PROMPT_TEMPLATE.format(context=context or "(контекст пуст)", question=question),
    }


def answer_question(question: str, config: Dict[str, Any], top_k: int = None,
                    use_llm: bool = True, retriever: Retriever = None,
                    llm: LLMClient = None) -> Dict[str, Any]:
    """
    Полный цикл для одного вопроса.
    retriever/llm можно передать снаружи, чтобы не переинициализировать модель
    при прогоне серии вопросов (см. rag_eval.py).
    """
    own_retriever = retriever is None
    retriever = retriever or Retriever(config)

    llm_init_error = None
    if llm is None:
        llm, llm_init_error = create_client(config.get('llm', {}))

    try:
        retrieved = retriever.retrieve(question, top_k=top_k)
        context, used_chunks = build_context(retrieved, config)
        prompt = build_prompt(question, context, config)

        answer, error = None, None
        if llm is None:
            error = f"LLM не сконфигурирован: {llm_init_error}"
        elif not use_llm:
            error = "Генерация отключена флагом --no-llm: показан только найденный контекст."
        elif not llm.enabled:
            error = "Генерация отключена в конфиге (llm.provider: none)."
        else:
            try:
                answer = llm.complete(prompt['system'], prompt['user'])
            except LLMError as exc:
                error = str(exc)

        return {
            'question': question,
            'top_k': int(top_k or retriever.default_top_k),
            'answer': answer,
            'error': error,
            'retrieved_chunks': retrieved,
            'used_chunks': used_chunks,
            'context': context,
            'llm': llm.describe() if llm else f"LLM не сконфигурирован: {llm_init_error}",
        }
    finally:
        if own_retriever:
            retriever.close()


def print_result(result: Dict[str, Any]) -> None:
    print("=" * 78)
    print(f"ВОПРОС: {result['question']}")
    print(f"top_k = {result['top_k']} | {result['llm']}")
    print("=" * 78)
    print("\nНАЙДЕННЫЕ ФРАГМЕНТЫ (контекст для LLM):")
    print(format_chunks_for_display(result['used_chunks']))
    print("\nОТВЕТ:")
    print(result['answer'] if result['answer'] else f"[нет ответа] {result['error']}")
    print()


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="RAG: ответ на вопрос по базе знаний")
    parser.add_argument("question", help="вопрос пользователя")
    parser.add_argument("--top-k", type=int, default=None,
                        help="сколько чанков извлекать (по умолчанию из config/rag.yaml)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="путь к config/rag.yaml")
    parser.add_argument("--no-llm", action="store_true", help="только retrieval, без генерации")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="вывести результат в JSON")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = answer_question(args.question, config, top_k=args.top_k, use_llm=not args.no_llm)

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result)

    return 0 if (result['answer'] or args.no_llm) else 1


if __name__ == "__main__":
    sys.exit(main())
