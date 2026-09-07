"""
Прогон пайплайна на тестовом наборе вопросов (config/questions.yaml).

Для каждого вопроса фиксируются: сам вопрос, найденные top-k чанки с метаданными
и score, собранный контекст и ответ LLM. Результат сохраняется в
results/qa_results.json и results/qa_report.md — это готовые артефакты для сдачи.

Оценка «корректно / частично корректно / некорректно» ставится вручную:
в отчёте под каждым ответом есть поля «Оценка» и «Комментарий», а также
ожидаемый ответ из questions.yaml для сверки.

Запуск:
    python src/rag_eval.py
    python src/rag_eval.py --top-k 5 --no-llm
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from llm_client import create_client
from rag_answer import answer_question, load_config
from retriever import Retriever


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    questions = data.get('questions', [])
    if not questions:
        raise ValueError(f"В {path} нет вопросов")
    return questions


def render_markdown(records: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    lines = [
        f"# Результаты прогона RAG на тестовых вопросах — {meta.get('label', 'default')}",
        "",
        f"Дата запуска: {meta['timestamp']}",
        f"Конфигурация: `{meta.get('config_file', '—')}`",
        "",
        "## Параметры пайплайна",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Embedding-модель | `{meta['embedding_model']}` ({meta.get('vector_size', '—')} измерений) |",
        f"| Vector store | Qdrant (local mode), коллекция `{meta['collection']}`, "
        f"метрика {meta.get('distance', '—')} |",
        f"| Чанкинг | стратегия `{meta.get('chunk_strategy', '—')}`, "
        f"chunk_size {meta.get('chunk_size', '—')}, overlap {meta.get('chunk_overlap', '—')} |",
        f"| Проиндексировано чанков | {meta.get('indexed_points', '—')} |",
        f"| top_k | {meta['top_k']} |",
        f"| score_threshold | {meta.get('score_threshold', '—')} |",
        f"| Лимиты контекста | {meta.get('max_context_chars', '—')} символов всего, "
        f"{meta.get('max_chars_per_chunk', '—')} на фрагмент |",
        f"| LLM | {meta['llm']} |",
        f"| temperature / max_tokens | {meta.get('temperature', '—')} / {meta.get('max_tokens', '—')} |",
        "",
        "### Системный промпт",
        "",
        "```",
        (meta.get('system_prompt') or '—').strip(),
        "```",
        "",
        "---",
        "",
    ]

    for rec in records:
        lines.append(f"## {rec['id']}. {rec['question']}")
        lines.append("")
        lines.append(f"**Тип вопроса:** {rec['type']}")
        lines.append("")
        lines.append(f"**Ожидаемый ответ (для сверки):** {rec.get('expected', '—')}")
        lines.append("")
        lines.append("**Найденные чанки (контекст):**")
        lines.append("")
        for chunk in rec['used_chunks']:
            m = chunk.get('metadata', {})
            lines.append(
                f"- `[Фрагмент {chunk.get('context_position')}]` score **{chunk.get('score')}** — "
                f"{m.get('document_name', '—')} (версия {m.get('version', '—')}, "
                f"файл `{m.get('file_name', '—')}`, chunk_index {m.get('chunk_index')})"
            )
            preview = (chunk.get('text') or '').replace('\n', ' ')
            lines.append(f"  > {preview[:400]}{'…' if len(preview) > 400 else ''}")
        lines.append("")
        lines.append("**Ответ системы:**")
        lines.append("")
        answer = rec.get('answer') or f"_[ответ не получен] {rec.get('error')}_"
        lines.append(answer)
        lines.append("")
        lines.append("**Оценка:** _(корректно / частично корректно / некорректно — заполните после прогона)_")
        lines.append("")
        lines.append("**Комментарий:** _(почему поставлена такая оценка)_")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend(render_conclusion(records, meta))
    return "\n".join(lines)


def render_conclusion(records: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[str]:
    """
    Заготовка краткого вывода (пункт 10 задания).

    Всё, что можно посчитать автоматически — параметры, сводная таблица по вопросам,
    факт наличия отказа на вопросе без ответа в базе — подставляется. Содержательные
    оценки помечены как заполняемые вручную: их нельзя вывести из логов.
    """
    refusal_marker = "Недостаточно информации"

    lines = [
        "# Краткий вывод",
        "",
        "## Использованные параметры",
        "",
        f"- **Embedding-модель:** `{meta['embedding_model']}` "
        f"({meta.get('vector_size', '—')} измерений, префиксы `passage:` / `query:`)",
        f"- **Vector store:** Qdrant в локальном режиме, коллекция `{meta['collection']}`, "
        f"метрика {meta.get('distance', '—')}, {meta.get('indexed_points', '—')} точек",
        f"- **Chunk size:** {meta.get('chunk_size', '—')} символов, "
        f"overlap {meta.get('chunk_overlap', '—')}, стратегия `{meta.get('chunk_strategy', '—')}`",
        f"- **top-k:** {meta['top_k']}",
        f"- **LLM:** {meta['llm']}, temperature {meta.get('temperature', '—')}, "
        f"max_tokens {meta.get('max_tokens', '—')}",
        "",
        "## Сводка по вопросам",
        "",
        "| # | Тип | Документ в топе выдачи | Лучший score | Отказ от ответа | Оценка |",
        "|---|---|---|---|---|---|",
    ]

    for rec in records:
        chunks = rec.get('used_chunks') or []
        if chunks:
            top = chunks[0]
            top_doc = (top.get('metadata') or {}).get('file_name', '—')
            top_score = top.get('score', '—')
        else:
            top_doc, top_score = "— (пусто)", "—"
        answer = rec.get('answer') or ""
        refused = "да" if refusal_marker.lower() in answer.lower() else "нет"
        lines.append(
            f"| {rec['id']} | {rec.get('type', '—')} | `{top_doc}` | {top_score} | {refused} | "
            f"_заполните_ |"
        )

    lines.extend([
        "",
        "Столбец «Оценка» заполняется вручную: корректно / частично корректно / некорректно.",
        "Основание — поле «Ожидаемый ответ» под каждым вопросом выше.",
        "",
        "## Какие вопросы отработали хорошо",
        "",
        "_Перечислите вопросы, где нужный документ попал в топ выдачи, а ответ полон и совпал",
        "с ожидаемым. Для каждого укажите score первого фрагмента и имя файла — они есть",
        "в сводке выше._",
        "",
        "## Где пайплайн ошибся или дал неполный ответ",
        "",
        "_Опишите фактические расхождения с ожидаемым ответом: потерянные пункты перечислений,",
        "попадание чанков из чужого документа, придуманные числа. Если таких случаев не было —",
        "напишите это прямо и приведите как аргумент значения score из сводки._",
        "",
        "## Что можно улучшить",
        "",
        f"- **Chunking:** текущий размер {meta.get('chunk_size', '—')} символов режет длинные",
        "  перечисления между соседними чанками; варианты — поднять `chunk_size` до 1000–1200",
        "  или перейти на разбиение по заголовкам разделов;",
        f"- **top-k:** сейчас {meta['top_k']}; увеличение повышает полноту, но тянет в контекст",
        "  слабые фрагменты — лечится reranker'ом (кросс-энкодер) поверх выдачи;",
        "- **Prompt:** правило про отказ работает, но полноту перечислений усиливает явное",
        "  требование «перечисли все пункты, встречающиеся в контексте»;",
        "- **Metadata:** `document_name` / `category` / `version` уже доходят до payload;",
        "  следующий шаг — фильтр `query_filter` по `document_name` для вопросов вида",
        "  «в документе X…», чтобы retrieval не смешивал продукты;",
        "- **Документы:** база из 5 обезличенных PDF покрывает продуктовую линейку, но не содержит",
        "  коммерческих условий — вопросы про цены принципиально безответны; при расширении базы",
        f"  стоит поднять `score_threshold` с {meta.get('score_threshold', '—')} до 0.75–0.80,",
        "  чтобы нерелевантные чанки не попадали в контекст;",
        "- **Гибридный поиск:** чисто векторный retrieval слабее на точных терминах",
        "  и аббревиатурах — связка BM25 + dense закрывает этот класс запросов.",
        "",
    ])
    return lines


def collect_pipeline_params() -> Dict[str, Any]:
    """
    Дособирает параметры соседних этапов для отчёта: чанкинг, параметры коллекции
    и фактическое число проиндексированных точек. Отсутствие любого файла не должно
    ронять прогон — недостающие значения просто не попадут в отчёт.
    """
    params: Dict[str, Any] = {}

    try:
        with open('config/chunking.yaml', 'r', encoding='utf-8') as f:
            chunking = (yaml.safe_load(f) or {}).get('chunking', {})
        params['chunk_size'] = chunking.get('chunk_size')
        params['chunk_overlap'] = chunking.get('chunk_overlap')
        params['chunk_strategy'] = chunking.get('strategy')
    except OSError:
        pass

    try:
        with open('config/vector_store.yaml', 'r', encoding='utf-8') as f:
            store = (yaml.safe_load(f) or {}).get('vector_store', {})
        params['distance'] = store.get('distance')
        params['vector_size'] = store.get('vector_size')
    except OSError:
        pass

    try:
        with open('data/vector_store/validation.json', 'r', encoding='utf-8') as f:
            params['indexed_points'] = json.load(f).get('actual_points_in_qdrant')
    except (OSError, ValueError):
        pass

    return params


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон RAG на тестовых вопросах")
    parser.add_argument("--config", default="config/rag.yaml")
    parser.add_argument("--questions", default="config/questions.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="только retrieval, без генерации")
    parser.add_argument(
        "--label", default=None,
        help="метка прогона: результаты пишутся в results/<label>/ вместо results/. "
             "Нужна, чтобы baseline и улучшенная версия не затирали друг друга "
             "(например: --label baseline, --label improved)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    questions = load_questions(args.questions)

    results_dir = Path(config.get('paths', {}).get('results_dir', 'results'))
    if args.label:
        results_dir = results_dir / args.label
    results_dir.mkdir(parents=True, exist_ok=True)

    # Отчёт с проставленными вручную оценками терять нельзя: перезапись только
    # по явному подтверждению.
    existing = results_dir / 'qa_report.md'
    if existing.exists():
        answer = input(f"{existing} уже существует и будет перезаписан. Продолжить? [y/N] ")
        if answer.strip().lower() not in ('y', 'yes', 'д', 'да'):
            print("Отменено. Используйте --label, чтобы сохранить прогон отдельно.")
            return 1

    # Модель и подключение к Qdrant инициализируются один раз на весь прогон
    retriever = Retriever(config)
    llm, llm_init_error = create_client(config.get('llm', {}))
    llm_state = llm.describe() if llm else f"LLM не сконфигурирован: {llm_init_error}"
    print(retriever.model_name, "|", llm_state)

    records: List[Dict[str, Any]] = []
    try:
        for item in questions:
            print(f"\n[{item['id']}] {item['question']}")
            result = answer_question(
                item['question'], config, top_k=args.top_k,
                use_llm=not args.no_llm, retriever=retriever, llm=llm,
            )
            if result['answer']:
                print(f"    -> {result['answer'][:200]}")
            else:
                print(f"    -> [нет ответа] {result['error']}")

            records.append({
                'id': item['id'],
                'type': item.get('type', ''),
                'question': item['question'],
                'expected': item.get('expected', ''),
                'top_k': result['top_k'],
                'answer': result['answer'],
                'error': result['error'],
                'used_chunks': result['used_chunks'],
                'context': result['context'],
            })
    finally:
        retriever.close()

    llm_config = config.get('llm', {})
    context_config = config.get('context', {})
    meta = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'label': args.label or 'default',
        'config_file': args.config,
        'embedding_model': retriever.model_name,
        'collection': retriever.collection_name,
        'top_k': args.top_k or retriever.default_top_k,
        'llm': llm_state,
        'temperature': llm_config.get('temperature'),
        'max_tokens': llm_config.get('max_tokens'),
        'score_threshold': config.get('retrieval', {}).get('score_threshold'),
        'max_context_chars': context_config.get('max_context_chars'),
        'max_chars_per_chunk': context_config.get('max_chars_per_chunk'),
        'system_prompt': config.get('prompt', {}).get('system', ''),
    }
    meta.update(collect_pipeline_params())

    json_path = results_dir / 'qa_results.json'
    md_path = results_dir / 'qa_report.md'
    json_path.write_text(json.dumps({'meta': meta, 'results': records}, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    md_path.write_text(render_markdown(records, meta), encoding='utf-8')

    print(f"\nСохранено: {json_path}")
    print(f"Сохранено: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
