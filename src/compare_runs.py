"""
Сравнение двух прогонов RAG на одном и том же наборе вопросов (пункт 9 задания).

Читает results/<baseline>/qa_results.json и results/<improved>/qa_results.json,
сверяет параметры пайплайна и собирает таблицу «до/после» по каждому вопросу:
найденные чанки, score, ответ LLM, изменение.

Запуск:
    python src/compare_runs.py
    python src/compare_runs.py --baseline baseline --improved improved
    python src/compare_runs.py --out results/comparison.md

Колонки «Оценка» и «Стало лучше» заполняются вручную: автоматика не знает,
correct ли ответ по существу. Всё измеримое — score, состав документов в выдаче,
факт отказа, длина контекста — подставляется.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REFUSAL_MARKER = "Недостаточно информации"

# Параметры, по которым сверяются два прогона. Ключ -> человекочитаемое имя.
TRACKED_PARAMS = [
    ('embedding_model', 'Embedding-модель'),
    ('chunk_size', 'Chunk size'),
    ('chunk_overlap', 'Overlap'),
    ('chunk_strategy', 'Стратегия чанкинга'),
    ('indexed_points', 'Чанков в индексе'),
    ('top_k', 'top-k'),
    ('score_threshold', 'score_threshold'),
    ('max_context_chars', 'Лимит контекста'),
    ('max_chars_per_chunk', 'Лимит на фрагмент'),
    ('temperature', 'temperature'),
    ('llm', 'LLM'),
]


def load_run(results_dir: Path, label: str) -> Dict[str, Any]:
    path = results_dir / label / 'qa_results.json'
    if not path.exists():
        raise SystemExit(
            f"Не найден {path}.\n"
            f"Сначала выполните прогон: python src/rag_eval.py --label {label}"
        )
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def index_by_id(run: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {rec['id']: rec for rec in run.get('results', [])}


def describe_chunks(rec: Optional[Dict[str, Any]]) -> str:
    """Короткая сводка выдачи: сколько чанков, из каких файлов, лучший score."""
    if rec is None:
        return "— (вопроса не было в прогоне)"
    chunks = rec.get('used_chunks') or []
    if not chunks:
        return "контекст пуст"

    files: List[str] = []
    for chunk in chunks:
        name = (chunk.get('metadata') or {}).get('file_name', '?')
        if name not in files:
            files.append(name)

    top = chunks[0]
    top_meta = top.get('metadata') or {}
    parts = [
        f"{len(chunks)} шт., score {top.get('score')}…{chunks[-1].get('score')}",
        f"топ: `{top_meta.get('file_name', '?')}` chunk_index {top_meta.get('chunk_index')}",
    ]
    if len(files) > 1:
        parts.append(f"файлов в выдаче: {len(files)}")
    return "<br>".join(parts)


def describe_answer(rec: Optional[Dict[str, Any]], limit: int = 300) -> str:
    if rec is None:
        return "—"
    answer = rec.get('answer') or ""
    if not answer:
        return f"_[ответ не получен] {rec.get('error', '')}_"
    text = answer.replace('\n', ' ').strip()
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def refused(rec: Optional[Dict[str, Any]]) -> bool:
    if rec is None:
        return False
    return REFUSAL_MARKER.lower() in (rec.get('answer') or "").lower()


def render_params_table(base_meta: Dict[str, Any], impr_meta: Dict[str, Any]) -> List[str]:
    lines = [
        "## Параметры: baseline и после улучшений",
        "",
        "| Параметр | Baseline | После улучшений | Изменено |",
        "|---|---|---|---|",
    ]
    for key, title in TRACKED_PARAMS:
        before = base_meta.get(key, '—')
        after = impr_meta.get(key, '—')
        changed = "**да**" if str(before) != str(after) else ""
        lines.append(f"| {title} | {before} | {after} | {changed} |")
    lines.extend([
        "",
        f"Конфигурации: baseline — `{base_meta.get('config_file', '—')}`, "
        f"после улучшений — `{impr_meta.get('config_file', '—')}`.",
        "",
        "Промпт сравнивайте отдельно: полные тексты напечатаны в шапке каждого отчёта.",
        "",
    ])
    return lines


def render_comparison(base: Dict[str, Any], impr: Dict[str, Any]) -> str:
    base_by_id = index_by_id(base)
    impr_by_id = index_by_id(impr)

    # Порядок берём из baseline, вопросы только из improved добавляем в конец
    ids = [rec['id'] for rec in base.get('results', [])]
    ids += [qid for qid in impr_by_id if qid not in ids]

    lines = [
        "# Сравнение baseline и улучшенной версии",
        "",
        f"Baseline: прогон от {base.get('meta', {}).get('timestamp', '—')}  ",
        f"После улучшений: прогон от {impr.get('meta', {}).get('timestamp', '—')}",
        "",
        f"Вопросов сравнивается: {len(ids)}. Набор вопросов идентичен в обоих прогонах — "
        "иначе сравнение некорректно.",
        "",
    ]
    lines.extend(render_params_table(base.get('meta', {}), impr.get('meta', {})))

    lines.extend([
        "## Сводка по вопросам",
        "",
        "| # | Тип | Baseline: чанки | Baseline: оценка | После: чанки | После: оценка | Стало лучше |",
        "|---|---|---|---|---|---|---|",
    ])
    for qid in ids:
        b, i = base_by_id.get(qid), impr_by_id.get(qid)
        qtype = (b or i or {}).get('type', '—')
        lines.append(
            f"| {qid} | {qtype} | {describe_chunks(b)} | _заполните_ | "
            f"{describe_chunks(i)} | _заполните_ | _да / частично / нет_ |"
        )

    lines.extend([
        "",
        "Столбцы «Оценка» и «Стало лучше» заполняются вручную по шкале "
        "корректно / частично корректно / некорректно.",
        "",
        "---",
        "",
    ])

    for qid in ids:
        b, i = base_by_id.get(qid), impr_by_id.get(qid)
        rec = b or i or {}
        lines.extend([
            f"## {qid}. {rec.get('question', '—')}",
            "",
            f"**Тип:** {rec.get('type', '—')}",
            "",
            f"**Ожидаемый ответ:** {rec.get('expected', '—')}",
            "",
        ])
        if rec.get('expected_source'):
            lines.extend([f"**Ожидаемый источник:** {rec['expected_source']}", ""])

        lines.extend([
            "### Baseline",
            "",
            f"Найденные чанки: {describe_chunks(b).replace('<br>', '; ')}",
            "",
            f"Ответ: {describe_answer(b, limit=600)}",
            "",
            f"Отказ от ответа: {'да' if refused(b) else 'нет'}",
            "",
            "Оценка: _(корректно / частично корректно / некорректно)_",
            "",
            "### После улучшений",
            "",
            f"Найденные чанки: {describe_chunks(i).replace('<br>', '; ')}",
            "",
            f"Ответ: {describe_answer(i, limit=600)}",
            "",
            f"Отказ от ответа: {'да' if refused(i) else 'нет'}",
            "",
            "Оценка: _(корректно / частично корректно / некорректно)_",
            "",
            "**Проблема baseline:** _(что именно было не так)_",
            "",
            "**Что изменили:** _(какое улучшение адресует эту проблему)_",
            "",
            "**Стало лучше:** _(да / частично / нет — и почему)_",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнение двух прогонов RAG")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--baseline", default="baseline", help="метка исходного прогона")
    parser.add_argument("--improved", default="improved", help="метка улучшенного прогона")
    parser.add_argument("--out", default=None, help="куда сохранить (по умолчанию results/comparison.md)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    base = load_run(results_dir, args.baseline)
    impr = load_run(results_dir, args.improved)

    base_ids = {r['id'] for r in base.get('results', [])}
    impr_ids = {r['id'] for r in impr.get('results', [])}
    if base_ids != impr_ids:
        only_base = sorted(base_ids - impr_ids)
        only_impr = sorted(impr_ids - base_ids)
        print("ВНИМАНИЕ: наборы вопросов различаются, сравнение будет неполным.")
        if only_base:
            print(f"  только в baseline: {', '.join(only_base)}")
        if only_impr:
            print(f"  только в improved: {', '.join(only_impr)}")

    out_path = Path(args.out) if args.out else results_dir / 'comparison.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_comparison(base, impr), encoding='utf-8')

    print(f"Сохранено: {out_path}")
    print(f"Сравнено вопросов: {len(base_ids | impr_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
