import json
import time
import uuid
from pathlib import Path
from typing import List, Dict


def save_manifest(config: Dict, stats: Dict, indexed_count: int, validation_results: Dict,
                  search_results: List, output_dir: str) -> str:
    """Создаёт manifest.json с параметрами и результатами запуска индексации."""
    out = Path(output_dir)
    manifest = {
        'run_id': str(uuid.uuid4()),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),  # UTC
        'config_file': config.get('_config_path'),
        'config': {k: v for k, v in config.items() if not k.startswith('_')},
        'vector_store_type': config['vector_store']['type'],
        'collection_name': config['vector_store']['collection_name'],
        'input_file': config['paths']['input_jsonl'],
        'input_total_lines': stats['total_lines'],
        'input_blank_lines': stats['blank_lines'],
        'input_data_lines': stats['data_lines'],
        'input_valid_records': stats['valid_records'],
        'input_error_records': stats['error_records'],
        'input_error_details': stats.get('error_details', []),
        'indexed_points': indexed_count,
        'validation': validation_results,
        'index_is_valid': validation_results.get('index_is_valid'),
        'search_tests_count': len(search_results),
        'search_relevance_summary': [
            {'test_id': r['test_id'], 'relevance_judgment': r['relevance_judgment']}
            for r in search_results
        ],
        'validation_file': (out / 'validation.json').as_posix(),
        'search_results_file': (out / 'search_results.json').as_posix(),
    }
    output_path = out / 'manifest.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return str(output_path)
