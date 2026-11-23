#!/usr/bin/env python3
"""Train Stage-7 char TF-IDF + logistic regression on frozen train only.

Validation selects one threshold and gives model diagnostics. Test remains sealed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.detectors import (  # noqa: E402
    FROZEN_DATASET_SHA256, LLAMA_ID, QWEN_ID,
    choose_validation_threshold, load_detector_training_data,
)


def _slug(model_id: str) -> str:
    return 'llama-3.2-3b' if model_id == LLAMA_ID else 'qwen3-4b-base'


def _json_write(path: Path, object_: object) -> None:
    path.write_text(json.dumps(object_, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', required=True, choices=[LLAMA_ID, QWEN_ID])
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/stage7/dataset/detector_records.jsonl')
    parser.add_argument('--manifest', type=Path, default=ROOT/'results/stage7/dataset_manifest.json')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/stage7/detector_training.json')
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/stage7/checkpoints/baseline')
    parser.add_argument('--model-root', type=Path, default=ROOT/'models/detectors/stage7')
    parser.add_argument('--dry-run', action='store_true', help='Check frozen dataset without fitting')
    args = parser.parse_args()
    try:
        config_bytes = args.config.read_bytes()
        cfg = json.loads(config_bytes)
        expected_base = {
            'analyzer': 'char', 'ngram_range': [3, 5], 'min_df': 2,
            'max_features': 100000, 'sublinear_tf': True,
            'logistic_regression_C': 1.0,
            'logistic_regression_max_iter': 2000, 'class_weight': None,
        }
        if (cfg.get('schema_version') != 'stage7.detector_training.v1' or
                cfg.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
                cfg.get('baseline') != expected_base):
            raise ConfigurationError('Stage-7 frozen detector baseline configuration changed')
        data = load_detector_training_data(dataset=args.dataset, manifest=args.manifest,
                model_id=args.model_id, expected_sha256=FROZEN_DATASET_SHA256)
        output = args.output_root/_slug(args.model_id)
        model_dir = args.model_root/_slug(args.model_id)
        preflight = {'status':'frozen_dataset_verified', 'model_id':args.model_id,
                     'dataset_sha256':data.dataset_sha256, 'counts':data.counts,
                     'test_access':'none', 'dry_run':args.dry_run}
        if args.dry_run:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0
        if output.exists() or (model_dir/'tfidf_logreg.joblib').exists():
            raise ConfigurationError('detector baseline outputs exist; refusing overwrite')
        try:
            import joblib
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            import sklearn
        except ImportError as exc:
            raise ConfigurationError('install scikit-learn and joblib for Stage-7 baseline') from exc
        train_text = [row['text'] for row in data.train]
        train_y = [row['label'] for row in data.train]
        val_text = [row['text'] for row in data.validation]
        val_y = [row['label'] for row in data.validation]
        if set(train_y) != {0, 1} or set(val_y) != {0, 1}:
            raise ConfigurationError('both classes required in train and validation')
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2,
                max_features=100000, sublinear_tf=True)
        x_train = vectorizer.fit_transform(train_text)
        classifier = LogisticRegression(C=1.0, max_iter=2000, class_weight=None)
        classifier.fit(x_train, train_y)
        scores = classifier.predict_proba(vectorizer.transform(val_text))[:, 1].tolist()
        threshold, balanced_accuracy = choose_validation_threshold(val_y, scores)
        auc = float(roc_auc_score(val_y, scores))
        predictions = [
            {'record_id': row['record_id'], 'pair_id': row['pair_id'], 'label': row['label'],
             'score_stego': score, 'partition': 'validation'}
            for row, score in zip(data.validation, scores)
        ]
        summary = {**preflight, 'status':'complete', 'dry_run':False,
                   'baseline':expected_base, 'config_sha256':sha256(config_bytes).hexdigest(),
                   'validation_roc_auc':auc, 'validation_threshold':threshold,
                   'validation_balanced_accuracy':balanced_accuracy,
                   'tfidf_feature_count':len(vectorizer.vocabulary_),
                   'sklearn_version':sklearn.__version__,
                   'test_access':'none'}
        output.mkdir(parents=True, exist_ok=False)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({'vectorizer':vectorizer, 'classifier':classifier,
                     'dataset_sha256':data.dataset_sha256, 'config_sha256':summary['config_sha256']},
                    model_dir/'tfidf_logreg.joblib')
        _json_write(output/'summary.json', summary)
        (output/'validation_predictions.jsonl').write_text(
            ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in predictions), encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True))
        return 0
    except (OSError, ValueError, ConfigurationError) as exc:
        print(f'Stage-7 baseline training: ERROR ({type(exc).__name__}: {exc})',file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
