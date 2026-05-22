#!/usr/bin/env python

from collections import Counter
from typing import Any, Dict, List, Sequence


def _format_code(code: Sequence[int], one_based: bool = True) -> str:
    if one_based:
        return ".".join(str(int(x) + 1) for x in code)
    return ".".join(str(int(x)) for x in code)


def _layer1_bucket(code: Sequence[int], one_based: bool = True) -> str:
    if not code:
        return "unknown.x.y"
    first = int(code[0]) + 1 if one_based else int(code[0])
    return f"{first}.x.y"


def _top_code_for_class(code_strings: List[str], total: int) -> Dict[str, Any]:
    if total == 0:
        return {
            "code": None,
            "count": 0,
            "total": 0,
            "proportion": 0.0,
        }

    counter = Counter(code_strings)
    top_code, top_count = counter.most_common(1)[0]
    proportion = float(top_count / total)
    return {
        "code": top_code,
        "count": int(top_count),
        "total": int(total),
        "proportion": proportion,
    }


def analyze_epoch_code_distribution(
    labels: Sequence[int],
    codes: Sequence[Sequence[int]],
    epoch: int,
    one_based_display: bool = True,
) -> Dict[str, Any]:
    n_samples = min(len(labels), len(codes))
    if n_samples == 0:
        return {
            "epoch": int(epoch),
            "n_samples": 0,
            "n_positive": 0,
            "n_negative": 0,
            "layer1_distribution": [],
            "negative_top_code": {
                "code": None,
                "count": 0,
                "total": 0,
                "proportion": 0.0,
            },
            "positive_top_code": {
                "code": None,
                "count": 0,
                "total": 0,
                "proportion": 0.0,
            },
        }

    layer1_counts = Counter()
    neg_codes = []
    pos_codes = []

    pos_total = 0
    neg_total = 0

    for label, code in zip(labels[:n_samples], codes[:n_samples]):
        code_int = [int(x) for x in code]
        full_code = _format_code(code_int, one_based=one_based_display)
        bucket = _layer1_bucket(code_int, one_based=one_based_display)

        layer1_counts[bucket] += 1

        if int(label) == 1:
            pos_total += 1
            pos_codes.append(full_code)
        else:
            neg_total += 1
            neg_codes.append(full_code)

    layer1_distribution = []
    for bucket, count in sorted(layer1_counts.items(), key=lambda x: x[0]):
        layer1_distribution.append(
            {
                "layer1_code": bucket,
                "count": int(count),
                "percentage": float(count / n_samples),
            }
        )

    return {
        "epoch": int(epoch),
        "n_samples": int(n_samples),
        "n_positive": int(pos_total),
        "n_negative": int(neg_total),
        "layer1_distribution": layer1_distribution,
        "negative_top_code": _top_code_for_class(neg_codes, neg_total),
        "positive_top_code": _top_code_for_class(pos_codes, pos_total),
    }
