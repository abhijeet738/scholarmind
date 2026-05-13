"""
ScholarMind — Entity Resolution

Merges duplicate entities: "BERT-large" = "bert_large" = "BERT Large"
Uses string similarity (Levenshtein) to group entities under canonical names.
No ML model needed — just string operations.
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd


def normalize_name(name: str) -> str:
    """Normalize an entity name for comparison."""
    name = name.lower().strip()
    name = re.sub(r"[\-_\.]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def string_similarity(a: str, b: str) -> float:
    """Compute string similarity using SequenceMatcher (0 to 1)."""
    return SequenceMatcher(None, a, b).ratio()


MANUAL_ALIASES = {
    "gpt 4": "gpt-4", "gpt4": "gpt-4", "gpt 3": "gpt-3", "gpt3": "gpt-3",
    "gpt 2": "gpt-2", "gpt2": "gpt-2", "llama 2": "llama-2", "llama2": "llama-2",
    "llama 3": "llama-3", "llama3": "llama-3", "bert large": "bert-large",
    "bert base": "bert-base", "roberta large": "roberta-large",
    "f1 score": "f1-score", "f 1 score": "f1-score", "f1": "f1-score",
    "squad 2 0": "squad-2.0", "cifar 10": "cifar-10", "cifar 100": "cifar-100",
    "auc roc": "auc-roc", "chain of thought": "chain-of-thought",
    "cot": "chain-of-thought", "lora": "lora", "qlora": "qlora",
}


def resolve_entities(entities_df: pd.DataFrame, similarity_threshold: float = 0.85) -> pd.DataFrame:
    """Resolve duplicate entities by assigning canonical names."""
    print(f"Resolving entities: {len(entities_df):,} total mentions")
    resolved = []

    for entity_type in entities_df["type"].unique():
        type_df = entities_df[entities_df["type"] == entity_type].copy()
        unique_names = type_df["name"].unique()
        print(f"  Type: {entity_type} — {len(unique_names):,} unique names")

        canonical_map = {}
        for name in unique_names:
            normalized = normalize_name(name)
            if normalized in MANUAL_ALIASES:
                canonical_map[name] = MANUAL_ALIASES[normalized]
                continue
            matched = False
            for existing_name, canonical in canonical_map.items():
                if string_similarity(normalized, normalize_name(canonical)) >= similarity_threshold:
                    canonical_map[name] = canonical
                    matched = True
                    break
            if not matched:
                canonical_map[name] = normalized

        type_df["canonical_name"] = type_df["name"].map(canonical_map)
        resolved.append(type_df)
        n_canonical = len(set(canonical_map.values()))
        print(f"    → {n_canonical:,} canonical entities ({len(unique_names) - n_canonical:,} merged)")

    result = pd.concat(resolved, ignore_index=True)
    print(f"Resolution complete: {result['canonical_name'].nunique():,} unique canonical entities")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/entities.parquet")
    parser.add_argument("--output", default="data/entities_resolved.parquet")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    resolved = resolve_entities(df, similarity_threshold=args.threshold)
    resolved.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
