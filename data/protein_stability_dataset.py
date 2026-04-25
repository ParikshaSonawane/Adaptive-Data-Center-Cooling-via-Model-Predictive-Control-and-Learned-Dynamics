"""
Utilities for finding, loading, and standardizing protein stability datasets.

The fastest path for a ProtBERT mutation-stability prototype is the
`RosettaCommons/MegaScale` Hugging Face dataset, specifically the
`dataset3_single` subset:

  - it is already hosted in Hugging Face format
  - it contains single-point mutations
  - it exposes `ddG_ML` and a stabilizing flag

This module standardizes those records into a consistent format:

    protein_id
    wildtype_sequence
    mutation
    mutant_sequence
    ddg
    label
    split

`label` is binary (`1` for stabilizing, `0` for destabilizing/non-stabilizing).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MUTATION_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class MutationRecord:
    wildtype_aa: str
    position: int
    mutant_aa: str


def parse_mutation(mutation: str) -> MutationRecord:
    match = MUTATION_PATTERN.fullmatch(mutation.strip())
    if not match:
        raise ValueError(
            f"Unsupported mutation format '{mutation}'. Expected a single-site "
            "mutation like 'A42V'."
        )

    wildtype_aa, position_text, mutant_aa = match.groups()
    position = int(position_text)

    if wildtype_aa not in AMINO_ACIDS or mutant_aa not in AMINO_ACIDS:
        raise ValueError(f"Mutation contains non-canonical amino acids: '{mutation}'.")

    return MutationRecord(
        wildtype_aa=wildtype_aa,
        position=position,
        mutant_aa=mutant_aa,
    )


def replace_residue(sequence: str, position: int, expected_aa: str, new_aa: str) -> str:
    index = position - 1
    if index < 0 or index >= len(sequence):
        raise ValueError(
            f"Mutation position {position} is outside sequence length {len(sequence)}."
        )

    observed = sequence[index]
    if observed != expected_aa:
        raise ValueError(
            f"Residue mismatch at position {position}: expected '{expected_aa}', "
            f"found '{observed}'."
        )

    return sequence[:index] + new_aa + sequence[index + 1 :]


def derive_sequence_pair(sequence: str, mutation: str) -> Tuple[str, str]:
    """
    Infer wild-type and mutant sequences from a sequence column plus mutation code.

    Some datasets store the mutant sequence; some store the wild-type sequence.
    We detect which one we have from the residue at the mutation site.
    """
    record = parse_mutation(mutation)
    index = record.position - 1
    if index < 0 or index >= len(sequence):
        raise ValueError(
            f"Mutation '{mutation}' does not fit sequence length {len(sequence)}."
        )

    residue = sequence[index]

    if residue == record.wildtype_aa:
        wildtype_sequence = sequence
        mutant_sequence = replace_residue(
            sequence,
            record.position,
            record.wildtype_aa,
            record.mutant_aa,
        )
        return wildtype_sequence, mutant_sequence

    if residue == record.mutant_aa:
        mutant_sequence = sequence
        wildtype_sequence = replace_residue(
            sequence,
            record.position,
            record.mutant_aa,
            record.wildtype_aa,
        )
        return wildtype_sequence, mutant_sequence

    raise ValueError(
        f"Could not align mutation '{mutation}' with sequence residue '{residue}' "
        f"at position {record.position}."
    )


def load_megascale_dataset(cache_dir: Optional[str] = None):
    """
    Load the single-mutation ddG benchmark split from Hugging Face.

    Source:
      https://huggingface.co/datasets/RosettaCommons/MegaScale
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with "
            "`py -3 -m pip install datasets` or `py -3 -m pip install -r requirements.txt`."
        ) from exc

    dataset_tag = "dataset3_single"
    return load_dataset(
        path="RosettaCommons/MegaScale",
        name=dataset_tag,
        data_dir=dataset_tag,
        cache_dir=cache_dir,
    )


def standardize_megascale_split(
    split_df: pd.DataFrame,
    split_name: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    required_columns = {"WT_name", "aa_seq", "mut_type", "ddG_ML", "Stabilizing_mut"}
    missing_columns = required_columns - set(split_df.columns)
    if missing_columns:
        raise ValueError(f"MegaScale split is missing columns: {sorted(missing_columns)}")

    frame = split_df.loc[:, ["WT_name", "aa_seq", "mut_type", "ddG_ML", "Stabilizing_mut"]]
    frame = frame.dropna(subset=["WT_name", "aa_seq", "mut_type", "ddG_ML"])

    if limit is not None:
        frame = frame.head(limit)

    standardized_rows = []
    skipped = 0

    for row in frame.itertuples(index=False):
        try:
            wildtype_sequence, mutant_sequence = derive_sequence_pair(row.aa_seq, row.mut_type)
        except ValueError:
            skipped += 1
            continue

        label = int(bool(row.Stabilizing_mut))
        standardized_rows.append(
            {
                "protein_id": row.WT_name,
                "wildtype_sequence": wildtype_sequence,
                "mutation": row.mut_type,
                "mutant_sequence": mutant_sequence,
                "ddg": float(row.ddG_ML),
                "label": label,
                "split": split_name,
            }
        )

    result = pd.DataFrame(standardized_rows)
    result.attrs["skipped_rows"] = skipped
    return result


def load_megascale_as_dataframes(
    cache_dir: Optional[str] = None,
    per_split_limit: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    dataset = load_megascale_dataset(cache_dir=cache_dir)
    standardized = {}

    for split_name, split_dataset in dataset.items():
        standardized[split_name] = standardize_megascale_split(
            split_dataset.to_pandas(),
            split_name=split_name,
            limit=per_split_limit,
        )

    return standardized


def save_standardized_splits(
    output_dir: str,
    frames: Dict[str, pd.DataFrame],
) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}
    for split_name, frame in frames.items():
        file_path = output_path / f"megascale_{split_name}.csv"
        frame.to_csv(file_path, index=False)
        saved_paths[split_name] = file_path

    return saved_paths


def preview_rows(frames: Dict[str, pd.DataFrame], max_rows: int = 3) -> str:
    lines = []
    for split_name, frame in frames.items():
        skipped = frame.attrs.get("skipped_rows", 0)
        lines.append(
            f"{split_name}: {len(frame)} usable rows"
            + (f" ({skipped} skipped during sequence reconstruction)" if skipped else "")
        )
        if not frame.empty:
            lines.append(frame.head(max_rows).to_string(index=False))
            lines.append("")
    return "\n".join(lines).strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and standardize the MegaScale single-mutation protein stability dataset."
    )
    parser.add_argument(
        "--output-dir",
        default="data/protein_stability",
        help="Directory where standardized CSV files will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to keep per split for a fast prototype.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Load and print a preview without saving CSV files.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    frames = load_megascale_as_dataframes(
        cache_dir=args.cache_dir,
        per_split_limit=args.limit,
    )

    print(preview_rows(frames))

    if args.preview_only:
        return

    saved_paths = save_standardized_splits(
        output_dir=args.output_dir,
        frames=frames,
    )

    print("\nSaved files:")
    for split_name, file_path in saved_paths.items():
        print(f"  {split_name}: {file_path}")


if __name__ == "__main__":
    main()
