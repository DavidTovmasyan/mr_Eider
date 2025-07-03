import json
from typing import Iterable
import argparse
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np


def filter_dataset(input_file: str, output_file_pretrain: str, output_file_incremental: str,
                   extractable_rel_id: Iterable[str]) -> None:
    with open(input_file, 'r') as f:
        initial_data = json.load(f)

    print(f"{input_file} was loaded successfully!")

    pretrain_data = []
    incremental_data = []

    for sample in initial_data:
        is_present = False
        for label in sample["labels"]:
            if label['r'] in extractable_rel_id:
                is_present = True
        if is_present:
            incremental_data.append(filter_sample(sample, extractable_rel_id, preserve=True))
        else:
            pretrain_data.append(filter_sample(sample, extractable_rel_id, preserve=False))

    print("Filtering has been done successfully!")

    with open(output_file_incremental, "w") as f:
        json.dump(incremental_data, f)

    print(f"Saved incremental data in {output_file_incremental}")

    with open(output_file_pretrain, "w") as f:
        json.dump(pretrain_data, f)

    print(f"Saved pretrain data in {output_file_pretrain}")


def filter_supplementary(input_file: str, output_file_pretrain: str, output_file_incremental: str,
                         extractable_rel_id: Iterable[str]) -> None:
    with open(input_file, 'r') as f:
        initial_data = json.load(f)

    print(f"{input_file} was loaded successfully!")

    pretrain_data = []
    incremental_data = []

    for sample in initial_data:
        is_present = False
        for label in sample["labels"]:
            if label['r'] in extractable_rel_id:
                is_present = True
        incremental_data.append(sample)
        if is_present:
            pretrain_data.append(filter_sample(sample, extractable_rel_id, preserve=False))
        else:
            pretrain_data.append(sample)

    print("Filtering has been done successfully!")

    with open(output_file_incremental, "w") as f:
        json.dump(incremental_data, f)

    print(f"Saved incremental data in {output_file_incremental}")

    with open(output_file_pretrain, "w") as f:
        json.dump(pretrain_data, f)

    print(f"Saved pretrain data in {output_file_pretrain}")


def filter_corefs(input_file: str, output_file_pretrain: str, output_file_incremental: str,
                  extractable_rel_id: Iterable[str], coref_file: str) -> None:
    with open(input_file, 'r') as f:
        initial_data = json.load(f)

    print(f"{input_file} was loaded successfully!")

    mask = []

    for sample in initial_data:
        is_present = False
        for label in sample["labels"]:
            if label['r'] in extractable_rel_id:
                is_present = True
        mask.append(1 if is_present else 0)

    with open(coref_file, 'r') as f:
        corefs = json.load(f)

    print(f"{input_file} was loaded successfully!")

    incremental_data, pretrain_data = [], []
    for present, sample in zip(mask, corefs):
        if present:
            incremental_data.append(sample)
        else:
            pretrain_data.append(sample)

    print("Filtering has been done successfully!")

    with open(output_file_incremental, "w") as f:
        json.dump(incremental_data, f)

    print(f"Saved incremental data in {output_file_incremental}")

    with open(output_file_pretrain, "w") as f:
        json.dump(pretrain_data, f)

    print(f"Saved pretrain data in {output_file_pretrain}")


def filter_sample(sample: dict, ext_id: Iterable[str], preserve=True):
    new_sample = {}
    for k, v in sample.items():
        if k != "labels":
            new_sample[k] = v
        else:
            if preserve:
                new_sample[k] = list(filter(lambda d: d["r"] in ext_id, v))
            else:
                new_sample[k] = list(filter(lambda d: d["r"] not in ext_id, v))
    return new_sample


def divide_uniformly(file_in: str, file_out_1: str, file_out_2: str, extractable: Iterable[str], scale: float = 0.1):
    # Load the dataset
    with open(file_in) as f:
        dataset = json.load(f)
    print(f"{file_in} has been loaded successfully!")

    # Step 1: Build relation vocab
    relation_types = set()
    for doc in dataset:
        for label in doc.get('labels', []):
            relation_types.add(label['r'])
    relation2id = {r: i for i, r in enumerate(sorted(relation_types))}
    num_relations = len(relation2id)

    # Step 2: Build multi-hot matrix
    X = np.arange(len(dataset))[:, None]
    Y = np.zeros((len(dataset), num_relations), dtype=int)

    for i, doc in enumerate(dataset):
        for label in doc.get('labels', []):
            Y[i, relation2id[label['r']]] = 1

    # Step 3: Split using iterative stratification
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=scale, random_state=42)
    X_1, X_2 = next(msss.split(X, Y))

    # Step 4: Extract samples
    set_1, set_2 = [], []

    for i, sample in enumerate(dataset):
        new_sample = {}
        if i in X_1:
            for k, v in sample.items():
                if k != "labels":
                    new_sample[k] = v
                else:
                    new_sample[k] = [d for d in v if d["r"] not in extractable]
            set_1.append(new_sample)
        else:
            for k, v in sample.items():
                if k != "labels":
                    new_sample[k] = v
                else:
                    new_sample[k] = [d for d in v if d["r"] in extractable]
            set_2.append(new_sample)

    # Step 5: Save or return
    with open(file_out_1, 'w') as f:
        json.dump(set_1, f)
    print(f"{file_out_1} has been dumped successfully!")
    with open(file_out_2, 'w') as f:
        json.dump(set_2, f)
    print(f"{file_out_2} has been dumped successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_f", default="", type=str)
    parser.add_argument("--output_f_pretrain", default="", type=str)
    parser.add_argument("--output_f_incremental", default="", type=str)
    parser.add_argument("--extractable_rel_id", default="", type=str, nargs="+")
    parser.add_argument("--coref_mode", default=False, type=bool)
    parser.add_argument("--coref_file", default="", type=str)
    parser.add_argument("--supplementary", default=False, type=bool)
    parser.add_argument("--uniformly", default=False, type=bool)
    parser.add_argument("--scale", default=0.1, type=float)
    args = parser.parse_args()

    if args.coref_mode:
        filter_corefs(args.input_f, args.output_f_pretrain, args.output_f_incremental, args.extractable_rel_id,
                      args.coref_file)
    elif args.supplementary:
        filter_supplementary(args.input_f, args.output_f_pretrain, args.output_f_incremental, args.extractable_rel_id)
    elif args.uniformly:
        divide_uniformly(args.input_f, args.output_f_pretrain, args.output_f_incremental, args.extractable_rel_id, scale=args.scale)
    else:  # Extract mode
        filter_dataset(args.input_f, args.output_f_pretrain, args.output_f_incremental, args.extractable_rel_id)


if __name__ == "__main__":
    main()
