import json
from typing import Iterable
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np


def filter_dataset(input_file: str, output_file_pretrain: str, output_file_incremental: str,
                   extractable_rel_id: Iterable[str]) -> None:
    """
    Splits a dataset into two subsets based on the presence of specified relation IDs in each sample's labels,
    and writes the subsets to separate output files.

    The dataset is loaded from a JSON file where each sample contains a "labels" field with relation entries.
    Samples containing at least one label with a relation ID (`"r"`) in `extractable_rel_id` are included in
    the "incremental" dataset. All other samples go to the "pretrain" dataset.

    Each sample is filtered using `filter_sample()`:
        - For the incremental dataset, only labels with matching relation IDs are preserved.
        - For the pretrain dataset, matching relation IDs are removed from the labels.

    Args:
        input_file (str): Path to the input JSON file containing the dataset.
        output_file_pretrain (str): Path to save the filtered pretrain dataset.
        output_file_incremental (str): Path to save the filtered incremental dataset.
        extractable_rel_id (Iterable[str]): A collection of relation IDs used to split and filter the data.

    Returns:
        None
    """

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
    """
    Processes a supplementary dataset by generating two versions: one for incremental learning and one for pretraining.

    The dataset is read from a JSON file where each sample contains a "labels" field.

    - The **incremental dataset** includes all original samples unchanged.
    - The **pretrain dataset** includes:
        - A version of each sample with relation IDs in `extractable_rel_id` removed (if present in the sample).
        - Or the original sample if none of its labels match the `extractable_rel_id`.

    This ensures that the pretrain data does not contain labels intended for incremental learning,
    while still preserving all samples for the incremental dataset.

    Args:
        input_file (str): Path to the input JSON file containing the supplementary dataset.
        output_file_pretrain (str): Path to save the pretrain dataset (filtered if needed).
        output_file_incremental (str): Path to save the unmodified incremental dataset.
        extractable_rel_id (Iterable[str]): Relation IDs to be excluded from the pretrain dataset.

    Returns:
        None
    """

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
    """
    Filters a coreference resolution dataset into pretrain and incremental subsets based on relation IDs
    present in a separate labeled dataset.

    The function uses the `input_file` (containing samples with labeled relations) to determine which samples
    are relevant for incremental learning (i.e., they contain at least one label with a relation ID in
    `extractable_rel_id`). This relevance is captured in a binary mask.

    Then, the corresponding entries in `coref_file` (a separate coreference-annotated dataset assumed to be
    aligned in order with `input_file`) are split into:
    - **Incremental data**: Coref samples corresponding to samples that contain any of the `extractable_rel_id`.
    - **Pretrain data**: Coref samples corresponding to all other samples.

    Args:
        input_file (str): Path to a JSON file containing samples with labeled relations.
        output_file_pretrain (str): Path to save the filtered pretrain coreference dataset.
        output_file_incremental (str): Path to save the filtered incremental coreference dataset.
        extractable_rel_id (Iterable[str]): A collection of relation IDs used to determine relevance.
        coref_file (str): Path to the JSON file containing coreference-annotated samples, aligned with `input_file`.

    Returns:
        None
    """

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
    """
    Filters the 'labels' field of a data sample based on a set of relation IDs.

    If `preserve` is True, only labels with relation IDs in `ext_id` are kept.
    If `preserve` is False, labels with relation IDs in `ext_id` are removed.
    All other fields in the sample are preserved unchanged.

    Args:
        sample (dict): A single data sample containing a "labels" field and other metadata.
        ext_id (Iterable[str]): A collection of relation IDs to include or exclude.
        preserve (bool, optional): Determines whether to keep (`True`) or remove (`False`) matching labels.
                                   Defaults to True.

    Returns:
        dict: A new sample with the filtered labels.
    """
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


def divide_uniformly(file_in: str, file_out_1: str, file_out_2: str,
                     extractable: Iterable[str], scale: float = 0.1):
    """
    Splits a dataset into two subsets using iterative stratification to ensure uniform distribution
    of relation types, and filters each subset based on extractable relation IDs.

    The function reads a JSON dataset from `file_in`, where each sample contains a list of labels with
    relation IDs (`"r"`). It builds a multi-hot label matrix to represent the presence of relation types
    across samples, and uses `MultilabelStratifiedShuffleSplit` to divide the dataset while preserving
    the label distribution.

    After splitting:
    - **Set 1** contains samples from the first split (`file_out_1`) with only labels **not in** `extractable`.
    - **Set 2** contains samples from the second split (`file_out_2`) with only labels **in** `extractable`.

    Args:
        file_in (str): Path to the input JSON file containing the dataset.
        file_out_1 (str): Path to save the first output subset (non-extractable relations only).
        file_out_2 (str): Path to save the second output subset (extractable relations only).
        extractable (Iterable[str]): Relation IDs used to filter each subset.
        scale (float, optional): Proportion of the dataset to allocate to `file_out_2`.
                                 Default is 0.1 (i.e., 10%).

    Returns:
        None
    """

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
