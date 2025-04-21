import json
from typing import Iterable
import argparse


# def get_filtered_dataset(input_file_path: str, to_be_extracted: Iterable[list], output_file_path: str):
#     with open(input_file_path, "r") as f:
#         data = json.load(f)

#     print(f"The input file size is: {len(data)}.\n")

#     for i in range(len(data)):
#         new_labels = filter(lambda d: d["r"] not in to_be_extracted, data[i]["labels"])
#         data[i]["labels"] = list(new_labels)

#     with open(output_file_path, "w") as f:
#         json.dump(data, f)

#     print(f"Filtration was successfull! Results are saved in {output_file_path}")

# def get_relations(rel_info_path: str):
#     with open(rel_info_path, "r") as f:
#         rels = json.load(f)
#     print("Filter relations were gotten successfully!")
#     return rels

# def get_incremental_dataset(input_file_path: str, to_be_incremented: str, output_file_path: str):
#     with open(input_file_path, "r") as f:
#         data = json.load(f)

#     print(f"The input file size is: {len(data)}.\n")
#     for i in range(len(data)):
#         new_labels = filter(lambda d: d["r"] == to_be_incremented, data[i]["labels"])
#         data[i]["labels"] = list(new_labels)

#     with open(output_file_path, "w") as f:
#         json.dump(data, f)

#     print(f"Filtration was successfull! Results are saved in {output_file_path}")

def filter_dataset(input_file: str, output_file_pretrain: str, output_file_incremental: str, extractable_rel_id: str) -> None:
    with open(input_file, 'r') as f:
        initial_data = json.load(f)

    print(f"{input_file} was loaded successfully!")

    pretrain_data = []
    incremental_data = []
    
    for sample in initial_data:
        is_present = False
        for label in sample["labels"]:
            if label['r'] == extractable_rel_id:
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

def filter_sample(sample: dict, ext_id: str, preserve=True):
    new_sample = {}
    for k, v in sample.items():
        if k != "labels":
            new_sample[k] = v
        else:
            if preserve:
                new_sample[k] = list(filter(lambda d: d["r"] == ext_id, v))
            else:
                new_sample[k] = list(filter(lambda d: d["r"] != ext_id, v))
    return new_sample
            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_f", default="", type=str)
    parser.add_argument("--output_f_pretrain", default="", type=str)
    parser.add_argument("--output_f_incremental", default="", type=str)
    parser.add_argument("--extractable_rel_id", default="", type=str)
    args = parser.parse_args()
    
    filter_dataset(args.input_f, args.output_f_pretrain, args.output_f_incremental, args.extractable_rel_id)



if __name__ == "__main__":
    main()