import argparse
import json
from time import time

import torch
from maverick import Maverick


def generate_maps(sentences, tokens):
    """
    Generate mapping indices for sentences and tokens.

    Creates mappings between sentence-level and token-level representations,
    where each token is mapped to its corresponding sentence index.

    Args:
        sentences (list): List of sentences, where each sentence is a list of tokens
        tokens (list): Flattened list of all tokens across all sentences

    Returns:
        tuple: A tuple containing:
            - sentence_map (list): List where each index corresponds to a token position
              and the value is the sentence index that token belongs to
            - subtoken_map (list): List of sequential indices from 0 to len(tokens)-1
    """
    sentence_map = []
    sentence_index = 0

    for sentence in sentences:
        for token in sentence:
            sentence_map.append(sentence_index)
        sentence_index += 1

    return sentence_map, list(range(len(tokens)))


def construct_coref(docred_sample, model):
    """
    Construct coreference resolution results for a DocRED sample.

    Takes a DocRED document sample and applies coreference resolution using the
    provided model, then formats the results in HOI (Higher Order Inference) format
    for downstream processing.

    Args:
        docred_sample (dict): A DocRED document sample containing:
            - 'sents': List of sentences, where each sentence is a list of tokens
        model: Coreference resolution model with a predict() method that returns
            predictions with 'tokens' and 'clusters_token_offsets' keys

    Returns:
        dict: HOI-formatted coreference result containing:
            - 'sentences': List containing the flattened token sequence
            - 'sentence_map': Mapping from token indices to sentence indices
            - 'subtoken_map': Sequential token indices
            - 'predicted_clusters': Coreference clusters as token offset ranges
    """
    sentences = docred_sample['sents']
    predictions = model.predict(sentences)
    tokens = predictions['tokens']
    clusters = predictions['clusters_token_offsets']

    sentence_map, subtoken_map = generate_maps(sentences, tokens)

    return {'sentences': [tokens],
            'sentence_map': sentence_map,
            'subtoken_map': subtoken_map,
            'predicted_clusters': clusters}


def get_corefs(file_in: str, file_out: str, model):
    """
    Process coreference resolution for all samples in a DocRED dataset file.

    Reads a JSON file containing DocRED samples, applies coreference resolution
    to each sample using the provided model, and saves the results to an output file.
    Provides progress tracking and timing information during processing.

    Args:
        file_in (str): Path to input JSON file containing DocRED samples
        file_out (str): Path to output JSON file where coreference results will be saved
        model: Coreference resolution model with a predict() method

    Returns:
        list: List of coreference resolution results for all processed samples

    Side Effects:
        - Reads from file_in
        - Writes processed results to file_out
        - Prints progress information to stdout including:
            - Current sample being processed
            - Average processing time per sample
    """
    with open(file_in, "r") as f:
        data = json.load(f)

    start = time()
    results = []

    for i, sample in enumerate(data):
        print(f"Processing the {i}-th sample of {len(data)}.")
        results.append(construct_coref(sample, model))
        print(f"Avg processing time: {(time()-start)/(i+1)} seconds.\n")

    with open(file_out, 'w') as f:
        json.dump(results, f)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_in", type=str)
    parser.add_argument("--file_out", type=str)
    args = parser.parse_args()

    # Setting the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    # Usage
    model = Maverick(device=device)

    get_corefs(args.file_in, args.file_out, model)


if __name__ == "__main__":
    main()
