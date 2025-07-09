import json
import argparse
import re
from typing import List


def simple_sent_tokenize(text: str) -> List[str]:
    """
    Splits a block of text into sentences using a basic punctuation-based rule.

    Sentences are split at '.', '!', or '?' followed by one or more spaces.

    Args:
        text (str): The input text to split.

    Returns:
        List[str]: A list of sentence strings.
    """
    return re.split(r'(?<=[.!?]) +', text)


def simple_word_tokenize(sent: str) -> List[str]:
    """
    Tokenizes a sentence into words and punctuation marks using regex.

    This tokenizer splits on word boundaries and captures punctuation as separate tokens.

    Args:
        sent (str): The input sentence.

    Returns:
        List[str]: A list of word and punctuation tokens.
    """
    return re.findall(r'\w+|[^\w\s]', sent)


def convert_redfm_to_docred(input_path: str, output_path: str):
    """
    Converts a dataset in REDFM format to the DocRED format and saves the result.

    This function processes each document by:
    - Sentence and word tokenizing the raw text.
    - Mapping character-level entity boundaries to token positions.
    - Constructing the `vertexSet` (entity mentions grouped by URI).
    - Building `labels` representing subject-predicate-object triples with evidence.

    Args:
        input_path (str): Path to the REDFM-format input JSONL file (one JSON object per line).
        output_path (str): Path to save the converted dataset in DocRED format (JSON list).

    Returns:
        None
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        redfm_data = [json.loads(line) for line in f]
        # redfm_data = json.load(f)

    docred_data = []

    for doc in redfm_data:
        text = doc["text"]
        sents_raw = simple_sent_tokenize(text)
        sents_tokens = [simple_word_tokenize(sent) for sent in sents_raw]

        # Map char position to (sent_id, token_id)
        char_to_token = {}
        char_offset = 0
        for sent_id, sent in enumerate(sents_tokens):
            for token_id, token in enumerate(sent):
                match = re.search(re.escape(token), text[char_offset:])
                if match:
                    start = char_offset + match.start()
                    for i in range(len(token)):
                        char_to_token[start + i] = (sent_id, token_id)
                    char_offset = start + len(token)

        # Build vertexSet
        entity_map = {}
        vertexSet = []
        for ent in doc["entities"]:
            start, end = ent["boundaries"]
            sent_id, token_start = char_to_token.get(start, (0, 0))
            _, token_end = char_to_token.get(end - 1, (0, 0))
            ent_obj = {
                "name": ent["surfaceform"],
                "type": ent["type"],
                "pos": [token_start, token_end + 1],
                "sent_id": sent_id
            }
            uri = ent["uri"]
            if uri not in entity_map:
                entity_map[uri] = len(vertexSet)
                vertexSet.append([ent_obj])
            else:
                vertexSet[entity_map[uri]].append(ent_obj)

        # Build labels
        labels = []
        for rel in doc.get("relations", []):
            subj_uri = rel["subject"]["uri"]
            obj_uri = rel["object"]["uri"]
            predicate = rel["predicate"]["uri"]
            evi = rel.get("sentence_id", 0)
            if subj_uri in entity_map and obj_uri in entity_map:
                labels.append({
                    "h": entity_map[subj_uri],
                    "t": entity_map[obj_uri],
                    "r": predicate,
                    "evidence": [evi] if evi < len(sents_tokens) else []
                })

        docred_doc = {
            "title": doc.get("title", ""),
            "sents": sents_tokens,
            "vertexSet": vertexSet,
            "labels": labels
        }

        docred_data.append(docred_doc)

    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(docred_data, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RED-FM JSONL to DocRED JSONL format")
    parser.add_argument("--input", required=True, help="Path to RED-FM style input file")
    parser.add_argument("--output", required=True, help="Path to output DocRED-style file")
    args = parser.parse_args()

    convert_redfm_to_docred(args.input, args.output)
