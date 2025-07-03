# Tools Directory

This directory contains utility tools for dataset processing and conversion, specifically designed for relation extraction and continual learning experiments.

## Overview

The tools directory includes two main utilities:

1. **`cil_filter_dataset.py`** - Dataset filtering and splitting for Continual/Incremental Learning
2. **`convert_redfm_to_docred.py`** - Format conversion from RED-FM to DocRED format

## Tools Description

### 1. CIL Dataset Filter (`cil_filter_dataset.py`)

A comprehensive tool for preparing datasets for continual learning experiments with multiple filtering and splitting strategies.

#### Features
- **Extract Mode**: Split data based on extractable relations
- **Supplementary Mode**: Split data by supplementary annotation scenario based on extractable relations
- **Coref Mode**: Filter coreference data based on relation presence and split datasets
- **Uniform Mode**: Stratified sampling for balanced splits (needs further type filtering)

#### Key Capabilities
- Multi-label stratified sampling
- Flexible relation filtering
- Coreference resolution support
- Configurable scaling parameters

### 2. RED-FM to DocRED Converter (`convert_redfm_to_docred.py`)

Converts datasets from RED-FM format to DocRED format, enabling compatibility between different relation extraction dataset formats.

#### Features
- **Format Conversion**: Transform RED-FM JSONL to DocRED JSON format
- **Entity Mapping**: Convert character-based entity positions to token-based positions
- **Relation Preservation**: Maintain all relation information during conversion
- **Sentence Tokenization**: Simple sentence and word tokenization

#### Key Capabilities
- Character-to-token position mapping
- Entity deduplication and grouping
- Evidence sentence mapping
- Robust text processing

## Usage

### CIL Dataset Filter

```bash
# Basic usage - extract mode
python cil_filter_dataset.py \
    --input_f dataset.json \
    --output_f_pretrain pretrain.json \
    --output_f_incremental incremental.json \
    --extractable_rel_id "relation_1" "relation_2"

# Uniform division with stratified sampling
python cil_filter_dataset.py \
    --input_f dataset.json \
    --output_f_pretrain pretrain.json \
    --output_f_incremental incremental.json \
    --extractable_rel_id "relation_1" "relation_2" \
    --uniformly true \
    --scale 0.1
    
# Supplementary annotation splitting
python cil_filter_dataset.py \
    --input_f dataset.json \
    --output_f_pretrain pretrain.json \
    --output_f_incremental incremental.json \
    --extractable_rel_id "relation_1" "relation_2" \
    --supplementary true
    
# Splitting the coreference results
python cil_filter_dataset.py \
    --input_f dataset.json \
    --output_f_pretrain pretrain_coref_results.json \
    --output_f_incremental incremental_coref_results.json \
    --extractable_rel_id "relation_1" "relation_2" \
    --coref_mode true \
    --coref_file coref_results.json
```

### RED-FM to DocRED Converter

```bash
# Convert RED-FM format to DocRED format
python convert_redfm_to_docred.py \
    --input redfm_dataset.jsonl \
    --output docred_dataset.json
```

## Data Formats

### CIL Dataset Filter

#### Data format (same as DocRED)
```
{
  'title',
  'sents':     [
                  [word in sent 0],
                  [word in sent 1]
               ]
  'vertexSet': [
                  [
                    { 'name': mention_name, 
                      'sent_id': mention in which sentence, 
                      'pos': postion of mention in a sentence, 
                      'type': NER_type}
                    {anthor mention}
                  ], 
                  [anthoer entity]
                ]
  'labels':   [
                {
                  'h': idx of head entity in vertexSet,
                  't': idx of tail entity in vertexSet,
                  'r': relation,
                  'evidence': evidence sentences' id
                }
              ]
}
```

#### Output Format
Same as input, but with filtered labels based on the selected mode.

### RED-FM to DocRED Converter

#### RED-FM Input Format (JSONL)
```json
{
    "docid": "Document id",
    "text": "Document text content",
    "title": "Document title",
    "uri": "Document uri",
    "entities": [
        {
            "surfaceform": "Entity mention",
            "type": "PERSON",
            "boundaries": [10, 20],
            "uri": "entity_id",
            "annotator": "Annotator"
        }
    ],
    "relations": [
        {
            "subject": {"uri": "entity_1", "boundaries":  [63, 69], "surfaceform":  "Entity mention", "annotator":  "Annotator", "type":  "ORG"},
            "object": {"uri": "entity_2", "boundaries":  [0, 15], "surfaceform":  "Entity mention", "annotator":  "Annotator", "type":  "ORG"},
            "predicate": {"uri": "relation_type", "boundaries":  null, "surfaceform":  "Relation name", "annotator":  "Annotator"},
            "sentence_id": 0,
          "dependency_path": null,
          "confidence": 0.284423828125,
          "annotator": "Annotator",
          "answer_true": 3,
          "answer_false": 0
        }
    ]
}
```

#### DocRED Output Format (JSON)
```
{
  'title',
  'sents':     [
                  [word in sent 0],
                  [word in sent 1]
               ]
  'vertexSet': [
                  [
                    { 'name': mention_name, 
                      'sent_id': mention in which sentence, 
                      'pos': postion of mention in a sentence, 
                      'type': NER_type}
                    {anthor mention}
                  ], 
                  [anthoer entity]
                ]
  'labels':   [
                {
                  'h': idx of head entity in vertexSet,
                  't': idx of tail entity in vertexSet,
                  'r': relation,
                  'evidence': evidence sentences' id
                }
              ]
}
```

## Implementation Details

### CIL Dataset Filter

- **Stratified Sampling**: Uses `MultilabelStratifiedShuffleSplit` for balanced splits
- **Memory Efficient**: Processes data in single pass where possible
- **Flexible Filtering**: Supports inclusion/exclusion of relations
- **Coreference Support**: Handles external coreference data

### RED-FM to DocRED Converter

- **Tokenization**: Simple regex-based sentence and word tokenization
- **Position Mapping**: Character-to-token position conversion
- **Entity Deduplication**: Groups multiple mentions by URI
- **Evidence Mapping**: Preserves sentence-level evidence information
