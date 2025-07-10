# Tools Directory

This directory contains utility tools for dataset processing and conversion, specifically designed for relation extraction and continual learning experiments.

## Overview

The tools directory includes three main utilities:

1. **`cil_filter_dataset.py`** - Dataset filtering and splitting for Continual/Incremental Learning
2. **`convert_redfm_to_docred.py`** - Format conversion from RED-FM to DocRED format
3. **`coref_processing.py`** - Coreference resolution processing for DocRED datasets

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

### 3. Coreference Processing (`coref_processing.py`)

Processes DocRED datasets to extract coreference information using the Maverick coreference resolution model, converting results to HOI (Higher Order Inference) format for downstream relation extraction tasks.

#### Features
- **Coreference Resolution**: Apply state-of-the-art coreference resolution to DocRED samples
- **HOI Format Conversion**: Convert coreference results to HOI format compatible with relation extraction models
- **Batch Processing**: Process entire datasets with progress tracking and timing information
- **GPU Support**: Leverages CUDA acceleration for efficient processing

#### Key Capabilities
- Sentence-to-token mapping generation
- Coreference cluster prediction and formatting
- Progress monitoring with timing statistics
- Integration with Maverick coreference model

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

### Coreference Processing

```bash
# Process DocRED dataset for coreference resolution
python coref_processing.py \
    --file_in train_distant.json \
    --file_out train_distant_coref_results.json
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

### Coreference Processing

#### Input Format
Standard DocRED format (same as above)

#### Output Format (HOI Format)
```json
{
  "sentences": [["token1", "token2", "token3", ...]],
  "sentence_map": [0, 0, 0, 1, 1, 2, 2, 2, ...],
  "subtoken_map": [0, 1, 2, 3, 4, 5, 6, 7, ...],
  "predicted_clusters": [
    [[0, 2], [5, 7]], 
    [[10, 12], [15, 16]]
  ]
}
```

Where:
- `sentences`: Flattened token sequence
- `sentence_map`: Maps each token to its sentence index
- `subtoken_map`: Sequential token indices
- `predicted_clusters`: Coreference clusters as token offset ranges

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

### Coreference Processing

- **Model Integration**: Uses Maverick coreference resolution model with CUDA support
- **Efficient Processing**: Batch processing with progress tracking and timing statistics
- **Format Conversion**: Converts between DocRED and HOI formats for compatibility
- **Mapping Generation**: Creates sentence-to-token and subtoken mappings for downstream tasks
