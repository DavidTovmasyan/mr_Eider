# REIDER: Towards Class-Incremental Document-Level Relation Extraction

This repository contains the codebase for our IEEE Intelligent Systems 2025 article:  
**"Towards Class-Incremental Document-Level Relation Extraction"**  
by David H. Tovmasyan and V. D. Mayorov, Center of Advanced Software Technologies, Yerevan, Armenia.

We extend the EIDER framework to support **class-incremental learning** of document-level relations. Our method decouples document encoding and classification, enabling the addition of new relation types without retraining on previous data.

Paper: [IEEE Xplore (forthcoming)](https://ieeexplore.ieee.org/)  
Code: [GitHub Repository](https://github.com/DavidTovmasyan/mr_Eider)

---

## 🔍 Overview

Most existing DocRE models assume a fixed set of relation types. In contrast, we simulate a realistic scenario where **new relation types and documents arrive over time**, and demonstrate how to:

- Pretrain on large-scale silver (distantly supervised) data.
- Fine-tune using gold (annotated) data.
- Add new relation types via **plug-in classification heads** without retraining the encoder.
- Avoid catastrophic forgetting using frozen feature extractors.
- Apply **“no relation” class fusion during inference**.

We evaluate our approach across multiple training configurations using DocRED and RedFM datasets.
(For dataset and coreference splitting/filtration and datasets themselves information please check ./tools)
---

## 📁 Dataset Setup

We use the [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset for both supervised (gold) and distantly supervised (silver) training.  
We additionally include relation types and documents from the [RedFM](https://aclanthology.org/2023.acl-long.630/) dataset for incremental learning.

### Required Structure
```
REIDER/
 ├── dataset/
 │   └── docred/
 │       ├── train_annotated.json
 │       ├── train_distant.json
 │       ├── dev.json
 │       ├── test.json
 │
 ├── redfm/
 │   └── redfm_incremental.json
 │
 ├── meta/
 │   └── rel2id.json
```

We use the **original `dev.json` (1000 documents)** as our validation set.

---

## 🔁 Coreference Resolution for Silver Evidence

We replace the original HOI model with **[Maverick-Coref](https://aclanthology.org/2024.acl-long.1033/)**, a more accurate coreference resolution system, for generating silver evidence. Coreference results must be stored as:

```
REIDER/
 └── coref_results/
     ├── train_distant_coref.json
     ├── train_annotated_coref.json
     ├── dev_coref.json
     └── test_coref.json
```

---

## 🚀 Training & Inference

### Pretraining with Silver + Fine-tuning with Gold
```bash
bash scripts/train_bert.sh reider_pretrain gold_train maverick
bash scripts/test_bert.sh reider_pretrain dev maverick
```

### Incremental Learning with New Relation Types
```bash
bash scripts/train_incremental.sh reider_incremental base maverick
bash scripts/fine_tune_new_heads.sh reider_incremental incremental maverick
```

### Combined Inference with “No Relation” Class Fusion
```bash
bash scripts/combined_inference.sh reider_combined
```

---

## ⚙️ Experimental Scenarios

We simulate class-incremental learning through multiple data partitioning strategies:

1. **Extraction-based Splitting:** Select documents containing new relation types for incremental training.
2. **Uniform Splitting:** Uniformly divide the data, removing base/incremental type overlaps.
3. **Supplementary Annotation Simulation:** Simulate re-annotation of the same documents for new types.

---

## 🧪 Model Variants

| Model                         | Description |
|------------------------------|-------------|
| `REIDER Base`                | Trained on base types only |
| `REIDER Head`                | New head trained on incremental types |
| `REIDER Combined`            | Combined inference with fusion |
| `REIDER Pretrain`            | Pretrained on silver data |
| `REIDER Pretrain RoBERTa`    | Pretraining with RoBERTa-large |
| `REIDER RedFM`               | Incremental learning using RedFM types |

See the paper for full experimental results.

---

## 📊 Results

Our method demonstrates:

- **66.60%** F1 score with silver+gold pretraining and RoBERTa
- Strong performance under **incremental learning without forgetting**
- Effective incorporation of **external relation types (RedFM)**

Refer to Table 1 in the paper for a full breakdown.

---

## 🧾 Citation

If you use this code, please cite:

```
@article{tovmasyan2025reider,
  title={Towards Class-Incremental Document-Level Relation Extraction},
  author={Tovmasyan, David H. and Mayorov, V. D.},
  journal={IEEE Intelligent Systems},
  year={2025},
  publisher={IEEE}
}
```

---

## 📬 Contact

For questions or contributions, please contact [David Tovmasyan](https://github.com/DavidTovmasyan).

---

## 🔗 Acknowledgements

This repository builds upon:
- [EIDER (ACL 2022 Findings)](https://arxiv.org/abs/2106.08657)
- [ATLOP (AAAI 2021)](https://arxiv.org/abs/2010.11304)
- [RedFM Dataset (ACL 2023)](https://aclanthology.org/2023.acl-long.630/)
