# REIDER: Towards Class-Incremental Document-Level Relation Extraction

This repository contains the codebase for our article:  
**"Towards Class-Incremental Document-Level Relation Extraction"**  
by David H. Tovmasyan and V. D. Mayorov, Center of Advanced Software Technologies, Yerevan, Armenia.

We extend the EIDER framework to support **class-incremental learning** of document-level relations. Our method decouples document encoding and classification, enabling the addition of new relation types without retraining on previous data.
Additionally, we suggest multilingual training if required.

Paper: [Towards Class-Incremental Document-Level Relation Extraction](NOT_READY_YET)  
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
## 🛠️ Environment Setup

Follow these steps to set up the environment:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv

2. **Activate the virtual environment**:

   * On Linux/macOS:

     ```bash
     source venv/bin/activate
     ```
   * On Windows:

     ```cmd
     venv\Scripts\activate
     ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

You're now ready to run the project!

---

## 📁 Dataset Setup

We use the [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset for both supervised (gold) and distantly supervised (silver) training.  
We additionally include relation types and documents from the [RedFM](https://aclanthology.org/2023.acl-long.237/) dataset for incremental learning.

### Required Structure
```
mr_Eider/
 ├── dataset/
 │   └── docred/
 │       ├── train_annotated.json
 │       ├── train_distant.json
 │       ├── dev.json
 │       ├── test.json
 │       └── (Other base and incremental datasets created from docred)
 │
 ├── redfm/
 │   ├── redfm.json
 │   └── (Other base and incremental datasets created from redfm)
 │
 ├── meta/
 │   ├── rel2id.json
 │   └── (Other metadata)
```

We use the **original `dev.json` (1000 documents)** as our validation set (there are two versions in DocRED- 998 and 1000 docs).

You can find datasets and all the required metadata **[here](https://zenodo.org/records/15861910?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE0NjJlODU4LTllY2UtNGYwZi05YzFmLTk5NGFjNzY2YTZmOCIsImRhdGEiOnt9LCJyYW5kb20iOiIwMjY3YTZhMWVmNDJkOTdmOWQ3MjEwY2RmZDdkNWUwNiJ9.V32Q5kbj7lKof4x4nzP6vjtjdvAOO1ib11xM7xSP8sgubXzdMbFTm6CtXqvy31Phs4B9rRLtCrJdWDaLYyZ4Ew)**.

---

## 🔁 Coreference Resolution for Silver Evidence

We replace the original HOI model with **[Maverick-Coref](https://aclanthology.org/2024.acl-long.722/)** (for those which did not have yet), a more accurate coreference resolution system, for generating silver evidence. Coreference results must be stored as:

```
mr_Eider/
 └── coref_results/
     ├── train_distant_coref.json
     ├── train_annotated_coref.json
     ├── dev_coref.json
     ├── test_coref.json
     └── (Other datasets' coreference results)
```
You can find the computed coreference results **[here](https://zenodo.org/records/15861910?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE0NjJlODU4LTllY2UtNGYwZi05YzFmLTk5NGFjNzY2YTZmOCIsImRhdGEiOnt9LCJyYW5kb20iOiIwMjY3YTZhMWVmNDJkOTdmOWQ3MjEwY2RmZDdkNWUwNiJ9.V32Q5kbj7lKof4x4nzP6vjtjdvAOO1ib11xM7xSP8sgubXzdMbFTm6CtXqvy31Phs4B9rRLtCrJdWDaLYyZ4Ew)**.

---

## 🚀 Training & Testing

### Original Models

**For original models training and testing scripts:**
- scripts/original_eider/train_bert.sh
- scripts/original_eider/test_bert.sh
- scripts/original_eider/train_roberta.sh
- scripts/original_eider/test_roberta.sh

usage please refer to the **[EIDER](https://github.com/yiqingxyq/Eider)** original repository.

### Pretraining with Silver + Fine-tuning with Gold (Pretraining experiments)

**Experiment(s) from the article**: 
- EIDER Distant Only (dev_F1 : 53.00%)
- EIDER Pretrain (dev_F1 : 64.76%) 

**Corresponding to the model(s)**: 
- EIDER_bert_eider__distant_only__best.pt
```bash
# Train
bash scripts/pretrain_experiments/train_bert_distant_only.sh
# Test
bash scripts/pretrain_experiments/test_bert_distant_only.sh
```

The experiment under the same settings with a larger (roberta-large) model.

**Experiment(s) from the article:**
- EIDER Pretrain RoBERTa (dev_F1 : 66.39%)

**Corresponding to the model(s):**
- EIDER_roberta_eider__exp_pretrain_2_rep__best.pt

```bash
# Train
bash scripts/pretrain_experiments/train_roberta_pretrain.sh
# Test
bash scripts/pretrain_experiments/test_roberta_pretrain.sh
```



### Incremental Learning with 1 New Relation Type and Extraction
**Experiment(s) from the article**: 
- REIDER Base 1T (dev_F1 : 62.22%)
- REIDER Head 1T (dev_F1 : 71.91%) 
- REIDER Combined 1T (dev_F1 : 61.44%)

**Corresponding to the model(s)**: 
- EIDER_bert_eider__exp_pretrain_241__best.pt
- EIDER_bert_eider__exp_incremental_241__best.pt

```bash
# Train base
bash scripts/inc1/train_bert_inc1_pretrain.sh
# Train incremental
bash scripts/inc1/train_bert_inc1_inc.sh
# Test base
bash scripts/inc1/test_bert_inc1_pretrain.sh
# Test incremental
bash scripts/inc1/test_bert_inc1_inc.sh
# Test combined ("no relation" type fusion)
bash scripts/inc1/test_bert_inc1_comb.sh
```

### Incremental Learning with 6 New Relation Types and Extraction
**Experiment(s) from the article**: 
- REIDER Base 6T (dev_F1 : 62.25%)
- REIDER Head 6T (dev_F1 : 57.52%) 
- REIDER Combined 6T (dev_F1 : 60.96%)

**Corresponding to the model(s)**: 
- EIDER_bert_eider__exp_sub6_pretrain__best.pt
- EIDER_bert_eider__exp_sub6_incremental__best.pt

```bash
# Train base
bash scripts/sub6/train_bert_sub6_pre.sh
# Train incremental
bash scripts/sub6/train_bert_sub6_inc.sh
# Test base
bash scripts/sub6/test_bert_sub6_pre.sh
# Test incremental
bash scripts/sub6/test_bert_sub6_inc.sh
# Test combined ("no relation" type fusion)
bash scripts/sub6/test_bert_sub6_comb.sh
```

### Incremental Learning with 6 New Relation Types and Supplementary Annotation
**Experiment(s) from the article**: 
- REIDER Base 6T Sup (dev_F1 : 62.37%)
- REIDER Head 6T Sup (dev_F1 : 59.26%) 
- REIDER Combined 6T Sup (dev_F1 : 61.42%)

**Corresponding to the model(s)**: 
- EIDER_bert_eider__exp_sub6_sup_pretrain__best.pt
- EIDER_bert_eider__exp_sub6_sup_incremental__best.pt

```bash
# Train base
bash scripts/sub6_sup/train_bert_sub6_sup_pre.sh
# Train incremental
bash scripts/sub6_sup/train_bert_sub6_sup_inc.sh
# Test base
bash scripts/sub6_sup/test_bert_sub6_sup_pre.sh
# Test incremental
bash scripts/sub6_sup/test_bert_sub6_sup_inc.sh
# Test combined ("no relation" type fusion)
bash scripts/sub6_sup/test_bert_sub6_sup_comb.sh
```

### Incremental Learning with 6 New Relation Types and Uniform Division
**Experiment(s) from the article**: 
- REIDER Base 6T Uni (dev_F1 : 62.06%)
- REIDER Head 6T Uni (dev_F1 : 28.57%) 
- REIDER Combined 6T Uni (dev_F1 : 61.83%)

**Corresponding to the model(s)**: 
- EIDER_bert_eider__exp_sub6_uni_pretrain__best.pt 
- EIDER_bert_eider__exp_sub6_uni_incremental__best.pt

```bash
# Train base
bash scripts/sub6_uni/train_bert_sub6_uni_pre.sh
# Train incremental
bash scripts/sub6_uni/train_bert_sub6_uni_inc.sh
# Test base
bash scripts/sub6_uni/test_bert_sub6_uni_pre.sh
# Test incremental
bash scripts/sub6_uni/test_bert_sub6_uni_inc.sh
# Test combined ("no relation" type fusion)
bash scripts/sub6_uni/test_bert_sub6_uni_comb.sh
```

### Incremental DocRED + RedFM
**Experiment(s) from the article**: 
- REIDER Base REDFM (dev_F1 : 62.31%)
- REIDER Head REDFM (dev_F1 : 51.85%) 
- REIDER Combined REDFM (dev_F1 : 62.21%)

**Corresponding to the model(s)**: 
- EIDER_bert_eider__exp_redfm_sub3_pretrain__best.pt
- EIDER_bert_eider_rule__exp_redfm_sub3_incremental__best.pt

```bash
# Train base
bash scripts/redfm_sub3/train_bert_redfm_sub3_pre.sh
# Train incremental
bash scripts/redfm_sub3/train_bert_redfm_sub3_inc.sh
# Test base
bash scripts/redfm_sub3/test_bert_redfm_sub3_pre.sh
# Test incremental
bash scripts/redfm_sub3/test_bert_redfm_sub3_inc.sh
# Test combined
# Note! In the article has been evaluated by fusing F1-s of types DocRED and RedFM
```

### Pretraining, RoBERTa-based, Incremental DocRED + RedFM
**Experiment(s) from the article**: 
- REIDER Pretrain Base REDFM (dev_F1 : 66.39%)
- REIDER Pretrain Head REDFM (dev_F1 : 60.40%) 
- REIDER Pretrain Combined REDFM (dev_F1 : 66.34%)

**Corresponding to the model(s)**: 
- EIDER_roberta_eider__exp_pretrain_2_rep__best.pt
- EIDER_roberta_eider_rule__exp_redfm_pre_sub3_incremental__best.pt

```bash
# Train base
bash scripts/pretrain_roberta_redfm/train_bert_sub6_uni_pre.sh
# Train incremental
bash scripts/pretrain_roberta_redfm/train_bert_sub6_uni_inc.sh
# Test base
bash scripts/pretrain_roberta_redfm/test_bert_sub6_uni_pre.sh
# Test incremental
bash scripts/pretrain_roberta_redfm/test_bert_sub6_uni_inc.sh
# Test combined
# Note! In the article has been evaluated by fusing F1-s of types DocRED and RedFM
```

### Multilingual Experiments
**Experiment(s) from the article**:

(THESE EXPERIMENTS ARE NOT INCLUDED IN THE ARTICLE)

**Corresponding to the model(s)**: 
- EIDER_bert_eider__exp_mbert__best.pt
- EIDER_bert_eider__exp_xlm__best.pt
- EIDER_bert_eider__exp_combined__best.pt
- EIDER_bert_eider__exp_combined_incremental__best.pt

```bash
# Train mbert
bash scripts/multilingual/train_mbert.sh
# Test mbert
bash scripts/multilingual/test_mbert.sh
# Train xlm-r
bash scripts/multilingual/train_xlmr.sh
# Test xlm-r
bash scripts/multilingual/test_xlmr.sh
# Train xlm-r pretrain base (w/o P241 type)
bash scripts/multilingual/train_xlmr_pre.sh
# Test xlm-r pretrain base
bash scripts/multilingual/test_xlmr_pre.sh
# Train xlm-r pretrain incremental (P241 type)
bash scripts/multilingual/train_xlmr_inc.sh
# Test xlm-r pretrain incremental 
bash scripts/multilingual/test_xlmr_inc.sh
# Test xlm-r pretrain combined ("no relation" type fusion)
bash scripts/multilingual/test_xlmr_comb.sh
```

(Note! If you want to test on test data (any settings), you should: change "--eval_mode dev_only" to "--eval_mode test",
pass an appropriate test data, be cautious with "saved_features" to not get confusions (and not only test time, but always) 
and after getting the results submit it to the codalab competition presented from DocRED)

## 📝 Output Tracking
For tracking outputs run:
```bash
tail -f output_name.log
```
where ```output_name.log``` is in the last line of each script.

Dev and test results after inference are provided with the datasets. Logs and individual results of each experiment are provided as well.
```individual_results_base.json``` shows the individual result of each type in basic training (EIDER_bert_eider).

---

## ⚙️ Experimental Scenarios

We simulate class-incremental learning through multiple data partitioning strategies:

1. **Extraction-based Splitting:** Select documents containing new relation types for incremental training.
2. **Uniform Splitting:** Uniformly divide the data, removing base/incremental type overlaps.
3. **Supplementary Annotation Simulation:** Simulate re-annotation of the same documents for new types.

For more information or splitting tools please check the ```./tools``` directory.

---

## 🧪 Model Variants

| Model                              | Dev F1 (%) | Encoder   | Corresponding model                                        |
|-----------------------------------|------------|-----------|------------------------------------------------------------|
| EIDER Distant Only                | 53.00      | BERT      | EIDER_bert_eider__distant_only__best.pt                    |
| EIDER Pretrain                    | 64.76      | BERT      | EIDER_bert_eider__distant_only__best.pt                    |
| EIDER Pretrain RoBERTa           | 66.39      | RoBERTa   | EIDER_roberta_eider__exp_pretrain_2_rep__best.pt           |
| REIDER Base 1T                    | 62.22      | BERT      | EIDER_bert_eider__exp_pretrain_241__best.pt                |
| REIDER Head 1T                    | 71.91      | BERT      | EIDER_bert_eider__exp_incremental_241__best.pt             |
| REIDER Combined 1T                | 61.44      | BERT      | EIDER_bert_eider__exp_incremental_241__best.pt             |
| REIDER Base 6T                    | 62.25      | BERT      | EIDER_bert_eider__exp_sub6_pretrain__best.pt               |
| REIDER Head 6T                    | 57.52      | BERT      | EIDER_bert_eider__exp_sub6_incremental__best.pt            |
| REIDER Combined 6T               | 60.96      | BERT      | EIDER_bert_eider__exp_sub6_incremental__best.pt            |
| REIDER Base 6T Sup                | 62.37      | BERT      | EIDER_bert_eider__exp_sub6_sup_pretrain__best.pt           |
| REIDER Head 6T Sup                | 59.26      | BERT      | EIDER_bert_eider__exp_sub6_sup_incremental__best.pt        |
| REIDER Combined 6T Sup           | 61.42      | BERT      | EIDER_bert_eider__exp_sub6_sup_incremental__best.pt        |
| REIDER Base 6T Uni                | 62.06      | BERT      | EIDER_bert_eider__exp_sub6_uni_pretrain__best.pt           |
| REIDER Head 6T Uni                | 28.57      | BERT      | EIDER_bert_eider__exp_sub6_uni_incremental__best.pt        |
| REIDER Combined 6T Uni           | 61.83      | BERT      | EIDER_bert_eider__exp_sub6_uni_incremental__best.pt        |
| REIDER Base RedFM                 | 62.31      | BERT      | EIDER_bert_eider__exp_redfm_sub3_pretrain__best.pt         |
| REIDER Head RedFM                 | 51.85      | BERT      | EIDER_bert_eider_rule__exp_redfm_sub3_incremental__best.pt |
| REIDER Combined RedFM            | 62.21      | BERT      | _                                                          |
| REIDER Pretrain Base RedFM       | 66.39      | RoBERTa   |EIDER_roberta_eider__exp_pretrain_2_rep__best.pt|
| REIDER Pretrain Head RedFM       | 60.40      | RoBERTa   |EIDER_roberta_eider_rule__exp_redfm_pre_sub3_incremental__best.pt|
| REIDER Pretrain Combined RedFM   | 66.34      | RoBERTa   | _                                                          |


See the paper for full experiments' descriptions.
The models can be checked and downloaded **[here](https://zenodo.org/records/15861910?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE0NjJlODU4LTllY2UtNGYwZi05YzFmLTk5NGFjNzY2YTZmOCIsImRhdGEiOnt9LCJyYW5kb20iOiIwMjY3YTZhMWVmNDJkOTdmOWQ3MjEwY2RmZDdkNWUwNiJ9.V32Q5kbj7lKof4x4nzP6vjtjdvAOO1ib11xM7xSP8sgubXzdMbFTm6CtXqvy31Phs4B9rRLtCrJdWDaLYyZ4Ew)**.

---

## 🧪 Multilingual Model Variants

| Model                        | Dev F1 (%) | Encoder | Corresponding model                                        |
|------------------------------|------------|---------|------------------------------------------------------------|
| mrEIDER_mbert_base           | 61.03      | mBERT   | EIDER_bert_eider__exp_mbert__best.pt                   |
| mrEIDER_xlm_base             | 61.61      | XLM-R   | EIDER_bert_eider__exp_xlm__best.pt                   |
| mrEIDER_xlm_pretrain_pre_241 | 64.31      | XLM-R   | EIDER_bert_eider__exp_combined__best.pt           |
| mrEIDER_xlm_pretrain_inc_241 | 75.00      | XLM-R   | EIDER_bert_eider__exp_combined_incremental__best.pt                |


These experiments are not presented in the article.
- mrEIDER_mbert_base: training with original settings on mBERT.
- mrEIDER_xlm_base: training with original settings on XLM-R.
- mrEIDER_xlm_pretrain_pre_241: pretrain on distant data (excluded P241), fine-tune on gold data (excluded P241) (trained on XLM-R)
- mrEIDER_xlm_pretrain_inc_241: by freezing base encoder pretrain on distant data (P241 only), fine-tune on gold data (P241 only) (trained on XLM-R)


The models can be checked and downloaded **[here](https://zenodo.org/records/15861910?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE0NjJlODU4LTllY2UtNGYwZi05YzFmLTk5NGFjNzY2YTZmOCIsImRhdGEiOnt9LCJyYW5kb20iOiIwMjY3YTZhMWVmNDJkOTdmOWQ3MjEwY2RmZDdkNWUwNiJ9.V32Q5kbj7lKof4x4nzP6vjtjdvAOO1ib11xM7xSP8sgubXzdMbFTm6CtXqvy31Phs4B9rRLtCrJdWDaLYyZ4Ew)**.

---

## 📊 Results

Our method demonstrates:

- **66.60%** F1 score (test) with silver+gold pretraining and RoBERTa
- Strong performance under **incremental learning without forgetting**
- Effective incorporation of **external relation types (RedFM)**

Refer to Table 1 in the paper for a full breakdown.

---

## 📬 Contact

For questions or contributions, please contact [David Tovmasyan](https://github.com/DavidTovmasyan).

---

## 🔗 Acknowledgements

This repository builds upon:
- [EIDER (ACL 2022 Findings)](https://arxiv.org/abs/2106.08657)
