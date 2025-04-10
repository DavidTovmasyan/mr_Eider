# Eider
Code for ACL 2022 Finding paper "[EIDER: Empowering Document-level Relation Extraction with Efficient Evidence Extraction and Inference-stage Fusion](https://arxiv.org/abs/2106.08657)"

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://drive.google.com/drive/folders/1owp7ZRbrMl_s1ljIh6AvnmniLJSliV6h?usp=sharing).
Noted that the dev.json file has been modified in Aug, 2021. The modified version contains 998 documents. We use the original version of dev.json, which contains 1000 documents.

The expected structure of files is:
```
Eider
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |-- meta
 |    |-- rel2id.json
 ```
 
 ## Coreference Results Required by Eider_rule
 We use [hoi](https://github.com/emorynlp/coref-hoi) as the coreference model for Eider_rule. The processed data can be found [here](https://drive.google.com/drive/folders/1xceCD96VUbqZ4-IDVICCBIpke5VlZedz?usp=sharing).
 
 The expected structure of files is:
```
Eider
 |-- coref_results
 |    |-- train_annotated_coref_results.json
 |    |-- dev_coref_results.json
 |    |-- test_coref_results.json
 ```
 
 
 ## Training and Inference
 Train Eider-BERT on DocRED with the following commands:
 ```
 >> bash scripts/train_bert.sh eider test hoi
 >> bash scripts/test_bert.sh eider test hoi
 ```
 
 Alternatively, you can train Eider-RoBERTa using:
 ```
 >> bash scripts/train_roberta.sh eider test hoi
 >> bash scripts/test_roberta.sh eider test hoi
 ```
 
 The commands for Eider_rule is similar:
 ```
 >> bash scripts/train_bert.sh eider_rule test hoi # BERT
 >> bash scripts/test_bert.sh eider_rule test hoi

 >> bash scripts/train_roberta.sh eider_rule test hoi # RoBERTa
 >> bash scripts/test_roberta.sh eider_rule test hoi
 ```
 
 ## Citation
 If you make use of this code in your work, please kindly cite the following paper:
```
@inproceedings{xie2021eider,
      title={EIDER: Empowering Document-level Relation Extraction with Efficient Evidence Extraction and Inference-stage Fusion}, 
      author={Yiqing Xie and Jiaming Shen and Sha Li and Yuning Mao and Jiawei Han},
      year={2022},
      booktitle = {Findings of the 60th Annual Meeting of the Association for Computational Linguistics},
      publisher = {Association for Computational Linguistics},
}
```
 
