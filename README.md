# Incremental Reasoning

This repo contains tools to reproduce the work [ Can We Guide a Multi-Hop Reasoning Language Model to Incrementally Learn at each Hop?](https://aclanthology.org/2022.coling-1.125/).

# Artificial Datasets
Download the data for the multi-hop reasoning experiments and downstream tasks in the following tables

## Generated datasets
| Dataset                                                                                          | Hops Number | Description              |
|--------------------------------------------------------------------------------------------------|-------------|--------------------------|
| [SHINET](https://drive.google.com/file/d/169CS6Q3-O1sL2oiEpL_moFjYBRMxcPuh/view?usp=sharing)     | 1           |                          |
| [SHINET](https://drive.google.com/file/d/1fiNQqseC0_60ymObxm31IzlWOS_R10Tr/view?usp=sharing)     | 2           | Distractors included     |
| [RuleTakers](https://drive.google.com/file/d/1dhROWEM1a1DRNhntw_huM7nJFIu3SJld/view?usp=sharing) | 0 - 5       | Filtered by single hops. innoculation files  |
| [RACE](https://drive.google.com/file/d/1SIWO9G0LmPz8mIbl-LJ35dUcPH2oFFFH/view?usp=sharing)                                                                                         | -           | Innoculation files       |

## External Resources
| Dataset                                                                                                         | Source                                          | 
|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| [LeapOfThought](https://github.com/alontalmor/LeapOfThought/blob/master/README.md#artisets-artificial-datasets) | Leap-Of-Thought - Talmor et al.                 | 
| [Semantic Fragments](https://github.com/allenai/semantic_fragments/tree/master/scripts_mcqa)                    | What does my QA model knows? - Richardson et al. |

## Trained models (HuggingFace Transformers  version)
| Model       | Description                     | Link     |
|-------------|---------------------------------|----------|
| RoBERTa     | Trained on SHINet 2-hop (g-inf) | [link](https://drive.google.com/file/d/1DcL5sbYg8P13pA4brY6TYiykCWvr5QpP/view?usp=sharing) |
| XLNet       | Trained on SHINet 2-hop (g-inf) | [link](https://drive.google.com/file/d/1RQIQ0XiYNX0h60iHBwZJodGlqtZokf3B/view?usp=sharing) |
| BERT | Trained on SHINet 2-hop (g-inf) | [link](https://drive.google.com/file/d/1DudzWUtl5D7C-oXl0-t9grPdX5B4O9Hb/view?usp=sharing) |

To load the models:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained(path)
```


# Setup
We used the [AllenNLP](https://github.com/allenai/allennlp) framework to perform the experiments.

## 1. Requirements
- Python 3.8
- AllenNLP 2.8.0

## 2. Create and activate the environment
```commandline
git clone 
cd inc_reasoning
conda env create -f env.yml
conda activate inchop
```

## 3. Training and Evaluation
With the **config** and **inc_reason** folders we obtain similar results to Table 2 and 3 from our paper. 
 
### For training the models:

1. For BERT 
```
python -m allennlp train config/bert.jsonnet -s output/test_bert --include-package inc_reason
```

2. For RoBERTa 
```
python -m allennlp train config/roberta.jsonnet -s output/test_roberta --include-package inc_reason
```
3. For XLNet 
```
python -m allennlp train config/xlnet.jsonnet -s output/test_xlnet --include-package inc_reason
```

Change in the jsonnet files the input files for training and dev in the **train_data_path** and **validation_data_path** fields. 


#### - Example to train RoBERTa with 1-hop SHINET
1. Modify the config file (config/roberta.jsonnet file)
```
{
...
  //"datasets_for_vocab_creation": [],
  "train_data_path": "/data/shinet-1hop/shinet_h1_sinf.train.jsonl",
  "validation_data_path": "/data/shinet-1hop/shinet_h1_sinf.dev.jsonl",
...
}
```

### For Evaluation:
Evaluate model on SHINET-H1 (SH1)
```
python -m allennlp evaluate YOUR_OUTPUT_FOLDER/model.tar.gz /data/shinet-1hop/shinet_h1_sinf.test.jsonl --include-package inc_reason
```

4. Evaluate model on SHINET-H2 (SH2)
```
python -m allennlp evaluate YOUR_OUTPUT_FOLDER/model.tar.gz /data/shinet-2hop/shinet_h2_sinf.test.jsonl --include-package inc_reason
```

## 4. Downstream Tasks

The trained models are tested using the code from [Semantic Fragments - Allen AI](https://github.com/allenai/semantic_fragments)


# AFLITE Algorithm for Algorithmic Data Bias Reduction

### To apply AFLITE to your data:
```
python main.py --config config/test1.yaml
```
Input data, output directory and parameters for the algorithm can be modified in *config/test1.yaml* file or directly in the *config/default.yaml* file.

# Citation
```
@inproceedings{lovon-2022-guide,
    title = "Can We Guide a Multi-Hop Reasoning Language Model to Incrementally Learn at Each Single-Hop?",
    author = "Lovon-Melgarejo, Jesus  and
      Moreno, Jose G.  and
      Besan{\c{c}}on, Romaric  and
      Ferret, Olivier  and
      Tamine, Lynda",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.125",
    pages = "1455--1466",
}
```
