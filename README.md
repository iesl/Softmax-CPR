# Softmax-CPR <img src="https://github.com/iesl/Softmax-CPR/blob/main/imgs/automated-external-defibrillators-g7991e1588_640.png?raw=true" width="20" height="20"> 
CPR refers to "C"ontext Partition, "P"ointer Network, and "R"eranker Partition.

![Softmax CEPR](https://github.com/iesl/Softmax-CPR/blob/main/imgs/all_partitions.png?raw=true)

## Introduction

The softmax bottleneck [(Chang and McCallum (2022))](https://aclanthology.org/2022.acl-long.554.pdf) sometimes prevents the language models from predicting the desired distribution and the pointer networks can be used to break the bottleneck efficiently. Based on the finding, we propose the context/encoder partition by simplifying the pointer networks and the reranker partition to accelerate the word-by-word rerankers. By combining these softmax alternatives, softmax-CPR is significantly better and more efficient than mixture of softmax (MoS) in GPT-2, a state-of-the-art softmax alternative. In summarization experiments, without significantly decreasing its training/testing speed, softmax-CEPR based on T5-Small improves factCC score by 2 points in CNN/DM and XSUM dataset, and improves MAUVE scores by around 30\% in BookSum paragraph-level dataset.

<p align="center"><img src="https://github.com/iesl/Softmax-CPR/blob/main/imgs/dynamic_partitions.png?raw=true" width="651" height="500"></p>


## How to Run

### For GPT2-related LM experiments
1. Put your text data into ./data (see an small example in ./data/openwebtext_2017_18_small).
2. Run the python code src/LM/preprocessing/prepare_gpt2_id_corpus_from_raw.py (change the file paths if necessary) to preprocess your text data
3. Run the script ./bin/LM/main.sh (change the python path, data paths, or different configurations if necessary) to train the model
4. Compare the validation differences from the log files


### For summarization experiments
1. Run the script ./bin/summarization/main.sh (change the python path, data paths, or different configurations if necessary) to train the model
2. Compare the validation differences from the log files


If using `conda`, you can get this to work as follows:

```
conda create -n rerankLM python=3.8
conda activate rerankLM
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch (change this 10.2 according to official website https://pytorch.org/)
conda install -c conda-forge matplotlib
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
conda install pandas
conda install nltk

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

pip install datasets
```

## Citation

Codes for ACL2023 finding paper `Revisiting the Architectures like Pointer Networks to Efficiently Improve the Next Word Distribution, Summarization Factuality, and Beyond`
