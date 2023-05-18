# dynamic-partitions

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