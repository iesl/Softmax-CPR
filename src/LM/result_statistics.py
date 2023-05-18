import torch
import os
import math
from ngram import ngram
from spacy.lang.en import English
import nltk
import sys

def perplexity(model, context, generated_sent):
    feature = context
    feature_generated = generated_sent
    device = next(model.parameters()).device
    feature_empty = torch.tensor((), device = device, dtype=torch.long).new_full((feature.shape[0], feature.shape[1]), -100)
    feature = torch.cat((feature, feature_generated), dim=1)
    feature_generated = torch.cat((feature_empty, feature_generated), dim=1)
    outputs_GPT2LMHeadModel= model(feature, labels=feature_generated)
    return outputs_GPT2LMHeadModel[0]

def eval_using_BLEU(generated_sentence, word_raw_list_i_j, word_raw_rest_list_i_j, spacy_nlp):
    if len(word_raw_list_i_j) <= 1:
        return -1, -1, []
    future_window_size = 25
    if len(word_raw_rest_list_i_j) <= future_window_size + 1:
        return -1, -1, []

    doc = spacy_nlp(generated_sentence)
    gen_list = []
    for tok in doc:
        gen_list.append(tok.text)
    #print(word_raw_list_i_j[:-1])
    #print(word_raw_rest_list_i_j[1:future_window_size + 1])
    #print(gen_list)
    sys.stdout.flush()
    cc = nltk.translate.bleu_score.SmoothingFunction()
    context_BLEU = nltk.translate.bleu_score.sentence_bleu([word_raw_list_i_j[:-1]], gen_list, weights = (0.5, 0.5), smoothing_function=cc.method3)
    BLEU = nltk.translate.bleu_score.sentence_bleu([word_raw_rest_list_i_j[1:future_window_size + 1]], gen_list, weights = (0.5, 0.5), smoothing_function=cc.method3)
    return BLEU, context_BLEU, gen_list

class result_statistics:

    def __init__(self, model):
        self.model_results = {}
        #self.model_results["time_count"] = 0
        self.gpt2_model = model
        self.nlp = English()

    def add_model(self, model_name):
        self.model_results[model_name] = {}
        #self.model_results[model_name]["batch count"] = 0
        self.model_results[model_name]["count"] = 0
        self.model_results[model_name]["BLEU_count"] = 0
        self.model_results[model_name]["BLEU_count_arr"] = []
        self.model_results[model_name]["BLEU"] = 0
        self.model_results[model_name]["BLEU_arr"] = []
        self.model_results[model_name]["context_len_arr"] = []
        self.model_results[model_name]["context BLEU"] = 0
        self.model_results[model_name]["gen_list_temp"] = []
        self.model_results[model_name]["self BLEU count"] = 0
        self.model_results[model_name]["self BLEU"] = 0
        self.model_results[model_name]["self BLEU arr"] = []
        self.model_results[model_name]["self BLEU count arr"] = []
        self.model_results[model_name]["perplexity"] = 0
        self.model_results[model_name]["ngram"] = ngram()
        self.model_results[model_name]["ngram count"] = 0
        self.model_results[model_name]["ngram count arr"] = []
        self.model_results[model_name]["unigram"] = 0
        self.model_results[model_name]["bigram"] = 0
        self.model_results[model_name]["unigram arr"] = []
        self.model_results[model_name]["bigram arr"] = []
        self.model_results[model_name]["time_sum"] = 0
        self.model_results[model_name]["time_count"] = 0

    def update_self_BLEU(self, model_name, head_idx):
        count = 0
        gen_list_temp = self.model_results[model_name]["gen_list_temp"]
        num_sent_gen = len(gen_list_temp)
        cc = nltk.translate.bleu_score.SmoothingFunction()
        for i in range(num_sent_gen):
            for j in range(i+1,num_sent_gen):
                self_BLEU = nltk.translate.bleu_score.sentence_bleu([gen_list_temp[i]],gen_list_temp[j], weights = (0.5, 0.5), smoothing_function=cc.method3)
                self.model_results[model_name]["self BLEU count"] += 1
                self.model_results[model_name]["self BLEU"] += self_BLEU
                self.model_results[model_name]["self BLEU count arr"][head_idx] += 1
                self.model_results[model_name]["self BLEU arr"][head_idx] += self_BLEU
        self.model_results[model_name]["gen_list_temp"] = []
        

    def update(self, model_name, sentence, context, tokenizer, word_raw_list_i_j=None, word_raw_rest_list_i_j=None, j = None):
        generated_sent = tokenizer.convert_tokens_to_string( [tokenizer._convert_id_to_token(x) for x in sentence.tolist()] )
        #num_token_hit, num_word_type_hit, num_topic_hit = sample_statistics(selected_topic_idx, generated_sent, top_index, idx2word_freq)
        log_perplexity = perplexity(self.gpt2_model, context.unsqueeze(0), sentence.unsqueeze(0))
        if word_raw_list_i_j is not None:
            #context_sents = tokenizer.convert_tokens_to_string( [tokenizer._convert_id_to_token(x) for x in context.tolist()] )
            while j >= len(self.model_results[model_name]['BLEU_count_arr']):
                self.model_results[model_name]['BLEU_count_arr'].append(1e-15)
                self.model_results[model_name]['BLEU_arr'].append(0)
                self.model_results[model_name]['self BLEU arr'].append(0)
                self.model_results[model_name]['self BLEU count arr'].append(1e-15)
                self.model_results[model_name]['context_len_arr'].append(0)
                self.model_results[model_name]['unigram arr'].append(0)
                self.model_results[model_name]['bigram arr'].append(0)
                self.model_results[model_name]['ngram count arr'].append(1e-15)
            BLEU, context_BLEU, gen_list = eval_using_BLEU(generated_sent, word_raw_list_i_j, word_raw_rest_list_i_j, self.nlp)
            if BLEU != -1:
                self.model_results[model_name]["gen_list_temp"].append(gen_list)
                self.model_results[model_name]["BLEU"] += BLEU
                self.model_results[model_name]["context BLEU"] += context_BLEU
                self.model_results[model_name]["BLEU_count"] += 1
                self.model_results[model_name]['BLEU_count_arr'][j] += 1
                self.model_results[model_name]['BLEU_arr'][j] += BLEU
                self.model_results[model_name]['context_len_arr'][j] += len(word_raw_list_i_j[:-1])
        self.model_results[model_name]["count"] += 1
        self.model_results[model_name]["perplexity"] += log_perplexity
        self.model_results[model_name]["ngram"].add(generated_sent)

    
    def renew_ngram(self, head_idx):
        for model_name, model in self.model_results.items():
            model["ngram count"] += 1
            model["ngram count arr"][head_idx] += 1
            unigram, bigram = model["ngram"].diversity_n()
            #print(model["ngram"].diversity_n())
            model["unigram"] += unigram
            model["bigram"] += bigram
            model["unigram arr"][head_idx] += unigram
            model["bigram arr"][head_idx] += bigram
            model["ngram"] = ngram()
            #print(model["ngram"].diversity_n())


    def print(self):
        for model_name, model in self.model_results.items():
            print(model_name, "count: ", model["count"])
            print(model_name, "BLEU count: ", model["BLEU_count"])
            #print(model_name, "batch count: ", model["batch count"])
            print(model_name, "BLEU: ", model["BLEU"] / model["BLEU_count"])
            print(model_name, "BLEU_arr: "+ str( [model["BLEU_arr"][x] / model["BLEU_count_arr"][x] if model["BLEU_count_arr"][x] > 0 else 0 for x in range(len(model["BLEU_count_arr"]))] ) )
            print(model_name, "context_len_arr: "+ str( [model["context_len_arr"][x] / model["BLEU_count_arr"][x] if model["BLEU_count_arr"][x] > 0 else 0 for x in range(len(model["BLEU_count_arr"]))] ) )
            print(model_name, "self BLEU: ", model["self BLEU"] / (1e-13+model["self BLEU count"]))
            print(model_name, "self BLEU arr: "+ str( [model["self BLEU arr"][x] / model["self BLEU count arr"][x] if model["self BLEU count arr"][x] > 0 else 0 for x in range(len(model["self BLEU count arr"]))] ) )
            print(model_name, "context BLEU: ", model["context BLEU"] / model["BLEU_count"])
            print(model_name, "BLEU Diff: " + str(model["BLEU"] / model["BLEU_count"] - model["context BLEU"] / model["BLEU_count"] ) )
            print(model_name, "perplexity: ", math.exp(model["perplexity"] / model["count"]))
            print(model_name, "unigram: ", model["unigram"] / model["ngram count"])
            print(model_name, "bigram: ", model["bigram"] / model["ngram count"])
            print(model_name, "unigram arr: "+ str( [model["unigram arr"][x] / model["ngram count arr"][x] if model["ngram count arr"][x] > 0 else 0 for x in range(len(model["ngram count arr"]))] ) )
            print(model_name, "bigram arr: "+ str( [model["bigram arr"][x] / model["ngram count arr"][x] if model["ngram count arr"][x] > 0 else 0 for x in range(len(model["ngram count arr"]))] ) )
            if model['time_count'] > 0:
                print(model_name, "running time: ", model["time_sum"] / model['time_count'])
            print()



    def generate_report(self, outf):
        outf.write('Reports: \n')
        for model_name, model in self.model_results.items():
            outf.write(model_name + " " + "count: " + str(model["count"]) + '\n')
            outf.write(model_name + " " + "BLEU count: " + str(model["BLEU_count"]) + '\n')
            #outf.write(model_name + " " + "batch count: " + str(model["batch count"]) + '\n')
            outf.write(model_name + " " + "BLEU: " + str(model["BLEU"] / model["BLEU_count"]) + '\n')
            outf.write(model_name + " " + "BLEU_arr: "+ str( [model["BLEU_arr"][x] / model["BLEU_count_arr"][x] if model["BLEU_count_arr"][x] > 0 else 0 for x in range(len(model["BLEU_count_arr"]))] ) + '\n')
            outf.write(model_name + " " + "context_len_arr: "+ str( [model["context_len_arr"][x] / model["BLEU_count_arr"][x] if model["BLEU_count_arr"][x] > 0 else 0 for x in range(len(model["BLEU_count_arr"]))] ) + '\n')
            outf.write(model_name + " " + "self BLEU: " + str(model["self BLEU"] / (1e-13+ model["self BLEU count"])) + '\n')
            outf.write(model_name + " self BLEU arr: "+ str( [model["self BLEU arr"][x] / model["self BLEU count arr"][x] if model["self BLEU count arr"][x] > 0 else 0 for x in range(len(model["self BLEU count arr"]))] ) + '\n' )
            outf.write(model_name + " " + "context BLEU: " + str(model["context BLEU"] / model["BLEU_count"]) + '\n')
            outf.write(model_name + " " + "BLEU Diff: " + str(model["BLEU"] / model["BLEU_count"] - model["context BLEU"] / model["BLEU_count"] ) + '\n')
            outf.write(model_name + " " + "perplexity: " + str(math.exp(model["perplexity"] / model["count"])) + '\n')
            outf.write(model_name + " " + "unigram: " + str(model["unigram"] / model["ngram count"]) + '\n')
            outf.write(model_name + " " + "bigram: " + str(model["bigram"] / model["ngram count"]) + '\n')
            outf.write(model_name + " unigram arr: "+ str( [model["unigram arr"][x] / model["ngram count arr"][x] if model["ngram count arr"][x] > 0 else 0 for x in range(len(model["ngram count arr"]))] ) )
            outf.write(model_name + " bigram arr: "+ str( [model["bigram arr"][x] / model["ngram count arr"][x] if model["ngram count arr"][x] > 0 else 0 for x in range(len(model["ngram count arr"]))] ) )
            if model['time_count'] > 0:
                outf.write(model_name + " " + "running time: " + str(model["time_sum"] / model['time_count']) + '\n')
            outf.write('\n')
    

