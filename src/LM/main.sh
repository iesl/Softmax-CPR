#!/bin/bash

python src/pretraining/main.py --save models/pretraining/test --data data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 3 --n_facet 1 --n_facet_context 0 --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode add_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt


# # test
# python src/pretraining/main.py --save models/pretraining/test --load_file_name LM_weights_1_7.pt --continue_train --simple_eval True --model_name gpt2-medium --data data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 2 --n_facet 1  --n_facet_context 1 --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True



# different methods
# python src/LM/main_GPT2.py --save ./models/nall1_nfacet1_context0_pointer0_reranker0_stagetwo0_H1_W0_MLP0 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 1 --n_facet 1 --n_facet_context 0 --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 1 --n_facet_window 0 --n_facet_MLP 0 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #Softmax (GPT-2)

# python src/LM/main_GPT2.py --save ./models/nall1_nfacet1_context0_pointer0_reranker0_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 1 --n_facet 1 --n_facet_context 0 --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #Softmax + Mi

# python src/LM/main_GPT2.py --save ./models/nall2_nfacet1_context1_pointer0_reranker0_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 2 --n_facet 1 --n_facet_context 1 --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #Softmax + C + Mi

# python src/LM/main_GPT2.py --save ./models/nall2_nfacet1_context0_pointer0_reranker1(20)_stagetwo0_H1_W0_MLP0 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 2 --n_facet 1 --n_facet_context 0 --n_facet_reranker 1 --n_facet_stage2 0 --n_facet_hidden 1 --n_facet_window 0 --n_facet_MLP 0 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #Softmax + R:20

# python src/LM/main_GPT2.py --save ./models/nall2_nfacet1_context0_pointer0_reranker1(100_20)_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 3 --n_facet 1 --n_facet_context 0 --n_facet_reranker 1 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #Softmax + R:20,100 + Mi

# python src/LM/main_GPT2.py --save ./models/nall3_nfacet1_context0_pointer0_reranker0_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 3 --n_facet 1 --n_facet_context 0 --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode add_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #Softmax + P + Mi

# python src/LM/main_GPT2.py --save ./models/nall3_nfacet1_context1_pointer0_reranker1(100_20)_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 4 --n_facet 1 --n_facet_context 1 --n_facet_reranker 1 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #Softmax + CR:20,100 + Mi

# python src/LM/main_GPT2.py --save ./models/nall5_nfacet1_context1_pointer0_reranker1(100_20)_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 6 --n_facet 1 --n_facet_context 1 --n_facet_reranker 1 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #Softmax + CPR:20,100 + Mi

# python src/LM/main_GPT2.py --save ./models/nall7_nfacet3_context1_pointer0_reranker1(100_20)_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 8 --n_facet 3 --n_facet_context 1 --n_facet_reranker 1 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #MoS + CPR:20,100 + Mi

# python src/LM/main_GPT2.py --save ./models/nall3_nfacet1_context0_Pointer_Sentinel2_reranker0_stagetwo0_H1_W0_MLP0 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 3 --n_facet 1 --n_facet_context 0 --pointer_mode Pointer_Sentinel --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 1 --n_facet_window 0 --n_facet_MLP 0 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #PS

# python src/LM/main_GPT2.py --save ./models/nall3_nfacet1_context0_Pointer_Sentinel2_reranker0_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 3 --n_facet 1 --n_facet_context 0 --pointer_mode Pointer_Sentinel --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #PS+Mi

# python src/LM/main_GPT2.py --save ./models/nall3_nfacet1_context0_Pointer2_reranker0_stagetwo0_H1_W0_MLP0 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 3 --n_facet 1 --n_facet_context 0 --pointer_mode Pointer --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 1 --n_facet_window 0 --n_facet_MLP 0 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #PG

# python src/LM/main_GPT2.py --save ./models/nall3_nfacet1_context0_Pointer2_reranker0_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 3 --n_facet 1 --n_facet_context 0 --pointer_mode Pointer --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #PG+Mi

# python src/LM/main_GPT2.py --save ./models/nal3_nfacet3_context0_pointer0_reranker0_stagetwo0_H1_W0_MLP0 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 3 --n_facet 3 --n_facet_context 0 --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 1 --n_facet_window 0 --n_facet_MLP 0 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #MoS

# python src/LM/main_GPT2.py --save ./models/nal3_nfacet3_context0_pointer0_reranker0_stagetwo0_H3_W-2_MLP-1 --data ./data/processed/openwebtext17-18_gpt2 --seed 11 --n_facet_all 3 --n_facet 3 --n_facet_context 0 --n_facet_reranker 0 --n_facet_stage2 0 --n_facet_hidden 3 --n_facet_window -2 --n_facet_MLP -1 --context_efficient_mode assign_average_of_duplicates --reranker_efficient_mode assign_reranker_logits --reranker_stage2_efficient_mode add_reranker_stage2_logits --training_split_num 20 --valid_per_epoch 20 --log-interval 200 --bptt 200 --batch_size 4 --eval_batch_size 4 --optimizer AdamW --model_name gpt2 --lr 1e-5 --stage2_CAN_NUM 100 --reranker_CAN_NUM 100 20 --candidates_from_previous_reranker True --reranker_pred direction --stage2_Block_size 20 --use_MoS True --epochs 1 --load_file_name LM_weights.pt #MoS + Mi