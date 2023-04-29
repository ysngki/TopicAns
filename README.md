# TopicAns
Utilize TopicModel to faciliate Answer recommendation on Technical QA sites

## Requirement
``` Transformers, Torch, nltk ``` is requried.

## Dataset Construction and Preprocess

We will upload our datasets soon. Following is how we get them:

1. Download ```stackoverflow.com-Posts.7z``` from https://archive.org/details/stackexchange

2. Run ```dataset_scripts/split_qa.py``` to get 
    1. q_temp_output.csv: ["Id", "PostTypeId", "Score", "AnswerCount", "AcceptedAnswerId", "Title", "Body", "Tags"]
    2. a_temp_output.csv: ["Id", "PostTypeId", "Score", "ParentId", "Body"]
    3. python_question_info

3. Run ```dataset_scripts/extract_python_qa.ipynb``` to get question qa (This file can also be used to get java qa pairs!). This file does:
    1. Get python_question_info: (q_id, 1, accepted_a_id, other_a_ids...)
    2. Get python_q_id_content.csv, python_a_id_content.csv
    3. Then process the above two files
    4. delete some trash questions from python_question_info, and save the new one as python_qa_info.dataset. Please read this file by ```datasets``` (huggingface).
4. Then we need to construct candidate answer pools following Gao et al. (This is troublesome and we have lost some code files, sorry) by ```construct_candidate_pools.py```.

5. Finally, we combine all together and get our datasets by ```combine_py_related_question.ipynb```




## Train our model:
```python -u main.py -d so_python --model_class QATopicMemoryModel  -n 0 --model_save_prefix top10_topic25_pooler_ --train_batch_size 16 --gradient_accumulation_steps 4 --two_stage --composition pooler --text_max_len 512 --pretrained_bert_path huggingface/CodeBERTa-small-v1 --latent_dim 25 --no_initial_test```

The best model is saved at ```./model/``` and the lastest model is saved at ```./last_model/```. If training is broken, you can add ```---restore``` at the last without other changing. We have supported logging, so you can choose append ```| tee this_is_log.log``` to the command.

## Measure time
```python -u main.py -d so_python --model_class QATopicMemoryModel  -n 0 --model_save_prefix top10_topic25_pooler_ --train_batch_size 16 --gradient_accumulation_steps 4 --two_stage --composition pooler --text_max_len 512 --pretrained_bert_path huggingface/CodeBERTa-small-v1 --latent_dim 25 --no_initial_test --measure_time --no_train```

## todo....
## command

```bash
# vanilla bert
python -u main.py -d so_python -m 50 --model_class BasicModel  -n 1 --model_save_prefix pooler_ --train_batch_size 16 --gradient_accumulation_steps 4 --one_stage --composition pooler  | tee -a logs/python/one_stage_pooler_basic.log

# Simultaneous pre-training and training
python -u main.py -d ask_ubuntu -m 50 --model_class InputMemorySelfAtt -n 0 --mlm --model_class InputMemorySelfAtt --model_save_prefix mlm_m50_ask_ubuntu_ --train_batch_size 16 --gradient_accumulation_steps 4  --load_memory --memory_save_prefix layer4_ | tee logs/ask_ubuntu/mlm_m50_input_memory.log

# Retrieve the pre-trained memory for training
python -u main.py -d ask_ubuntu -m 50 --model_class InputMemorySelfAtt -n 1 --model_class InputMemorySelfAtt --model_save_prefix fast_mlm_m50_ask_ubuntu_ --train_batch_size 16 --gradient_accumulation_steps 4  --load_memory --memory_save_prefix layer4_ | tee logs/ask_ubuntu/fast_mlm_m50_input_memory.log

# Read the memory, perform the first stage training, and verify whether the pre-trained memory still needs the second stage
python -u main.py -d ask_ubuntu -m 50 --model_class InputMemorySelfAtt -n 1 --model_class InputMemorySelfAtt --model_save_prefix one_stage_mlm_m50_ask_ubuntu_ --train_batch_size 16 --gradient_accumulation_steps 4  --load_memory --memory_save_prefix new_layer4_ | tee logs/ask_ubuntu/one_stage_mlm_m50_input_memory.log

# Input is TBA, the most common training
python -u main.py -d so_python -m 50 --model_class InputMemorySelfAtt -n 1 --model_save_prefix one_stage_m50_ --train_batch_size 16 --gradient_accumulation_steps 4 | tee -a logs/so_python/one_stage_m50_input_memory.log

# Input is TBA, which was interrupted in the last phase, to resume training
python -u main.py -d super_user -m 75 --model_class InputMemorySelfAtt -n 1 --model_save_prefix two_stage_m75_ --train_batch_size 16 --gradient_accumulation_steps 4 --restore --only_final | tee -a logs/super_user/two_stage_m75_input_memory.log

# Input is QA, plus the most common training of memory (pooler)
python -u main.py -d ask_ubuntu -m 50 --model_class QAMemory -n 1 --model_save_prefix m50_ --train_batch_size 16 --gradient_accumulation_steps 4  | tee -a logs/ask_ubuntu/two_stage_m50_qa_input_memory.log

# Input is QA, the most common training without adding memory (pooler)
python -u main.py -d ask_ubuntu --model_class QAModel -n 1 --train_batch_size 16 --gradient_accumulation_steps 4  | tee -a logs/ask_ubuntu/two_stage_qa_input.log
```
