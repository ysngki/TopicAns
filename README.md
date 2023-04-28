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

