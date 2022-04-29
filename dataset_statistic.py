import datasets
import os
from transformers import AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("/data/yuanhang/topic_memory/tokenizer/_data_yuanhang_pretrained_model_prajjwal1_bert-small_SC_EC")

dataset_name = "so_java"
data = datasets.load_from_disk(os.path.join(dataset_name, "string_train.dataset"))


for column_name in ['body', 'answers', 'title']:
    this_column = data[column_name]
    
    whole_len = 0
    column_len = len(this_column)
    max_len = 0

    now_count = 0
    step = 1000
    while now_count < column_len:
        this_part_column = this_column[now_count:(now_count+step)]
        real_step = len(this_part_column)
        now_count += real_step

        encoded_text = tokenizer(this_part_column, padding=True, verbose=False, add_special_tokens=True, return_tensors='pt')
        attention_mask = encoded_text['attention_mask']

        this_part_len = torch.sum(attention_mask, dim=-1)

        max_len = max_len if max_len > torch.max(this_part_len).item() else torch.max(this_part_len).item()
        whole_len += torch.sum(this_part_len).item()

        print(f"\r{now_count}/{column_len}", end="")
    
    print()
    print(f"{column_name}, Max Len {max_len}, Average len {whole_len/column_len}")
    print()

