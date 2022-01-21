from transformers import BertModel, BertForSequenceClassification, BertConfig
from transformers.models.bert.out_vector_modeling_bert import MyBertModel, DecoderLayerChunk
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.nn.functional
import torch.nn.functional as F

from my_function import get_rep_by_avg, dot_attention, clean_input_ids, clean_input_embeddings

# model list
# 1. QAClassifierModel
#       basic bi-encoder for 1-1 classification, support mem
# 2. QAMatchModel
#       basic bi-encoder for 1-n match tasks, support mem
# 3. CrossBERT
#       basic cross-encoder for both classification and matching
# 4. ClassifyParallelEncoder
#       my model for 1-1 classification
# 5. MatchParallelEncoder
#       my model for 1-n match tasks
# 6. PolyEncoder
#       for both classification and matching


# --------------------------------------
# fen ge xian
# --------------------------------------
# No matter whether use memory mechanism, all models whose input consists of Q and A can use this class
class QAClassifierModelConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, composition='pooler'):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.composition = composition

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("composition:", self.composition)


class QAClassifierModel(nn.Module):
    def __init__(self, config):

        super(QAClassifierModel, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

        self.value_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.sentence_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.sentence_embedding_len, config.sentence_embedding_len),
            nn.Tanh()
        )

        self.self_attention_weight_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, 1),
            nn.Sigmoid()
        )

        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_self_att(self, input_ids, token_type_ids, attention_mask):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        last_hidden_state = out['last_hidden_state']

        # 接下来通过attention汇总信息----------------------------------------
        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # (batch, sequence, 1)
        weight = self.self_attention_weight_layer(last_hidden_state)
        weight = weight.squeeze(-1)

        # 创作出score mask
        with torch.no_grad():
            # (batch, sequence)
            mask = token_type_ids.type(dtype=torch.float)
            mask[mask == 1] = -np.inf
            mask[mask == 0] = 0.0
            mask = mask * attention_mask

        mask_weight = mask + weight
        final_weight = nn.functional.softmax(mask_weight, dim=-1)
        # (batch, 1, sequence)
        final_weight = final_weight.unsqueeze(1)

        # 求和
        # (batch, 1, output_embedding_len)
        embedding = final_weight.bmm(value)
        embedding = embedding.squeeze(1)

        final_embedding = self.relu(embedding)

        return final_embedding

    def get_rep_by_pooler(self, input_ids, token_type_ids, attention_mask):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = out['pooler_output']

        return out

    # use the average embedding of last layer as sentence representation
    def get_rep_by_avg(self, input_ids, token_type_ids, attention_mask):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        # get the average embeddings of last layer, excluding memory and special token(?)
        # (batch_size, sequence_length, hidden_size)
        last_hidden_state = out['last_hidden_state']

        representations = get_rep_by_avg(embeddings=last_hidden_state, attention_mask=attention_mask, token_type_ids=token_type_ids)
        representations = representations.squeeze(-2)

        return representations

    # Pre-compute representations. Used for time measuring.
    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        if self.config.composition == 'avg':
            composition_function = self.get_rep_by_avg
        elif self.config.composition == 'pooler':
            composition_function = self.get_rep_by_pooler
        else:
            raise Exception(f"Composition {self.config.composition} is not supported!!")

        candidate_seq_len = input_ids.shape[-1]

        input_ids = input_ids.reshape(-1, candidate_seq_len)
        token_type_ids = token_type_ids.reshape(-1, candidate_seq_len)
        attention_mask = attention_mask.reshape(-1, candidate_seq_len)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)
        candidate_embeddings = composition_function(input_ids=input_ids, token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)
        return candidate_embeddings

    # use pre-compute candidate to get logits
    def do_queries_classify(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        # if self.config.composition == 'avg':
        #     composition_function = self.get_rep_by_avg
        # elif self.config.composition == 'pooler':
        composition_function = self.get_rep_by_pooler
        # else:
        #     raise Exception(f"Composition {self.config.composition} is not supported!!")

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        query_embeddings = composition_function(input_ids=input_ids, token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
        logits = self.classifier(a_embedding=query_embeddings, b_embedding=candidate_context_embeddings)
        return logits

    # for training
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, **kwargs):

        if self.config.composition == 'avg':
            composition_function = self.get_rep_by_avg
        elif self.config.composition == 'pooler':
            composition_function = self.get_rep_by_pooler
        else:
            raise Exception(f"Composition {self.config.composition} is not supported!!")

        a_input_ids, a_attention_mask, a_token_type_ids = clean_input_ids(a_input_ids, a_attention_mask, a_token_type_ids)
        b_input_ids, b_attention_mask, b_token_type_ids = clean_input_ids(b_input_ids, b_attention_mask, b_token_type_ids)

        a_embeddings = composition_function(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                            attention_mask=a_attention_mask)

        b_embeddings = composition_function(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                            attention_mask=b_attention_mask)

        # # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(a_embedding=a_embeddings, b_embedding=b_embeddings)

        return logits


class QAMatchModel(nn.Module):
    def __init__(self, config):

        super(QAMatchModel, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # self.embeddings = self.bert_model.get_input_embeddings()
        #
        # self.value_layer = nn.Sequential(
        #     nn.Linear(config.word_embedding_len, 2 * config.sentence_embedding_len),
        #     nn.ReLU(),
        #     nn.Linear(2 * config.sentence_embedding_len, config.sentence_embedding_len),
        #     nn.Tanh()
        # )
        #
        # self.self_attention_weight_layer = nn.Sequential(
        #     nn.Linear(config.word_embedding_len, 2 * config.word_embedding_len),
        #     nn.ReLU(),
        #     nn.Linear(2 * config.word_embedding_len, 1),
        #     nn.Sigmoid()
        # )
        #
        # self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)
        #
        # self.relu = torch.nn.ReLU(inplace=True)
        # self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_pooler(self, input_ids, token_type_ids, attention_mask):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = out['last_hidden_state'][:, 0, :]

        return out

    # Pre-compute representations. Used for time measuring.
    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        candidate_seq_len = input_ids.shape[-1]

        input_ids = input_ids.reshape(-1, candidate_seq_len)
        token_type_ids = token_type_ids.reshape(-1, candidate_seq_len)
        attention_mask = attention_mask.reshape(-1, candidate_seq_len)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        candidate_embeddings = self.get_rep_by_pooler(input_ids=input_ids, token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask)
        return candidate_embeddings

    # use pre-compute candidate to get scores
    def do_queries_match(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        query_embeddings = self.get_rep_by_pooler(input_ids=input_ids, token_type_ids=token_type_ids,
                                                  attention_mask=attention_mask)
        query_embeddings = query_embeddings.unsqueeze(1)
        dot_product = torch.matmul(query_embeddings, candidate_context_embeddings.permute(0, 2, 1)).squeeze(1)
        return dot_product

    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, train_flag, **kwargs):
        """
        :param train_flag:
        :param a_input_ids: (batch size, sequence len)
        :param a_token_type_ids: (batch size, sequence len)
        :param a_attention_mask: (batch size, sequence len)
        :param b_input_ids: training: (batch size, sequence len) / val: (batch size, candidate num, sequence len)
        :param b_token_type_ids: training: (batch size, sequence len) / val: (batch size, candidate num, sequence len)
        :param b_attention_mask: training: (batch size, sequence len) / val: (batch size, candidate num, sequence len)
        :return: training: (batch size, batch size) / val: (batch size, candidate num)
        """
        if self.config.composition == 'pooler':
            composition_function = self.get_rep_by_pooler
        else:
            raise Exception(f"Composition {self.config.composition} is not supported!!")

        a_input_ids, a_attention_mask, a_token_type_ids = clean_input_ids(a_input_ids, a_attention_mask,
                                                                          a_token_type_ids)

        # encode context
        a_embeddings = composition_function(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                            attention_mask=a_attention_mask)

        # reshape and encode candidates
        candidate_seq_len = b_input_ids.shape[-1]
        b_input_ids = b_input_ids.reshape(-1, candidate_seq_len)
        b_token_type_ids = b_token_type_ids.reshape(-1, candidate_seq_len)
        b_attention_mask = b_attention_mask.reshape(-1, candidate_seq_len)

        b_input_ids, b_attention_mask, b_token_type_ids = clean_input_ids(b_input_ids, b_attention_mask,
                                                                          b_token_type_ids)
        b_embeddings = composition_function(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                            attention_mask=b_attention_mask)

        # for some purposes I don't compute logits in forward while training. Only this model behaves like this.
        if train_flag:
            # dot_product = torch.matmul(a_embeddings, b_embeddings.t())  # [bs, bs]
            # mask = torch.eye(a_embeddings.size(0)).to(a_embeddings.device)
            # loss = F.log_softmax(dot_product, dim=-1) * mask
            # loss = (-loss.sum(dim=1)).mean()
            return a_embeddings, b_embeddings
        else:
            b_embeddings = b_embeddings.reshape(a_embeddings.shape[0], -1, b_embeddings.shape[-1])
            a_embeddings = a_embeddings.unsqueeze(1)
            dot_product = torch.matmul(a_embeddings, b_embeddings.permute(0, 2, 1)).squeeze(1)
            return dot_product


# --------------------------------------
# fen ge xian
# --------------------------------------
class CrossBERTConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, composition='pooler'):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.composition = composition

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("composition:", self.composition)


class CrossBERT(nn.Module):
    def __init__(self, config):

        super(CrossBERT, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)
        this_bert_config.num_labels = self.num_labels

        self.bert_model = BertForSequenceClassification.from_pretrained(config.pretrained_bert_path, config=this_bert_config)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

    def forward(self, input_ids, token_type_ids, attention_mask, **kwargs):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        # # 根据输入，进行思考, 思考的结果要选择性遗忘
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = out['logits']

        return logits


class MatchCrossBERT(nn.Module):
    def __init__(self, config):

        super(MatchCrossBERT, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)
        this_bert_config.num_labels = self.num_labels

        self.bert_model = BertForSequenceClassification.from_pretrained(config.pretrained_bert_path, config=this_bert_config)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        # self.embeddings = self.bert_model.get_input_embeddings()

    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, train_flag, **kwargs):
        # (all_candidate_num, dim)
        candidate_seq_len = b_input_ids.shape[-1]
        b_input_ids = b_input_ids.reshape(-1, candidate_seq_len)
        b_token_type_ids = b_token_type_ids.reshape(-1, candidate_seq_len)
        b_attention_mask = b_attention_mask.reshape(-1, candidate_seq_len)

        # remove padding
        # a_input_ids, a_attention_mask, a_token_type_ids = clean_input_ids(a_input_ids, a_attention_mask, a_token_type_ids)
        # b_input_ids, b_attention_mask, b_token_type_ids = clean_input_ids(b_input_ids, b_attention_mask, b_token_type_ids)
        b_token_type_ids = b_token_type_ids + 1

        # move to cpu
        original_device = b_input_ids.device

        b_input_ids.to("cpu")
        b_token_type_ids.to("cpu")
        b_attention_mask.to("cpu")

        a_input_ids.to("cpu")
        a_token_type_ids.to("cpu")
        a_attention_mask.to("cpu")

        batch_size = a_input_ids.shape[0]
        if train_flag:
            candidate_num = batch_size
        else:
            candidate_num = b_input_ids.shape[0] // batch_size

        # concatenate them
        final_logits = []
        step_num = 4
        if train_flag:
            for this_a_input_ids, this_a_attention_mask, this_a_token_type_ids in zip(a_input_ids, a_attention_mask, a_token_type_ids):
                temp_count = 0
                this_logits = []
                while temp_count < candidate_num:
                    step_b_input_ids = b_input_ids[temp_count:temp_count + step_num]
                    step_b_attention_mask = b_attention_mask[temp_count:temp_count + step_num]
                    step_b_token_type_ids = b_token_type_ids[temp_count:temp_count + step_num]

                    this_step_real_num = step_b_input_ids.shape[0]

                    step_a_input_ids = this_a_input_ids.repeat(this_step_real_num, 1)
                    step_a_attention_mask = this_a_attention_mask.repeat(this_step_real_num, 1)
                    step_a_token_type_ids = this_a_token_type_ids.repeat(this_step_real_num, 1)

                    final_input_ids = torch.cat((step_a_input_ids, step_b_input_ids), dim=-1).to(original_device)
                    final_attention_mask = torch.cat((step_a_attention_mask, step_b_attention_mask), dim=-1).to(original_device)
                    final_token_type_ids = torch.cat((step_a_token_type_ids, step_b_token_type_ids), dim=-1).to(original_device)

                    temp_logits = self.bert_model(input_ids=final_input_ids, token_type_ids=final_token_type_ids, attention_mask=final_attention_mask)['logits'].squeeze(-1)

                    this_logits.append(temp_logits)
                    temp_count += step_num
                    torch.cuda.empty_cache()

                this_logits = torch.cat(this_logits, dim=0)
                final_logits.append(this_logits.unsqueeze(0))
        else:
            new_candidate_seq_len = b_input_ids.shape[-1]
            b_input_ids = b_input_ids.reshape(batch_size, candidate_num, new_candidate_seq_len)
            b_token_type_ids = b_token_type_ids.reshape(batch_size, candidate_num, new_candidate_seq_len)
            b_attention_mask = b_attention_mask.reshape(batch_size, candidate_num, new_candidate_seq_len)

            for index, (this_a_input_ids, this_a_attention_mask, this_a_token_type_ids) in enumerate(
                    zip(a_input_ids, a_attention_mask, a_token_type_ids)):
                this_a_input_ids = this_a_input_ids.repeat(candidate_num, 1)
                this_a_attention_mask = this_a_attention_mask.repeat(candidate_num, 1)
                this_a_token_type_ids = this_a_token_type_ids.repeat(candidate_num, 1)

                this_b_input_ids = b_input_ids[index]
                this_b_attention_mask = b_attention_mask[index]
                this_b_token_type_ids = b_token_type_ids[index]

                final_input_ids = torch.cat((this_a_input_ids, this_b_input_ids), dim=-1).to(original_device)
                final_attention_mask = torch.cat((this_a_attention_mask, this_b_attention_mask), dim=-1).to(original_device)
                final_token_type_ids = torch.cat((this_a_token_type_ids, this_b_token_type_ids), dim=-1).to(original_device)

                this_logits = self.bert_model(input_ids=final_input_ids, token_type_ids=final_token_type_ids,
                                              attention_mask=final_attention_mask)['logits'].squeeze(-1).unsqueeze(0)
                final_logits.append(this_logits)

        dot_product = torch.cat(final_logits, dim=0)

        if train_flag:
            mask = torch.eye(batch_size).to(dot_product.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            return dot_product


# --------------------------------------
# fen ge xian
# --------------------------------------
class ParallelEncoderConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, composition='pooler', context_num=1):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.composition = composition
        self.context_num = context_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("composition:", self.composition)
        print("context_num:", self.context_num)


# used for 1-1 classification task
class ClassifyParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(ClassifyParallelEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # self.embeddings = self.bert_model.get_input_embeddings()

        # used to compressing candidate to generate context vectors
        self.composition_layer = OutVectorSelfAttentionLayer(self.this_bert_config.hidden_size, config.context_num)

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': MeanLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        # create classifier
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.init_decoder_params()

    # use parameters of pre-trained encoder to initialize decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    # Pre-compute representations. Used for time measuring.
    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        # reshape and encode candidates
        # (all_candidate_num, dim)
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=False)
        last_hidden_state = out['last_hidden_state']

        # get (all_candidate_num, context_num, dim)
        candidate_embeddings = self.composition_layer(last_hidden_state, attention_mask=attention_mask)

        return candidate_embeddings

    # use pre-compute candidate to get logits
    def do_queries_classify(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        candidate_context_embeddings = candidate_context_embeddings.reshape(input_ids.shape[0], self.config.context_num,
                                                                            candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(candidate_context_embeddings.shape[0], 1, candidate_context_embeddings.shape[-1],
                                         device=candidate_context_embeddings.device)

        # (query_num, candidate_context_num + candidate_num, dim)
        candidate_context_embeddings = torch.cat((candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=True, candidate_embeddings=candidate_context_embeddings,
                                decoder=self.decoder, candidate_num=1)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        b_context_embeddings = decoder_output[:, :-1, :].reshape(decoder_output.shape[0], 1,
                                                                 self.config.context_num,
                                                                 decoder_output.shape[-1])
        # (query_num, candidate_num, dim)
        b_embeddings = self.decoder['candidate_composition_layer'](b_context_embeddings).squeeze(-2)

        a_embeddings = decoder_output[:, -1:, :]

        logits = self.classifier(a_embedding=a_embeddings, b_embedding=b_embeddings).squeeze(1)

        return logits

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, **kwargs):
        do_ablation = kwargs.get('do_ablation')

        b_input_ids, b_attention_mask, b_token_type_ids = clean_input_ids(b_input_ids, b_attention_mask,
                                                                          b_token_type_ids)

        # encoding candidate texts
        # (text_b_num, sequence len, dim)
        b_out = self.bert_model(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask,
                                enrich_candidate_by_question=False)
        b_last_hidden_state = b_out['last_hidden_state']

        # get (text_b_num, context_num, dim)
        b_embeddings = self.composition_layer(b_last_hidden_state, attention_mask=b_attention_mask)

        # convert to (query_num, context_num, dim)
        # perhaps this line could be removed
        b_embeddings = b_embeddings.reshape(a_input_ids.shape[0], self.config.context_num, b_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(a_input_ids.shape[0], 1, b_embeddings.shape[-1], device=b_embeddings.device)
        # (query_num, candidate_context_num + candidate_num, dim)
        b_embeddings = torch.cat((b_embeddings, lstm_initial_state), dim=1)

        a_input_ids, a_attention_mask, a_token_type_ids = clean_input_ids(a_input_ids, a_attention_mask,
                                                                          a_token_type_ids)
        # get q and new a
        a_out = self.bert_model(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask,
                                enrich_candidate_by_question=True, candidate_embeddings=b_embeddings,
                                decoder=self.decoder, candidate_num=1)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        # get final representations
        # (query_num, 1, context_num, dim)
        b_context_embeddings = decoder_output[:, :-1, :].reshape(decoder_output.shape[0], 1, self.config.context_num, decoder_output.shape[-1])
        # (query_num, 1, dim)
        b_embeddings = self.decoder['candidate_composition_layer'](b_context_embeddings).squeeze(-2)

        if do_ablation:
            # (query_num, 1, dim)
            a_embeddings = self.decoder['candidate_composition_layer'](a_last_hidden_state, attention_mask=a_attention_mask)
        else:
            # (query_num, candidate_num, dim)
            a_embeddings = decoder_output[:, -1:, :] + self.decoder['candidate_composition_layer'](a_last_hidden_state, attention_mask=a_attention_mask)
            a_embeddings = a_embeddings / 2

        logits = self.classifier(a_embedding=a_embeddings, b_embedding=b_embeddings)
        logits = logits.squeeze(1)

        return logits


# used for 1-n match task
class MatchParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(MatchParallelEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # used to compressing candidate to generate context vectors
        self.composition_layer = OutVectorSelfAttentionLayer(self.this_bert_config.hidden_size, config.context_num)

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': MeanLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        self.init_decoder_params()

    # use parameters of pre-trained encoder to initialize decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Used for time measuring. Return shape: (-1, context_num, context_dim) """

        # reshape and encode candidates, (all_candidate_num, dim)
        candidate_seq_len = input_ids.shape[-1]
        input_ids = input_ids.reshape(-1, candidate_seq_len)
        token_type_ids = token_type_ids.reshape(-1, candidate_seq_len)
        attention_mask = attention_mask.reshape(-1, candidate_seq_len)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=False)
        last_hidden_state = out['last_hidden_state']

        # get (all_candidate_num, context_num, dim)
        candidate_embeddings = self.composition_layer(last_hidden_state, attention_mask=attention_mask)

        return candidate_embeddings

    def do_queries_match(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        """ use pre-compute candidate to get scores. Only used by real test."""

        do_ablation = kwargs.get('do_ablation', False)
        
        candidate_num = candidate_context_embeddings.shape[1]
        this_candidate_context_embeddings = candidate_context_embeddings.reshape(candidate_context_embeddings.shape[0],
                                                                                 -1,
                                                                                 candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(this_candidate_context_embeddings.shape[0], candidate_num, this_candidate_context_embeddings.shape[-1],
                                         device=this_candidate_context_embeddings.device)

        # (query_num, candidate_context_num + candidate_num, dim)
        this_candidate_context_embeddings = torch.cat((this_candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=True, candidate_embeddings=this_candidate_context_embeddings,
                                decoder=self.decoder, candidate_num=candidate_num)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        # get final representations
        # (query_num, candidate_num, context_num, dim)
        this_candidate_context_embeddings = decoder_output[:, :-candidate_num, :].reshape(decoder_output.shape[0],
                                                                                          candidate_num,
                                                                                          self.config.context_num,
                                                                                          decoder_output.shape[-1])
        # (query_num, candidate_num, dim)
        candidate_embeddings = self.decoder['candidate_composition_layer'](this_candidate_context_embeddings).squeeze(-2)

        if do_ablation:
            # (query_num, 1, dim)
            query_embeddings = self.decoder['candidate_composition_layer'](a_last_hidden_state, attention_mask=attention_mask)
            dot_product = torch.matmul(query_embeddings, candidate_embeddings.permute(0, 2, 1)).squeeze(-2)
        else:
            # (query_num, candidate_num, dim)
            query_embeddings = decoder_output[:, -candidate_num:, :] + self.decoder['candidate_composition_layer'](a_last_hidden_state, attention_mask=attention_mask)
            query_embeddings = query_embeddings / 2
            # (query_num, candidate_num)
            dot_product = torch.mul(query_embeddings, candidate_embeddings).sum(-1)

        return dot_product

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, train_flag, **kwargs):
        """
        if train_flag==True, a_input_ids.shape and b_input_ids.shape are 2 dims, like (batch size, seq_len),
        otherwise, b_input_ids should look like (batch size, candidate num, seq len).
        """

        return_dot_product = kwargs.get('return_dot_product', False)
        do_ablation = kwargs.get('do_ablation')

        # (all_candidate_num, dim)
        b_embeddings = self.prepare_candidates(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask)

        # convert to (query_num, candidate_context_num, dim)
        if not train_flag:
            b_embeddings = b_embeddings.reshape(a_input_ids.shape[0], -1, self.config.context_num, b_embeddings.shape[-1])
        else:
            # need broadcast
            b_embeddings = b_embeddings.reshape(-1, self.config.context_num, b_embeddings.shape[-1]).unsqueeze(0).\
                expand(a_input_ids.shape[0], -1, -1, -1)

        dot_product = self.do_queries_match(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                            attention_mask=a_attention_mask, candidate_context_embeddings=b_embeddings,
                                            do_ablation=do_ablation)

        if train_flag:
            if return_dot_product:
                return dot_product
            mask = torch.eye(a_input_ids.size(0)).to(a_input_ids.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            return dot_product


# --------------------------------------
# fen ge xian
# --------------------------------------
# A gate, a substitution of resnet
class MyLSTMBlock(nn.Module):
    def __init__(self, input_dim):
        super(MyLSTMBlock, self).__init__()
        self.weight_layer = nn.Linear(input_dim*3, input_dim)

        self.sigmoid = nn.Sigmoid()

    # shapes are supposed to be (batch size, input_dim)
    def forward(self, this_compressed_vector, last_compressed_vector, weight_hint):
        weight = self.weight_layer(torch.cat((weight_hint, this_compressed_vector, last_compressed_vector), dim=-1))
        weight = self.sigmoid(weight)

        new_compressed_vector = weight * this_compressed_vector + (1-weight)*last_compressed_vector

        return new_compressed_vector


# --------------------------------------
# Composition Layers
# 1. LinearSelfAttentionLayer: Multi-layer self-attention
# 2. OutVectorSelfAttentionLayer: maintain several queries to get several representations by attention mechanism
# 3. MeanLayer: do average to get representations
# --------------------------------------
# according to Multihop Attention Networks for Question Answer Matching
class LinearSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, drop_prob=0.2):
        super(LinearSelfAttentionLayer, self).__init__()
        self.conversion_layer = nn.Linear(input_dim, input_dim*2)
        self.weight_layer = nn.Linear(input_dim*2, input_dim)

        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.tanh = torch.nn.Tanh()
        # here is different from that paper
        self.softmax = nn.Softmax(dim=-2)

    # context shape is supposed to be (batch size, sequence len, input_dim)
    # attention_mask shape is (batch size, sequence len)
    # output is (batch size, input_dim)
    def forward(self, context, attention_mask):
        # (batch size, sequence dim, input_dim)
        converted_context = self.tanh(self.conversion_layer(context))
        raw_attention_weight = self.weight_layer(converted_context)

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float).clone().detach()
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(context.shape[-1], 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        raw_attention_weight = raw_attention_weight + mask
        attention_weight = self.softmax(raw_attention_weight)

        processed_context = attention_weight.mul(context)

        # sum by dim -1
        # (batch size, input_dim)
        context_representation = processed_context.sum(dim=-2)
        return context_representation


class OutVectorSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, context_num):
        super(OutVectorSelfAttentionLayer, self).__init__()
        self.query = nn.Parameter(torch.randn(context_num, input_dim))

    def forward(self, context, attention_mask=None):
        """
        :param context: (..., sequence len, input_dim)
        :param attention_mask: (..., sequence len)
        :return: (..., context_num, input_dim)
        """

        context_representation = dot_attention(q=self.query, k=context, v=context, v_mask=attention_mask)

        return context_representation


class MeanLayer(nn.Module):
    def __init__(self):
        super(MeanLayer, self).__init__()

    @staticmethod
    def forward(embeddings, token_type_ids=None, attention_mask=None):
        """
        do average at dim -2 without remove this dimension
        :param embeddings: (..., sequence len, dim)
        :param token_type_ids: optional
        :param attention_mask: optional
        :return: (..., 1, dim)
        """

        representation = get_rep_by_avg(embeddings, token_type_ids, attention_mask)

        return representation


# --------------------------------------
# fen ge xian
# --------------------------------------
class PolyEncoderConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, context_num=1):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.context_num = context_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("context_num:", self.context_num)


class PolyEncoder(nn.Module):
    def __init__(self, config: PolyEncoderConfig):

        super(PolyEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        self.query_composition_layer = OutVectorSelfAttentionLayer(self.this_bert_config.hidden_size, self.config.context_num)

        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        torch.nn.init.normal_(self.query_composition_layer.query, self.this_bert_config.hidden_size ** -0.5)

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        # encoding candidate texts
        # (batch_size, sequence len, dim)
        candidate_seq_len = input_ids.shape[-1]

        input_ids = input_ids.reshape(-1, candidate_seq_len)
        token_type_ids = token_type_ids.reshape(-1, candidate_seq_len)
        attention_mask = attention_mask.reshape(-1, candidate_seq_len)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        candidate_embeddings = out['last_hidden_state']
        # (batch_size, 1, dim) ---- pooler
        candidate_embeddings = candidate_embeddings[:, 0, :]

        return candidate_embeddings

    def do_queries_match(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        query_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        query_last_hidden_state = query_out['last_hidden_state']
        # (query_num, context_num, dim)
        query_embeddings = self.query_composition_layer(query_last_hidden_state, attention_mask)

        # candidate_context_embeddings = (query_num, candidate_num, dim)
        # final_query_context_vec = (query_num, candidate_num, dim)
        final_query_context_vec = dot_attention(q=candidate_context_embeddings, k=query_embeddings,
                                                v=query_embeddings)

        dot_product = torch.sum(final_query_context_vec * candidate_context_embeddings, -1)
        return dot_product

    def do_queries_classify(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        query_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        query_last_hidden_state = query_out['last_hidden_state']
        # (query_num, context_num, dim)
        query_embeddings = self.query_composition_layer(query_last_hidden_state, attention_mask)
        # (query_num, 1, dim)
        candidate_context_embeddings = candidate_context_embeddings.unsqueeze(-2)

        # candidate_context_embeddings = (query_num, 1, dim)
        # final_query_context_vec = (query_num, 1, dim)
        final_query_context_vec = dot_attention(q=candidate_context_embeddings, k=query_embeddings,
                                                v=query_embeddings)

        final_query_context_vec = final_query_context_vec.squeeze(-2)
        candidate_context_embeddings = candidate_context_embeddings.squeeze(-2)

        logits = self.classifier(a_embedding=final_query_context_vec, b_embedding=candidate_context_embeddings)

        return logits

    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, **kwargs):
        batch_size = a_token_type_ids.shape[0]
        a_input_ids, a_attention_mask, a_token_type_ids = clean_input_ids(a_input_ids, a_attention_mask,
                                                                          a_token_type_ids)
        # encoding candidate texts
        # (batch_size, sequence len, dim)
        candidate_seq_len = b_input_ids.shape[-1]

        b_input_ids = b_input_ids.reshape(-1, candidate_seq_len)
        b_token_type_ids = b_token_type_ids.reshape(-1, candidate_seq_len)
        b_attention_mask = b_attention_mask.reshape(-1, candidate_seq_len)

        b_input_ids, b_attention_mask, b_token_type_ids = clean_input_ids(b_input_ids, b_attention_mask,
                                                                          b_token_type_ids)

        b_out = self.bert_model(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask)
        b_last_hidden_state = b_out['last_hidden_state']
        # (batch_size, 1, dim) ---- pooler
        candidate_context_vectors = b_last_hidden_state[:, 0, :].unsqueeze(-2)

        # encoding query texts
        a_out = self.bert_model(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask)
        a_last_hidden_state = a_out['last_hidden_state']
        # (batch_size, context_num, dim)
        query_context_vectors = self.query_composition_layer(a_last_hidden_state, a_attention_mask)

        # one-one classification task
        if self.config.num_labels > 0:
            pass
        else:
            train_flag = kwargs['train_flag']
            if train_flag:
                # (batch_size, dim)
                candidate_context_vectors = candidate_context_vectors.squeeze(-2)
            else:
                # (batch_size, candidate_num, dim)
                candidate_context_vectors = candidate_context_vectors.squeeze(-2).reshape(batch_size, -1, candidate_context_vectors.shape[-1])

        # (batch_size, 1, dim) or (batch_size, batch_size, dim) or (batch_size, candidate_num, dim)
        final_query_context_vec = dot_attention(q=candidate_context_vectors, k=query_context_vectors,
                                                v=query_context_vectors)

        # one-one classification task
        if self.config.num_labels > 0:
            final_query_context_vec = final_query_context_vec.squeeze(-2)
            candidate_context_vectors = candidate_context_vectors.squeeze(-2)

            logits = self.classifier(a_embedding=final_query_context_vec, b_embedding=candidate_context_vectors)

            return logits
        # cos full-match task
        else:
            # (batch size, batch size)
            # first is query, second is candidate
            dot_product = torch.sum(final_query_context_vec * candidate_context_vectors, -1)
            train_flag = kwargs['train_flag']
            if train_flag:
                mask = torch.eye(batch_size).to(dot_product.device)
                loss = F.log_softmax(dot_product, dim=-1) * mask
                loss = (-loss.sum(dim=1)).mean()

                return loss
            else:
                return dot_product


# --------------------------------------
# fen ge xian
# --------------------------------------
# 一个分类器，2个input len的向量作为输入
class QAClassifier(nn.Module):
    def __init__(self, input_len, keep_prob=0.9, num_labels=4):
        super(QAClassifier, self).__init__()

        self.linear1 = torch.nn.Linear(input_len * 3, input_len * 3, bias=True)
        self.linear2 = torch.nn.Linear(input_len * 3, num_labels, bias=True)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=1 - keep_prob)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, a_embedding, b_embedding):
        x = torch.cat((a_embedding, b_embedding, torch.abs(a_embedding-b_embedding)), dim=-1)
        # x = torch.cat((q_embedding, a_embedding), dim=-1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)

        return x


# --------------------------------------
# fen ge xian
# --------------------------------------
class DeformerConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, top_layer_num=2, text_max_len=512):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.top_layer_num = top_layer_num
        self.text_max_len = text_max_len

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("top_layer_numL:", self.top_layer_num)
        print("text_max_len:", self.text_max_len)


class ClassifyDeformer(nn.Module):
    def __init__(self, config: DeformerConfig):

        super(ClassifyDeformer, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)
        this_bert_config.num_labels = self.num_labels

        self.bert_model = BertForSequenceClassification.from_pretrained(config.pretrained_bert_path, config=this_bert_config)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        whole_encoder = self.bert_model.bert.encoder

        self.embeddings = self.bert_model.bert.embeddings
        self.lower_encoder = whole_encoder.layer[:this_bert_config.num_hidden_layers - self.config.top_layer_num]
        self.upper_encoder = whole_encoder.layer[this_bert_config.num_hidden_layers - self.config.top_layer_num:]
        # take cls in and out a new vector
        self.pooler_layer = self.bert_model.bert.pooler
        # take a vector in and out a logits
        self.classifier = self.bert_model.classifier

    # encode a text using lower layers
    def lower_encoding(self, input_ids, attention_mask, token_type_id, **kwargs):
        clean_flag = kwargs.get('clean_flag', True)
        if clean_flag:
            input_ids, attention_mask, token_type_id = clean_input_ids(input_ids, attention_mask, token_type_id)

        embeddings = self.embeddings(input_ids=input_ids,
                                     token_type_ids=token_type_id)
        hidden_states = embeddings

        # process attention mask
        input_shape = input_ids.size()
        device = input_ids.device
        extended_attention_mask = self.bert_model.get_extended_attention_mask(attention_mask, input_shape, device)

        for layer_module in self.lower_encoder:
            layer_outputs = layer_module(
                hidden_states,
                extended_attention_mask
            )
            hidden_states = layer_outputs[0]
        return hidden_states

    # concatenate embeddings of two texts and encoding them using top layers
    # dims of a_embeddings, b_embeddings are both 3, (batch size, seq len, dim)
    # dims of a_attention_mask, b_attention_mask are both 2, (batch size, seq len)
    def joint_encoding(self, a_embeddings, b_embeddings, a_attention_mask, b_attention_mask):
        device = a_embeddings.device

        # simplest case
        final_attention_mask = torch.cat((a_attention_mask, b_attention_mask), dim=-1)
        final_hidden_states = torch.cat((a_embeddings, b_embeddings), dim=-2)

        # process attention mask
        input_shape = final_attention_mask.size()
        final_attention_mask = self.bert_model.get_extended_attention_mask(final_attention_mask, input_shape, device)

        for layer_module in self.upper_encoder:
            layer_outputs = layer_module(
                final_hidden_states,
                final_attention_mask
            )
            final_hidden_states = layer_outputs[0]
        return final_hidden_states

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Used for time measuring. Return shape: (-1, dim) """
        token_type_ids = token_type_ids + 1

        # I don't want to change attention mask and other things, because it's so troublesome
        lower_encoded_embeddings = self.lower_encoding(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       token_type_id=token_type_ids,
                                                       clean_flag=False)
        return lower_encoded_embeddings

    def do_queries_classify(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings,
                            candidate_attention_mask, **kwargs):
        """ use pre-compute candidate to get scores. Only used by real test."""
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)
        candidate_attention_mask, candidate_context_embeddings = clean_input_embeddings(candidate_attention_mask, candidate_context_embeddings)

        # encoding a
        a_lower_encoded_embeddings = self.lower_encoding(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_id=token_type_ids)

        # encoding together
        joint_embeddings = self.joint_encoding(a_embeddings=a_lower_encoded_embeddings,
                                               b_embeddings=candidate_context_embeddings,
                                               a_attention_mask=attention_mask,
                                               b_attention_mask=candidate_attention_mask)

        pooler = self.pooler_layer(joint_embeddings)
        logits = self.classifier(pooler)

        return logits

    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, **kwargs):

        a_input_ids, a_attention_mask, a_token_type_ids = clean_input_ids(a_input_ids, a_attention_mask, a_token_type_ids)
        b_input_ids, b_attention_mask, b_token_type_ids = clean_input_ids(b_input_ids, b_attention_mask, b_token_type_ids)

        # encoding a
        a_lower_encoded_embeddings = self.lower_encoding(input_ids=a_input_ids,
                                                         attention_mask=a_attention_mask,
                                                         token_type_id=a_token_type_ids)

        # encoding b
        b_token_type_ids = b_token_type_ids + 1

        b_lower_encoded_embeddings = self.lower_encoding(input_ids=b_input_ids,
                                                         attention_mask=b_attention_mask,
                                                         token_type_id=b_token_type_ids)

        # encoding together
        joint_embeddings = self.joint_encoding(a_embeddings=a_lower_encoded_embeddings,
                                               b_embeddings=b_lower_encoded_embeddings,
                                               a_attention_mask=a_attention_mask,
                                               b_attention_mask=b_attention_mask)

        pooler = self.pooler_layer(joint_embeddings)
        logits = self.classifier(pooler)

        return logits


class MatchDeformer(nn.Module):
    def __init__(self, config: DeformerConfig):

        super(MatchDeformer, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)
        this_bert_config.num_labels = self.num_labels

        self.bert_model = BertForSequenceClassification.from_pretrained(config.pretrained_bert_path,
                                                                        config=this_bert_config)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        whole_encoder = self.bert_model.bert.encoder

        self.embeddings = self.bert_model.bert.embeddings
        self.lower_encoder = whole_encoder.layer[:this_bert_config.num_hidden_layers - self.config.top_layer_num]
        self.upper_encoder = whole_encoder.layer[this_bert_config.num_hidden_layers - self.config.top_layer_num:]
        # take cls in and out a new vector
        self.pooler_layer = self.bert_model.bert.pooler
        # take a vector in and out a logits
        self.classifier = self.bert_model.classifier

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Used for time measuring. Return shape: (-1, dim) """
        token_type_ids = token_type_ids + 1

        lower_encoded_embeddings = self.lower_encoding(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids,
                                                       clean_flag=False)
        return lower_encoded_embeddings

    def do_queries_match(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings,
                         candidate_attention_mask, **kwargs):
        """ use pre-compute candidate to get scores. Only used by real test."""

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        # encoding a
        a_lower_encoded_embeddings = self.lower_encoding(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids)

        # process candidate
        candidate_seq_len = candidate_attention_mask.shape[-1]
        candidate_attention_mask = candidate_attention_mask.reshape(-1, candidate_seq_len)
        candidate_context_embeddings = candidate_context_embeddings.reshape(candidate_attention_mask.shape[0],
                                                                            candidate_seq_len,
                                                                            candidate_context_embeddings.shape[-1])

        joint_embeddings = self.joint_encoding(a_embeddings=a_lower_encoded_embeddings,
                                               b_embeddings=candidate_context_embeddings,
                                               a_attention_mask=attention_mask,
                                               b_attention_mask=candidate_attention_mask,
                                               train_flag=False)

        pooler = self.pooler_layer(joint_embeddings)
        logits = self.classifier(pooler)
        logits = logits.reshape(input_ids.shape[0], -1)
        return logits

    # encode a text using lower layers
    def lower_encoding(self, input_ids, attention_mask, token_type_ids, **kwargs):
        clean_flag = kwargs.get('clean_flag', True)
        if clean_flag:
            input_ids, attention_mask, token_type_id = clean_input_ids(input_ids, attention_mask, token_type_ids)

        embeddings = self.embeddings(input_ids=input_ids,
                                     token_type_ids=token_type_ids)
        hidden_states = embeddings

        # process attention mask
        input_shape = input_ids.size()
        device = input_ids.device
        extended_attention_mask = self.bert_model.get_extended_attention_mask(attention_mask, input_shape, device)

        for layer_module in self.lower_encoder:
            layer_outputs = layer_module(
                hidden_states,
                extended_attention_mask
            )
            hidden_states = layer_outputs[0]
        return hidden_states

    # concatenate embeddings of two texts and encoding them using top layers
    # dims of a_embeddings, b_embeddings are both 3, (batch size, seq len, dim)
    # dims of a_attention_mask, b_attention_mask are both 2, (batch size, seq len)
    def joint_encoding(self, a_embeddings, b_embeddings, a_attention_mask, b_attention_mask, train_flag):
        device = a_embeddings.device

        a_max_seq_len = torch.max(a_attention_mask.sum(-1))

        if train_flag:
            candidate_num = a_embeddings.shape[0]
        else:
            candidate_num = int(b_embeddings.shape[0] / a_embeddings.shape[0])

        # repeat A
        a_embeddings = a_embeddings.unsqueeze(1)
        a_embeddings = a_embeddings.repeat(1, candidate_num, 1, 1).reshape(-1, a_max_seq_len,
                                                                           a_embeddings.shape[-1])
        a_attention_mask = a_attention_mask.unsqueeze(1)
        a_attention_mask = a_attention_mask.repeat(1, candidate_num, 1).reshape(-1, a_max_seq_len)

        # need repeat candidate
        if train_flag:
            b_embeddings = b_embeddings.repeat(candidate_num, 1, 1)
            b_attention_mask = b_attention_mask.repeat(candidate_num, 1)

        final_attention_mask = torch.cat((a_attention_mask, b_attention_mask), dim=-1)
        final_hidden_states = torch.cat((a_embeddings, b_embeddings), dim=1)

        # process attention mask
        input_shape = final_attention_mask.size()
        final_attention_mask = self.bert_model.get_extended_attention_mask(final_attention_mask, input_shape, device)

        for layer_module in self.upper_encoder:
            layer_outputs = layer_module(
                final_hidden_states,
                final_attention_mask
            )
            final_hidden_states = layer_outputs[0]
        return final_hidden_states

    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, train_flag, **kwargs):

        a_input_ids, a_attention_mask, a_token_type_ids = clean_input_ids(a_input_ids, a_attention_mask,
                                                                          a_token_type_ids)

        # encoding a
        a_lower_encoded_embeddings = self.lower_encoding(input_ids=a_input_ids,
                                                         attention_mask=a_attention_mask,
                                                         token_type_ids=a_token_type_ids)

        # encoding b
        candidate_seq_len = b_input_ids.shape[-1]

        b_input_ids = b_input_ids.reshape(-1, candidate_seq_len)
        b_token_type_ids = b_token_type_ids.reshape(-1, candidate_seq_len)
        b_attention_mask = b_attention_mask.reshape(-1, candidate_seq_len)

        b_token_type_ids = b_token_type_ids + 1

        b_input_ids, b_attention_mask, b_token_type_ids = clean_input_ids(b_input_ids, b_attention_mask,
                                                                          b_token_type_ids)

        b_lower_encoded_embeddings = self.lower_encoding(input_ids=b_input_ids,
                                                         attention_mask=b_attention_mask,
                                                         token_type_ids=b_token_type_ids)

        # encoding together
        joint_embeddings = self.joint_encoding(a_embeddings=a_lower_encoded_embeddings,
                                               b_embeddings=b_lower_encoded_embeddings,
                                               a_attention_mask=a_attention_mask,
                                               b_attention_mask=b_attention_mask,
                                               train_flag=train_flag)
        pooler = self.pooler_layer(joint_embeddings)
        logits = self.classifier(pooler)

        if train_flag:
            logits = logits.reshape(a_input_ids.shape[0], a_input_ids.shape[0])
            mask = torch.eye(logits.size(0)).to(logits.device)
            loss = F.log_softmax(logits, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            logits = logits.reshape(a_input_ids.shape[0], -1)
            return logits


def raise_test_exception():
    raise Exception("test end!")


def raise_assert_exception():
    raise Exception("assert true, please delete this assert!")
