from transformers import BertModel, BertForSequenceClassification, BertConfig
from transformers.models.bert.my_modeling_bert import MyBertModel, DecoderLayerChunk
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.nn.functional


# --------------------------------------
# fen ge xian
# --------------------------------------
# No matter whether use memory mechanism, all models whose input consists of Q and A can use this class
class QAModelConfig:
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


class QAModel(nn.Module):
    def __init__(self, config):

        super(QAModel, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
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

        with torch.no_grad():
            temp_attention_mask = attention_mask.clone().detach()

            # (batch_size, sequence_length)
            temp_mask = token_type_ids.clone().detach()

            # remove memory if existing
            temp_mask[temp_mask == 0] = 2
            temp_mask -= 1

            temp_mask = temp_mask * temp_attention_mask

            # remove cls, if cls is removed, some sentences may be empty
            # temp_mask[:, 0] = 0

            # remove sep--the last token whose type id equals 0
            sequence_len = temp_mask.sum(dim=-1) - 1
            sequence_len = sequence_len.unsqueeze(-1)
            temp_mask.scatter_(dim=1, index=sequence_len, src=torch.zeros((temp_mask.shape[0], 1), device=input_ids.device, dtype=temp_mask.dtype))

            # (batch_size, sequence_length, 1)
            temp_mask = temp_mask.unsqueeze(-1)

        last_hidden_state = last_hidden_state * temp_mask

        # get average embedding
        representations = last_hidden_state.sum(dim=1)

        representations = representations / sequence_len

        return representations

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask):

        if self.config.composition == 'avg':
            q_embeddings = self.get_rep_by_avg(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                               attention_mask=q_attention_mask)

            a_embeddings = self.get_rep_by_avg(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                               attention_mask=a_attention_mask)
        elif self.config.composition == 'pooler':
            q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                                  attention_mask=q_attention_mask)

            a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                                  attention_mask=a_attention_mask)
        else:
            raise Exception(f"Composition {self.config.composition} is not supported!!")

        # # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings)

        return logits


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

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)
        this_bert_config.num_labels = self.num_labels

        self.bert_model = BertForSequenceClassification.from_pretrained(config.pretrained_bert_path, config=this_bert_config)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

    def forward(self, input_ids, token_type_ids, attention_mask):

        # # 根据输入，进行思考, 思考的结果要选择性遗忘
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = out['logits']

        return logits


# --------------------------------------
# fen ge xian
# --------------------------------------
class ADecoderConfig:
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


class ADecoder(nn.Module):
    def __init__(self, config):

        super(ADecoder, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)

        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.decode_query = nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in range(self.this_bert_config.num_hidden_layers)])
        self.decode_layer_chunks = nn.ModuleList([DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)])

        # create classifier
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.init_decoder_params()

    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decode_layer_chunks[index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_query_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_query_static[k] = pretrained_static[new_k]

            this_static.update(updating_query_static)
            self.decode_layer_chunks[index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decode_query.state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decode_query.load_state_dict(query_static)

    # use the average embedding of last layer as sentence representation
    @staticmethod
    def get_rep_by_avg(embeddings, token_type_ids, attention_mask):
        with torch.no_grad():
            # (batch_size, sequence_length)
            temp_mask = token_type_ids.clone().detach()

            # remove tokens whose type id is 1
            temp_mask[temp_mask == 0] = 2
            temp_mask -= 1

            # remove tokens whose attention mask is 0
            temp_attention_mask = attention_mask.clone().detach()
            temp_mask = temp_mask * temp_attention_mask
            # remove cls, if cls is removed, some sentences may be empty
            # temp_mask[:, 0] = 0

            # remove sep--the last token
            sequence_len = temp_mask.sum(dim=-1) - 1
            sequence_len = sequence_len.unsqueeze(-1)
            temp_mask.scatter_(dim=1, index=sequence_len,
                               src=torch.zeros((temp_mask.shape[0], 1), device=embeddings.device,
                                               dtype=temp_mask.dtype))

            # (batch_size, sequence_length, 1)
            temp_mask = temp_mask.unsqueeze(-1)

        embeddings = embeddings * temp_mask

        # get average embedding
        representations = embeddings.sum(dim=1)

        representations = representations / sequence_len

        return representations

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask):

        # get a
        a_out = self.bert_model(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask,
                                enrich_answer_by_question=False)
        a_last_hidden_state = a_out['last_hidden_state']

        a_embeddings = self.get_rep_by_avg(a_last_hidden_state, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask)
        # (batch, 1, sequence dim)
        a_embeddings = a_embeddings.unsqueeze(-2)

        # get q and new a
        q_out = self.bert_model(input_ids=q_input_ids, token_type_ids=q_token_type_ids, attention_mask=q_attention_mask,
                                enrich_answer_by_question=True, answer_embeddings=a_embeddings,
                                decode_related_models=(self.decode_query, self.decode_layer_chunks))

        q_last_hidden_state, a_embeddings = q_out['last_hidden_state'], q_out['a_embedding']

        q_embeddings = self.get_rep_by_avg(q_last_hidden_state, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask)

        a_embeddings = a_embeddings.squeeze(-2)
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings)

        return logits


# --------------------------------------
# fen ge xian
# --------------------------------------
# 一个分类器，2个input len的向量作为输入
class QAClassifier(nn.Module):
    def __init__(self, input_len, keep_prob=0.9, num_labels=4):
        super(QAClassifier, self).__init__()

        self.linear1 = torch.nn.Linear(input_len * 2, input_len * 2, bias=True)
        self.bn1 = nn.BatchNorm1d(input_len * 2)
        self.linear2 = torch.nn.Linear(input_len * 2, input_len, bias=True)
        self.bn2 = nn.BatchNorm1d(input_len)
        self.linear3 = torch.nn.Linear(input_len, input_len, bias=True)
        self.bn3 = nn.BatchNorm1d(input_len)
        self.linear4 = torch.nn.Linear(input_len, num_labels, bias=True)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=1 - keep_prob)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)

    def forward(self, q_embedding, a_embedding):
        x = torch.cat((q_embedding, a_embedding), dim=-1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn1(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(x)

        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn3(x)

        x = self.linear4(x)

        return x


def raise_test_exception():
    raise Exception("test end!")
