"""
@Project ：illegal_context_recognition
@File ：roberta_model.py
@IDE ：PyCharm
@Author ：wujx
@Date ：2021/11/17 15:11
"""
import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel


class RoBERTaClassifier(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(RoBERTaClassifier, self).__init__(bert_config)
        self.roberta = BertModel(config=bert_config)
        if args.add_user_vocab:
            self.roberta.resize_token_embeddings(bert_config.vocab_size)
        self.num_labels = bert_config.num_labels

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.classifier = nn.Linear(bert_config.hidden_size, bert_config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs[1]
        # cls_output = self.fc(cls_output)
        cls_output = self.dropout(cls_output)
        logic = self.classifier(cls_output)

        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logic.view(-1, self.num_labels), labels.view(-1))
            return loss, logic

        return logic, None


















