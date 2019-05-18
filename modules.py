import logging
from typing import NamedTuple, Callable, Union

import torch.nn as nn
from pytorch_pretrained_bert.modeling import ACT2FN, BertLayerNorm, BertModel, BertSelfOutput

logging.basicConfig(level=logging.INFO)


class AdapterConfig(NamedTuple):
    hidden_size: int
    adapter_size: int
    adapter_act: Union[str, Callable]
    adapter_initializer_range: float


class Adapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        if isinstance(config.adapter_act, str):
            self.activation = ACT2FN[config.adapter_act]
        else:
            self.activation = config.adapter_act

        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected


class BertAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: BertSelfOutput,
                 config: AdapterConfig):
        super(BertAdaptedSelfOutput, self).__init__()
        self.self_output = self_output
        self.adapter = Adapter(config)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def adapt_bert_self_output(config: AdapterConfig):
    return lambda self_output: BertAdaptedSelfOutput(self_output, config=config)


def add_adapters(bert_model: BertModel,
                 config: AdapterConfig) -> BertModel:
    bert_encoder = bert_model.encoder
    for i in range(len(bert_model.encoder.layer)):
        bert_encoder.layer[i].attention.output = adapt_bert_self_output(config)(
            bert_encoder.layer[i].attention.output)

    # Freeze all parameters
    for param in bert_model.parameters():
        param.requires_grad = False
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in bert_model.named_modules():
        if isinstance(sub_module, (Adapter, BertLayerNorm)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return bert_model


class ClassificationModel(nn.Module):
    def __init__(self, bert: BertModel, n_labels: int, dropout_prob: float):
        super(ClassificationModel, self).__init__()
        self.n_labels = n_labels
        self.bert = bert

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(bert.pooler.dense.out_features, n_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.n_labels), labels.view(-1))
            return loss, logits
        else:
            return logits
