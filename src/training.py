import fire
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader

from data import DataFrameTextClassificationDataset
from adapters import add_adapters, AdapterConfig, freeze_all_parameters, unfreeze_adapters


class ClassificationModel(pl.LightningModule):
    def __init__(self, bert: BertModel, tokenizer: BertTokenizer, n_labels: int,
                 dropout_prob: float, lr: float):
        super(ClassificationModel, self).__init__()
        self.n_labels = n_labels
        self.lr = lr

        self.tokenizer = tokenizer
        self.bert = bert
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(bert.pooler.dense.out_features, n_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        pooler_output = self.bert(input_ids, token_type_ids, attention_mask)['pooler_output']
        pooled_output = self.dropout(pooler_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_function = nn.CrossEntropyLoss()
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1)
            loss = loss_function(logits, labels)
            return loss, logits
        else:
            return logits

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']

        x_tokenized = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt")
        x_tokenized = {arg_name: tensor.to(self.device) for arg_name, tensor in x_tokenized.items()}
        loss, logits = self.forward(**x_tokenized, labels=y)

        y_pred = torch.argmax(logits, dim=1)
        y_true = torch.argmax(y, dim=1)
        batch_accuracy = (y_true == y_pred).float().mean()
        return {'loss': loss, 'batch_accuracy': batch_accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train(
    num_epochs: int = 5,
    n_workers: int = 4,
    adapter_size: int = 64,
    dropout: float = 0.4,
    lr: float = 3e-3,
    bert_model_name: str = 'bert-base-multilingual-cased',
    train_file: str = './data/rusentiment/rusentiment_random_posts.csv',
    test_file: str = './data/rusentiment/rusentiment_test.csv',
):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Shuffle training data and take 10% as validation
    train_df = train_df.sample(frac=1.)
    val_df = train_df.iloc[int(len(train_df) * 0.9):]
    train_df = train_df.iloc[:int(len(train_df) * 0.9)]

    train_dataset = DataFrameTextClassificationDataset(train_df)
    val_dataset = DataFrameTextClassificationDataset(val_df)
    test_dataset = DataFrameTextClassificationDataset(test_df)

    train_loader = DataLoader(train_dataset, num_workers=n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, num_workers=n_workers)

    # Load pre-trained model (weights)
    bert_model = BertModel.from_pretrained(bert_model_name)
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Add adapters and freeze all initial layers
    config = AdapterConfig(
        hidden_size=768, adapter_size=adapter_size,
        adapter_act='relu', adapter_initializer_range=1e-2
    )
    bert_model = add_adapters(bert_model, config)
    bert_model = freeze_all_parameters(bert_model)
    bert_model = unfreeze_adapters(bert_model)

    model = ClassificationModel(bert_model, tokenizer, n_labels=train_dataset.n_classes, dropout_prob=dropout, lr=lr)

    trainer = pl.Trainer(max_epochs=num_epochs, gpus=1, auto_scale_batch_size=True)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == '__main__':
    fire.Fire(train)
