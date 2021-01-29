import fire
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data import TokenizedDataFrameDataset
from modules import add_adapters, AdapterConfig, ClassificationModel


def train(
    num_epochs: int = 5,
    batch_size: int = 10,
    n_workers: int = 4,
    adapter_size: int = 64,
    num_threads: int = 4,
    dropout: float = 0.4,
    lr: float = 3e-3,
    bert_model_name: str = 'bert-base-multilingual-cased',
    tensorboard: str = 'default_tb',
    train_file: str = './data/rusentiment/rusentiment_random_posts.csv',
    test_file: str = './data/rusentiment/rusentiment_test.csv',
):
    torch.set_num_threads(num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(tensorboard)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Shuffle training data and take 10% as validation
    train_df = train_df.sample(frac=1.)
    val_df = train_df.iloc[int(len(train_df) * 0.9):]
    train_df = train_df.iloc[:int(len(train_df) * 0.9)]

    train_dataset = TokenizedDataFrameDataset(tokenizer, df=train_df)
    val_dataset = TokenizedDataFrameDataset(tokenizer, df=val_df)
    test_dataset = TokenizedDataFrameDataset(tokenizer, df=test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers)

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(bert_model_name)

    config = AdapterConfig(
        hidden_size=768, adapter_size=adapter_size,
        adapter_act='relu', adapter_initializer_range=1e-2
    )
    model = add_adapters(model, config)

    model = ClassificationModel(model, n_labels=train_dataset.n_classes, dropout_prob=dropout)

    model.eval()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, amsgrad=True)

    print('Model have initialized')
    for epoch_i in range(num_epochs):
        model.train()
        labels = []
        predictions = []
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input_ids, input_mask, segment_ids = batch['x']
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            y = batch['y']
            y = y.to(device)

            loss, logits = model.forward(input_ids, input_mask, segment_ids, labels=y)
            loss.backward()

            optimizer.step()
            labels.append(torch.argmax(y, dim=1))
            predictions.append(torch.argmax(logits, dim=1))
        labels = torch.cat(labels).long()
        predictions = torch.cat(predictions).long()
        train_accuracy = (labels == predictions).float().mean()

        model.eval()
        labels = []
        predictions = []
        for batch in tqdm(val_loader):
            input_ids, input_mask, segment_ids = batch['x']
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            y = batch['y']
            y = y.to(device)

            loss, logits = model.forward(input_ids, input_mask, segment_ids, labels=y)
            labels.append(torch.argmax(y, dim=1))
            predictions.append(torch.argmax(logits, dim=1))
        labels = torch.cat(labels).long()
        predictions = torch.cat(predictions).long()

        valid_accuracy = (labels == predictions).float().mean()
        print(f'Epoch: {epoch_i}\tTrain Accuracy: {train_accuracy}\tVal Accuracy: {valid_accuracy}')

    model.eval()
    labels = []
    predictions = []
    for batch in tqdm(test_loader):
        input_ids, input_mask, segment_ids = batch['x']
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        y = batch['y']
        y = y.to(device)

        loss, logits = model.forward(input_ids, input_mask, segment_ids, labels=y)
        labels.append(torch.argmax(y, dim=1))
        predictions.append(torch.argmax(logits, dim=1))
    labels = torch.cat(labels).long()
    predictions = torch.cat(predictions).long()
    print(f'Epoch: {epoch_i}\tTest Accuracy: {(labels == predictions).float().mean()}')


if __name__ == '__main__':
    fire.Fire(train)
