import argparse

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data import TokenizedDataFrameDataset
from modules import add_adapters, AdapterConfig, ClassificationModel

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='bert_adapter')
    parser.add_argument('--num_epochs', type=int, default=5, metavar='NI',
                        help='num epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='S')
    parser.add_argument('--n_workers', type=int, default=4, metavar='S')
    parser.add_argument('--adapter_size', type=int,
                        default=64, metavar='S')
    parser.add_argument('--num-threads', type=int, default=4,
                        help='num threads (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='D',
                        help='dropout rate (default: 0.4)')
    parser.add_argument('--bert_model_name', type=str, default='bert-base-multilingual-cased',
                        help='Bert model name (ex. "bert-base-multilingual-cased")')
    parser.add_argument('--tensorboard', type=str, default='default_tb', metavar='TB',
                        help='Name for tensorboard model')
    parser.add_argument('--train_file', type=str,
                        default='./data/rusentiment/rusentiment_random_posts.csv',
                        metavar='TB',
                        help='Path to RuSentiment train')
    parser.add_argument('--test_file', type=str,
                        default='./data/rusentiment/rusentiment_test.csv',
                        metavar='TB',
                        help='Path to RuSentiment test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(args.tensorboard)

    torch.set_num_threads(args.num_threads)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

    train_dataset = TokenizedDataFrameDataset(tokenizer, file_path=args.train_file)
    val_dataset = TokenizedDataFrameDataset(tokenizer, file_path=args.train_file)
    train_dataset.df = train_dataset.df.iloc[:int(len(train_dataset) * 0.9)]
    val_dataset.df = val_dataset.df.iloc[int(len(train_dataset) * 0.9):]
    test_dataset = TokenizedDataFrameDataset(tokenizer, file_path=args.test_file)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.n_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                              num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(args.bert_model_name)

    config = AdapterConfig(
        hidden_size=768, adapter_size=args.adapter_size,
        adapter_act='relu', adapter_initializer_range=1e-2
    )
    model = add_adapters(model, config)

    model = ClassificationModel(model, n_labels=len(train_dataset.y_labels),
                                dropout_prob=0.0)

    model.eval()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001, amsgrad=True)

    print('Model have initialized')
    for i in range(args.num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input_ids, input_mask, segment_ids = batch['x']
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            y = batch['y']
            y = y.to(device)

            loss, _ = model.forward(input_ids, input_mask, segment_ids, labels=y)
            loss.backward()

            optimizer.step()

        model.eval()
        labels = []
        predictions = []
        for batch in tqdm(val_loader):
            optimizer.zero_grad()

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
        print(f'Epoch: {i}\tVal Accuracy: {(labels == predictions).float().mean()}')

    model.eval()
    labels = []
    predictions = []
    for batch in tqdm(test_loader):
        optimizer.zero_grad()

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
    print(f'Epoch: {i}\tTest Accuracy: {(labels == predictions).float().mean()}')
