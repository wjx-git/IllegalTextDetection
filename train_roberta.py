"""
@Project ：illegal_context_recognition
@File ：train_roberta.py
@IDE ：PyCharm
@Author ：wujx
"""
import argparse
import logging
import random
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from tqdm import trange, tqdm
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

from dataset import load_and_cache_examples, IllegalRecProcessor
from models.roberta_model import RoBERTaClassifier
from preprocess import user_vocab


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--task", default="illegal_rec", type=str, help="The name of the task to train")
parser.add_argument("--data_dir", default="data", type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.", )
parser.add_argument("--output_dir", default="outputs/roberta", type=str, help="Path to save model")
parser.add_argument("--model_dir", default="outputs/roberta/checkpoint-5", type=str, help="Path to load model")
parser.add_argument("--eval_dir", default="eval", type=str, help="Evaluation script, result directory", )
parser.add_argument("--train_file", default="train.txt", type=str, help="Train file")
parser.add_argument("--dev_file", default="dev.txt", type=str, help="dev file")
parser.add_argument("--test_file", default="test.txt", type=str, help="dev file")
parser.add_argument("--add_user_vocab", default=False, help="Whether add user dictionary")
parser.add_argument("--logs_file", default='logs', help="logs file")
parser.add_argument("--model_name_or_path", type=str, default="pre_trained_model/RoBERTa_wwm_ext",
                    help="Model Name or Path", )

parser.add_argument("--seed", type=int, default=77, help="random seed for initialization")
parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=16, type=int, help="Batch size for evaluation.")
parser.add_argument("--max_seq_len", default=150, type=int,
                    help="The maximum total input sequence length after tokenization.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.", )
parser.add_argument("--num_train_epochs", default=4, type=float,
                    help="Total number of training epochs to perform.", )
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.", )
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout for fully-connected layers", )

parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
parser.add_argument("--eval_steps", type=int, default=1500, help="Eval model every X updates steps.")
parser.add_argument("--save_steps", type=int, default=1500, help="Save checkpoint every X updates steps.", )

parser.add_argument("--do_train", default=True, help="Whether to run training.")
parser.add_argument("--do_test", default=False, help="Whether to run eval on the test set.")

args = parser.parse_args()


def save_model(args, model, tokenizer, global_step):
    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    tokenizer.save_pretrained(output_dir)
    logger.info('Saving model checkpoint to {}'.format(output_dir))


def load_model(model_file, device):
    if not os.path.exists(model_file):
        raise Exception("Model doesn't exists!")

    args = torch.load(os.path.join(model_file, "training_args.bin"))
    model = RoBERTaClassifier.from_pretrained(model_file, args=args)
    tokenizer = BertTokenizer.from_pretrained(model_file)
    model.eval()
    model.to(device)
    logger .info('******** Model Loaded ********')
    return model, tokenizer


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    if args.add_user_vocab:
        new_tokens = user_vocab(args)
        # with open('dicts/new.txt', 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(new_tokens))
        tokenizer.add_tokens(new_tokens)

    train_dataset = load_and_cache_examples(args, tokenizer, mode='train')
    dev_dataset = load_and_cache_examples(args, tokenizer, mode='dev')

    train_sample = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sample, batch_size=args.train_batch_size)

    train_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    labels = IllegalRecProcessor(args).labels
    bert_config = BertConfig.from_pretrained(args.model_name_or_path,
                                             num_labels=len(labels),
                                             finetuning_task=args.task,
                                             id2label={str(i): label for i, label in enumerate(labels)},
                                             label2id={label: i for i, label in enumerate(labels)})
    if args.add_user_vocab:
        bert_config.vocab_size = len(tokenizer)
    model = RoBERTaClassifier.from_pretrained(args.model_name_or_path, config=bert_config, args=args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Prepare optimizer and schedule
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.learning_rate
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.learning_rate
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=train_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", train_steps)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    if not os.path.exists(args.logs_file):
        os.mkdir(args.logs_file)
    writer = SummaryWriter(args.logs_file)

    global_step = 0
    train_loss = 0.0
    train_iter = trange(int(args.num_train_epochs), desc='Epoch')
    for _ in train_iter:
        epoch_iter = tqdm(train_loader, desc='Iteration')
        for step, batch in enumerate(epoch_iter):
            model.train()
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            loss, _ = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], labels=batch[3])
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()

            writer.add_scalar('train_loss', loss, global_step=global_step)

            train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step % args.eval_steps == 0:
                    labels, preds, eval_loss = evaluate(args, dev_dataset, model, mode='dev')
                    writer.add_scalar('eval_loss', eval_loss, global_step=global_step)
                    matrix = classification_report(labels, preds, digits=4)
                    logger.info('*********** Eval Results ***********')
                    logger.info(matrix)
                if global_step % args.save_steps == 0:
                    save_model(args, model, tokenizer, global_step)


def evaluate(args, dataset, model, mode):
    sample = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sample, batch_size=args.train_batch_size)

    logger.info("***** Running evaluation on %s dataset *****", mode)
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_loss = 0.0
    eval_steps = 0

    preds = []
    labels = []

    model.eval()
    for batch in tqdm(loader, desc='Evaluating'):
        batch = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            loss, logic = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], labels=batch[3])
            eval_loss += loss.item()
        eval_steps += 1

        preds.append(logic.cpu())
        labels.append(batch[3].cpu())

    eval_loss /= eval_steps

    preds = np.array(torch.cat(preds, dim=0))
    preds = np.argmax(preds, axis=1)
    labels = np.array(torch.cat(labels, dim=0))

    assert len(preds) == len(labels)

    return preds, labels, eval_loss


def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(args.model_dir, device)

    dataset = load_and_cache_examples(args, tokenizer, mode='test')

    labels, preds, _ = evaluate(args, dataset, model, 'test')
    matrix = classification_report(labels, preds, digits=4)
    logger.info('*********** Test Results ***********')
    logger.info(matrix)


if not args.do_train:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(args.model_dir, device)


def predict(text):
    """
    单条文本预测
    :param text:
    :return:
    """
    token = tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    token_type_ids = [1] * seq_len
    token_ids = tokenizer.convert_tokens_to_ids(token)
    if seq_len < args.max_seq_len:
        mask = [1] * seq_len + [0] * (args.max_seq_len - seq_len)
        token_ids += [0] * (args.max_seq_len - seq_len)
        token_type_ids += [1] * (args.max_seq_len - seq_len)
    else:
        mask = [1] * args.max_seq_len
        token_ids = token_ids[:args.max_seq_len]
        token_type_ids = token_type_ids[:args.max_seq_len]

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    mask = torch.tensor([mask], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)

    logic, _ = model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)
    preds = np.argmax(logic.detach().numpy(), axis=1)
    return preds[0]


if __name__ == '__main__':
    if args.do_train:
        train(args)
    if args.do_test:
        test(args)
