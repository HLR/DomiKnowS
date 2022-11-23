import argparse
import logging

import torch
import torch.nn.functional as F
from seqeval.metrics import accuracy_score, f1_score, classification_report
from torch.utils import data
from tqdm import trange, tqdm
# from transformers import BertTokenizer, AdamW, WarmupLinearSchedule
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from data_set import BIOProcessor, BIODataSet
from model import BIO_Model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(train_iter, eval_iter, model, optimizer, scheduler, num_epochs):
    logger.info("starting to train")
    max_grad_norm = 1.0  # should be a flag
    for _ in trange(num_epochs, desc="Epoch"):
        # TRAIN loop
        model = model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(tqdm(train_iter)):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
            # forward pass
            loss, logits, labels = model(b_input_ids, token_type_ids=b_token_type_ids,
                                         attention_mask=b_input_mask, labels=b_labels,
                                         label_masks=b_label_masks)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        # print train loss per epoch
        logger.info("Train loss: {}".format(tr_loss / nb_tr_steps))
        eval(eval_iter, model)


def eval(iter_data, model):
    logger.info("starting to evaluate")
    model = model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions, true_labels = [], []
    for batch in tqdm(iter_data):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch

        with torch.no_grad():
            tmp_eval_loss, logits, reduced_labels = model(b_input_ids,
                                                          token_type_ids=b_token_type_ids,
                                                          attention_mask=b_input_mask,
                                                          labels=b_labels,
                                                          label_masks=b_label_masks)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        reduced_labels = reduced_labels.to('cpu').numpy()

        labels_to_append = []
        predictions_to_append = []

        for prediction, r_label in zip(logits, reduced_labels):
            preds = []
            labels = []
            for pred, lab in zip(prediction, r_label):
                if lab.item() == -1:  # masked label; -1 means do not collect this label
                    continue
                preds.append(pred)
                labels.append(lab)
            predictions_to_append.append(preds)
            labels_to_append.append(labels)

        predictions.extend(predictions_to_append)
        true_labels.append(labels_to_append)

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info("Validation loss: {}".format(eval_loss))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    logger.info("Seq eval accuracy: {}".format(accuracy_score(valid_tags, pred_tags)))
    logger.info("F1-Score: {}".format(f1_score(valid_tags, pred_tags)))
    logger.info("Classification report: -- ")
    logger.info(classification_report(valid_tags, pred_tags))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pretrained_model_name", type=str, default="bert-base-cased")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--existing_model_path", type=str, default=None)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)

    bio_tagging_processor = BIOProcessor()

    train_examples = bio_tagging_processor.get_train_examples(args.data_dir)
    val_examples = bio_tagging_processor.get_dev_examples(args.data_dir)
    test_examples = bio_tagging_processor.get_test_examples(args.data_dir)

    tags_vals = bio_tagging_processor.get_labels()
    label_map = {}

    for (i, label) in enumerate(tags_vals):
        label_map[label] = i

    train_dataset = BIODataSet(data_list=train_examples, tokenizer=tokenizer, label_map=label_map,
                               max_len=128)

    eval_dataset = BIODataSet(data_list=val_examples, tokenizer=tokenizer, label_map=label_map,
                              max_len=128)

    test_dataset = BIODataSet(data_list=test_examples, tokenizer=tokenizer, label_map=label_map,
                              max_len=128)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=4)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=4)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=4)

    num_epochs = args.n_epochs

    model = BIO_Model.from_pretrained(args.pretrained_model_name,
                                            num_labels=len(label_map)).to(device)

    if args.existing_model_path is not None:
        logger.info("Loading model from {}".format(args.existing_model_path))
        model.load_state_dict(torch.load(args.existing_model_path))

    num_train_optimization_steps = int(len(train_examples) / args.batch_size) * num_epochs

    FULL_FINETUNING = True

    lr = args.lr

    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    warmup_steps = int(0.1 * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    # scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=warmup_steps,
    #                                  t_total=num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                     num_training_steps=num_train_optimization_steps)

    if args.train:
        train(train_iter, eval_iter, model, optimizer, scheduler, num_epochs)
        logger.info("--Starting test evaluation now!---")
        torch.save(model.state_dict(), 'model.torch')
        eval(test_iter, model)
