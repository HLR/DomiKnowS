import pandas as pd
from tokenizers import BertWordPieceTokenizer
from torch import nn
# Load the BERT tokenizer.
import torch
from transformers import BertModel


def QA_reader(max_lenght=256, sample_num=1000 * 1000 * 10):
    print('Loading BERT tokenizer...')
    tokenizer = BertWordPieceTokenizer("data/squad1.1/bert-base-uncased-vocab.txt", lowercase=True)

    print("[PAD] ids is: ", tokenizer.token_to_id("[PAD]"))
    print("[SEP] ids is: ", tokenizer.token_to_id("[SEP]"))
    print("[CLS] ids is: ", tokenizer.token_to_id("[CLS]"))

    train = pd.read_json("data/squad1.1/train-v1.1.json")
    print(train.head())
    train = train.data

    X_tokens = []
    Y = []
    Y_S = []
    Y_E = []
    Masks = []
    SentNumber = []

    counter = 0
    ans_found = 0  # how many answers are not truncated
    for i in train:
        if counter == sample_num:
            break
        for j in i['paragraphs']:

            context = j["context"].lower()
            for k in j['qas']:
                # the question is k['question'].lower() and the context is in context variable
                tokens = tokenizer.encode(k['question'].lower(), context, )
                if counter < 2:
                    print("*" * 20 + "\n")
                    print(k['question'].lower(), context)
                    print(tokens.tokens)
                    print(tokens.ids)
                    print(tokens.type_ids)
                    print(tokens.attention_mask)
                    print(tokens.offsets)
                    print("")

                # MAKING x
                if len(tokens.ids) >= max_lenght:
                    x = tokens.ids[:max_lenght - 1]
                    x.append(102)
                else:
                    x = tokens.ids[:]
                # MAKING TYPE IDS ( SENT1 AND SENT2)
                sent_number = tokens.type_ids[:max_lenght]
                sent_number.extend([0 for ii in range(max_lenght - len(x))])

                # MAKING MASK AND PADDING x WITH [PAD]
                mask = [1 for ii in range(len(x))]
                mask.extend([0 for ii in range(max_lenght - len(x))])
                x.extend([0 for ii in range(max_lenght - len(x))])

                # MAKING Y
                Flag = True
                y = [0 for ii in range(max_lenght)]
                seen = 0  # to skip [CLS] and [PAD]
                for ii in range(len(tokens.offsets)):
                    if tokens.offsets[ii] == (0, 0):
                        seen += 1
                    elif seen == 2 and ii < max_lenght:
                        if tokens.offsets[ii][0] == k['answers'][0]['answer_start']:
                            y[ii] = 1
                        if tokens.offsets[ii][1] == k['answers'][0]["answer_start"] + len(k['answers'][0]["text"]):
                            if y[ii] == 1:
                                Flag = False
                            y[ii] = 2

                # arraye of starting point of asnwers otherwise last word
                if Flag:
                    if 1 in y:
                        Y_S.append(y.index(1))
                        ans_found += 1
                    else:
                        Y_S.append(max_lenght - 1)

                    # arraye of ending point of asnwers otherwise last word
                    if 2 in y:
                        Y_E.append(y.index(2))
                        ans_found += 1
                    else:
                        Y_E.append(max_lenght - 1)
                else:
                    Y_S.append(y.index(2))
                    Y_E.append(y.index(2))

                # ADDING THE INPUTS TO ARRAYS
                X_tokens.append(x)
                Y.append(y)
                Masks.append(mask)
                SentNumber.append(sent_number)
                # Y_S and Y_E are already made

                counter += 1
                if counter == sample_num:
                    break
            if counter == sample_num:
                break

    print("by setting a fixed lenght we have this percentage of answers in context found: ",
          ans_found / 2 / counter * 100, "%")

    return X_tokens, Y, Y_S, Y_E, Masks, SentNumber


class DariusQABERT(nn.Module):

    def __init__(self):
        super(DariusQABERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)
        self.sm = nn.Softmax(dim=1)

    # (input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
    # encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None)
    def forward(self, input_ids, attention_mask, token_type_ids, use_soft_max=False):
        # output
        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
        # pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) – Last layer hidden-state of the first token of the sequence (classification token) further processed
        # by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids.unsqueeze(dim=0),
                                                     attention_mask=attention_mask.unsqueeze(dim=0),
                                                     token_type_ids=token_type_ids.unsqueeze(dim=0), return_dict=False)
        # print(last_hidden_state.shape)
        # print(pooled_output.shape)
        ans = self.out(last_hidden_state)
        # add dropout?
        if use_soft_max:
            return self.sm(ans)
        return ans


class DariusQABERT_2(nn.Module):

    def __init__(self):
        super(DariusQABERT_2, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)
        self.sm = nn.Softmax(dim=1)

    # (input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
    # encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None)
    def forward(self, input_ids, attention_mask, token_type_ids, use_soft_max=False):
        # output
        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
        # pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) – Last layer hidden-state of the first token of the sequence (classification token) further processed
        # by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids,
                                                     attention_mask=attention_mask, token_type_ids=token_type_ids,
                                                     return_dict=False)
        # print(last_hidden_state.shape)
        # print(pooled_output.shape)
        ans = self.out(last_hidden_state)
        # add dropout?
        if use_soft_max:
            return self.sm(ans)
        return ans


def QA_reader_2(sample_num=1000 * 1000 * 10):
    train = pd.read_json("data/squad1.1/train-v1.1.json")
    train = train.data

    Context = []
    Questions = []
    Answer_start = []
    Answer_end = []
    counter = 0
    for i in train:
        if counter == sample_num:
            break
        for j in i['paragraphs']:

            context = j["context"].lower()
            Context.append(context)
            questions = []
            answer_start = []
            answer_end = []
            for k in j['qas']:
                # the question is k['question'].lower() and the context is in context variable
                questions.append(k['question'].lower())
                answer_start.append(k['answers'][0]["answer_start"])
                answer_end.append(k['answers'][0]["answer_start"] + len(k['answers'][0]["text"]))
                counter += 1
                if counter == sample_num:
                    break
            Questions.append(questions)
            Answer_start.append(answer_start)
            Answer_end.append(answer_end)
            if counter == sample_num:
                break
    return Context, Questions, Answer_start, Answer_end


def make_questions(context, questions, start, end):
    #print("make_questions", context,questions, start, end)
    return torch.ones((len(questions[0]), 1)), [context for i in range(len(questions[0]))], questions[0], start[0], end[0]


class QA_Tokenize:
    def __init__(self, max_lenght):
        self.tokenizer = BertWordPieceTokenizer("data/squad1.1/bert-base-uncased-vocab.txt", lowercase=True)
        self.max_lenght = max_lenght

    def __call__(self,_, context, question):
        print("")
        print("QA_Tokenize: ",len(context),len(question))
        print("")
        X = []
        Mask = []
        Sent_number = []
        Offsets = []
        for i, j in zip(context, question):

            tokens = self.tokenizer.encode(j, i[0])

            # MAKING x
            if len(tokens.ids) >= self.max_lenght:
                x = tokens.ids[:self.max_lenght - 1]
                x.append(102)
            else:
                x = tokens.ids[:]
            # MAKING TYPE IDS ( SENT1 AND SENT2)
            sent_number = tokens.type_ids[:self.max_lenght]
            sent_number.extend([0 for ii in range(self.max_lenght - len(x))])

            # MAKING MASK AND PADDING x WITH [PAD]
            mask = [1 for ii in range(len(x))]
            mask.extend([0 for ii in range(self.max_lenght - len(x))])
            x.extend([0 for ii in range(self.max_lenght - len(x))])

            X.append(x)
            Mask.append(mask)
            Sent_number.append(sent_number)
            Offsets.append(tokens.offsets[:self.max_lenght])

        return torch.LongTensor(X), torch.LongTensor(Mask), torch.LongTensor(Sent_number), Offsets


class PutNear:
    def __init__(self, max_lenght):
        self.max_lenght = max_lenght

    def __call__(self, x1, x2, tokens_list, offsets):
        print("*"*10)
        print(x1)
        print(x2)
        print(tokens_list)
        print(offsets)
        print("*"*10)
        Y_S = []
        Y_E = []
        for k1, k2, tokens, offset in zip(x1, x2, tokens_list, offsets):
            # MAKING Y
            Flag = True
            y = [0 for ii in range(self.max_lenght)]
            seen = 0  # to skip [CLS] and [PAD]
            for ii in range(len(offset)):
                if offset[ii] == (0, 0):
                    seen += 1
                elif seen == 2 and ii < self.max_lenght:
                    if offset[ii][0] == k1:
                        y[ii] = 1
                    if offset[ii][1] == k2:
                        if y[ii] == 1:
                            Flag = False
                        y[ii] = 2

            # arraye of starting point of asnwers otherwise last word
            if Flag:
                if 1 in y:
                    Y_S.append(y.index(1))
                else:
                    Y_S.append(self.max_lenght - 1)

                # arraye of ending point of asnwers otherwise last word
                if 2 in y:
                    Y_E.append(y.index(2))
                else:
                    Y_E.append(self.max_lenght - 1)
            else:
                Y_S.append(y.index(2))
                Y_E.append(y.index(2))

        return torch.cat((torch.LongTensor(Y_S).unsqueeze(dim=1), torch.LongTensor(Y_E).unsqueeze(dim=1)), dim=1)
