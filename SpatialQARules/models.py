from torch.nn.modules.module import T
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, RobertaTokenizer, RobertaModel, \
    RobertaPreTrainedModel, AutoTokenizer, T5ForConditionalGeneration, T5PreTrainedModel, AutoModelForSeq2SeqLM
from torch import nn
import torch
from torch.autograd import Variable
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


class BERTTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, _, question, story):
        encoded_input = self.tokenizer(question, story, padding="max_length", truncation=True)
        input_ids = encoded_input["input_ids"]
        return torch.LongTensor(input_ids)


class RoBERTaTokenizer:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __call__(self, _, question, story):
        encoded_input = self.tokenizer(question, story, padding="max_length", truncation=True)
        input_ids = encoded_input["input_ids"]
        return torch.LongTensor(input_ids)


class MultipleClassYN(BertPreTrainedModel):
    def __init__(self, config, device="cpu", drp=False):
        super().__init__(config)

        if drp:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.cur_device = device
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes = 2
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output)

        return output


class MultipleClassYNRoberta(RobertaPreTrainedModel):
    def __init__(self, config, device="cpu", drp=False):
        super().__init__(config)

        if drp:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.cur_device = device
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes = 2
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input_ids):
        outputs = self.roberta(input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output)

        return output


class MultipleClassYN_Hidden(BertPreTrainedModel):
    def __init__(self, config, device="cpu", drp=False):
        super().__init__(config)

        if drp:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.cur_device = device
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        return pooled_output


class MultipleClassYN_Hidden_Roberta(RobertaPreTrainedModel):
    def __init__(self, config, device="cpu", drp=False):
        super().__init__(config)

        if drp:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.cur_device = device
        self.bert = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        return pooled_output


class ClassifyLayer(nn.Module):
    def __init__(self, hidden_size, device="cpu", drp=False):
        super().__init__()

        self.num_classes = 2
        self.classifier = nn.Linear(hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, pooled_output):
        output = self.classifier(pooled_output)

        return output


class ClassifyLayer2(nn.Module):
    def __init__(self, hidden_size, hidden_layer=1, device="cpu", drp=False):
        super().__init__()

        self.num_classes = 2
        layer_parameters = [hidden_size] + [256 for i in range(hidden_layer - 1)] + [self.num_classes]

        all_layer = []
        for i in range(len(layer_parameters) - 1):
            all_layer.append(nn.Linear(layer_parameters[i], layer_parameters[i + 1]))
        self.classifier = nn.Sequential(*all_layer)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, pooled_output):
        logits = self.classifier(pooled_output)
        # logits = self.sigmoid(logits)

        return logits


class MultipleClassYNT5(nn.Module):
    def __init__(self, config, device="cpu", adapter=False):
        super().__init__()

        self.cur_device = device
        if adapter:
            print("Using Lora")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config)

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.config.use_cache = False
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.num_classes = 2
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.output_size = 1

    def forward(self, input_ids):
        decoder_id = torch.tensor([[self.tokenizer.pad_token_id] * self.output_size] * input_ids.size(0)).to(
            self.cur_device)
        logits = self.model(input_ids, decoder_input_ids=decoder_id)[0]
        tokens = torch.argmax(logits, dim=2)
        # Yes token is 2163, No token is 465
        # Output ["Yes", "No"]
        logits = logits.squeeze(1)
        selected_logits = logits[:, [2163, 465]]
        output = self.softmax(selected_logits)

        return output


class T5Tokenizer:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(self, _, questions, stories):
        prompts = []
        for ind, question in enumerate(questions):
            prompts.append("You will answer the question based on the following context: " + stories[
                ind] + "\n Question: " + question)
        encoded_input = self.tokenizer(prompts, padding="max_length", truncation=True)
        input_ids = encoded_input["input_ids"]
        return torch.LongTensor(input_ids)


class MultipleClassFRT5(nn.Module):
    def __init__(self, model_name, expected_label, device="cpu", adapter=False):
        super().__init__()

        self.cur_device = device
        if adapter:
            print("Using Lora")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            lora_config = LoraConfig(
                r=64,
                lora_alpha=64,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.config.use_cache = False
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_tokens = self.tokenizer(expected_label)["input_ids"]
        self.map_token = {}
        self.unique_token = {}
        # FIXTHIS
        for token_label in self.label_tokens:
            for token in token_label:
                self.unique_token[token] = 1
        self.unique_token = [token for token in self.unique_token.keys()]
        for i, token in enumerate(self.unique_token):
            self.map_token[token] = i
        self.num_classes = 2
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.map_label = {label: i for i, label in enumerate(expected_label)}
        self.second_model = nn.Sequential(nn.Linear(len(self.unique_token) * 2, len(expected_label)),
                                          nn.Sigmoid())

    def forward(self, input_ids):
        # Force decoder to output 2 token
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]] * input_ids.size()[0]).to(
            self.cur_device)
        logits = self.model(input_ids, decoder_input_ids=decoder_input_ids)[0]
        first_word = logits.argmax(dim=2)

        decoder_input_ids = torch.concat((decoder_input_ids, first_word), dim=-1)

        logits = self.model(input_ids, decoder_input_ids=decoder_input_ids)[0]
        # Only output the selecting value of token in the unique_tokens
        logits = logits[:, :, self.unique_token]
        logits = torch.concat((logits[:, 0, :], logits[:, 1, :]), dim=1)
        output = self.second_model(logits)
        # tokens = torch.argmax(logits, dim=2)
        # Yes token is 2163, No token is 465
        # Output ["Yes", "No"]
        # logits = logits.squeeze(1)

        return output


class ClassifyLabelT5(nn.Module):
    def __init__(self, label_word, map_index, device="cpu", drp=False):
        super().__init__()
        self.map_index = map_index[label_word]

    def forward(self, logits):
        output = logits[:, self.map_index]
        output = output.reshape(-1, 1)
        output = torch.cat((torch.sub(torch.ones_like(output), output), output), dim=-1)
        return output


class T5WithLora(nn.Module):
    def __init__(self, model_name, device="cpu", adapter=False):
        super().__init__()

        self.cur_device = device
        if adapter:
            print("Using Lora")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            lora_config = LoraConfig(
                r=32,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.01,
                bias="lora_only",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.config.use_cache = False
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_classes = 2

    def forward(self, _, cat_input_ids):
        input_ids = cat_input_ids[0, :, :]
        attention_mask = cat_input_ids[1, :, :]
        logits = self.model.generate(**{'input_ids': input_ids, 'attention_mask': attention_mask}, max_new_tokens=20)
        return logits

    def loss(self, cat_input_ids, cat_encoded_label):
        input_ids = cat_input_ids[0, :, :]
        attention_mask = cat_input_ids[1, :, :]
        label_input_ids = cat_encoded_label[0, :, :]
        label_attention_mask = cat_encoded_label[1, :, :]
        loss_t5 = self.model(input_ids, attention_mask=attention_mask, labels=label_input_ids,
                             decoder_attention_mask=label_attention_mask).loss
        return loss_t5


class T5TokenizerInput:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(self, _, questions, stories, return_long=False):
        prompts = []
        for ind, question in enumerate(questions):
            prompts.append("Answer based on the context:\n\n" + stories[ind] + "\n\n" + question)
        encoded_input = self.tokenizer(prompts, padding="max_length", truncation=True)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        input_ids = torch.Tensor(input_ids) if not return_long else torch.LongTensor(input_ids)
        attention_mask = torch.Tensor(attention_mask) if not return_long else torch.LongTensor(attention_mask)
        return torch.stack((input_ids, attention_mask))


class T5TokenizerOutput:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(self, _, labels, return_long=False):
        encoded_input = self.tokenizer(labels, padding="max_length", truncation=True)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        input_ids = torch.Tensor(input_ids) if not return_long else torch.LongTensor(input_ids)
        attention_mask = torch.Tensor(attention_mask) if not return_long else torch.LongTensor(attention_mask)
        return torch.stack((input_ids, attention_mask))


class T5TokenizerDecoder:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(self, _, encoded):
        decoded = self.tokenizer.batch_decode(encoded, skip_special_tokens=True, clean_up_toenization_spaces=True)
        return decoded


class T5LossFunction(torch.nn.Module):
    def __init__(self, T5_model):
        super().__init__()
        self.T5_model = T5_model

    def forward(self, input, target):
        input = input.long()
        target = target.long()
        loss = self.T5_model.loss(input, target)
        return loss


class T5WithLoraGenerativeCLF(nn.Module):
    def __init__(self, model_name, label, tokenizer, output_length=32, device="cpu", adapter=False):
        super().__init__()

        self.cur_device = device
        if adapter:
            print("Using Lora")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.01,
                bias="lora_only",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.config.use_cache = False
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.train_t5_mode = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_classes = 2

        label_tokens = tokenizer(label + [","])["input_ids"]
        self.interested_tokens = []
        for tokens in label_tokens:
            self.interested_tokens.extend(tokens)
        self.interested_tokens = list(set(self.interested_tokens))
        self.output_length = output_length
        self.hidden_size = len(self.interested_tokens) * self.output_length

    def forward(self, _, cat_input_ids, cat_encoded_label):
        if self.train_t5_mode:
            return self.train_forward(cat_input_ids, cat_encoded_label)
        return self.inference_forward(cat_input_ids)

    def train_forward(self, cat_input_ids, cat_encoded_label):
        input_ids = cat_input_ids[0, :, :]
        attention_mask = cat_input_ids[1, :, :]
        label_input_ids = cat_encoded_label[0, :, :]
        label_attention_mask = cat_encoded_label[1, :, :]
        logits = self.model(input_ids, attention_mask=attention_mask,
                            labels=label_input_ids, decoder_attention_mask=label_attention_mask).logits
        return self.transform_logits(logits)

    def inference_forward(self, cat_input_ids):
        input_ids = cat_input_ids[0, :, :]
        attention_mask = cat_input_ids[1, :, :]

        seq = self.model.generate(
            **{'input_ids': input_ids, 'attention_mask': attention_mask, 'min_new_tokens': self.output_length,
               'max_new_tokens': self.output_length + 1})
        logits = self.model(input_ids, attention_mask=attention_mask,
                            decoder_input_ids=seq).logits
        return self.transform_logits(logits)

    def transform_logits(self, logit):
        logit = logit[:, :self.output_length, self.interested_tokens].flatten(1, 2)  # Combine last two dimensions
        return logit

    def train(self: T, mode: bool = True) -> T:
        return_val = super().train(mode)
        print("Setting Training on T5 to {:}".format(mode))
        self.train_t5_mode = mode
        return return_val


class T5WithLoraGenerativeCLF2(nn.Module):
    def __init__(self, model_name, label, max_group, group_label, tokenizer, device="cpu", adapter=False):
        super().__init__()

        self.cur_device = device
        if adapter:
            print("Using Lora")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.01,
                bias="lora_only",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.config.use_cache = False
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.train_t5_mode = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.group_label = group_label
        self.max_group = max_group

        self.token_each_label = 0

        self._space_token = tokenizer(" ")["input_ids"][0]
        self._comma_token = tokenizer(",")["input_ids"][1]
        self._eos_token = tokenizer(" ")["input_ids"][-1]
        self.empty_pred_end = None
        self.empty_pred = None
        self.label_token_map, self.label_token_map_normalize, self.interested_tokens = self.tokenize_label(label,
                                                                                                           tokenizer)
        self.output_length = self.max_group * self.token_each_label + 1  # (+ End of sentence)

    def tokenize_label(self, labels, tokenizer):
        labels = labels + [" "]
        label_tokens = tokenizer(labels)["input_ids"]
        self.token_each_label = max([len(tokens) for tokens in label_tokens])
        label_tokens_map = {}
        interested_tokens = []
        for label, label_token in zip(labels, label_tokens):
            # Format the token
            label_token[-1] = self._space_token
            label_token += [self._space_token] * (self.token_each_label - len(label_token))
            label_token[-1] = self._comma_token if self.group_label.get(label,
                                                                        0) != self.max_group - 1 else self._eos_token
            interested_tokens.extend(label_token)
            label_tokens_map[label] = label_token

        interested_tokens = sorted(list(set(interested_tokens)))
        map_token_loc = {tokens: i for i, tokens in enumerate(interested_tokens)}
        label_tokens_map_normalize = {}
        for label, tokens in label_tokens_map.items():
            new_tokens = [map_token_loc[token] for token in tokens]
            label_tokens_map_normalize[label] = new_tokens

        self.empty_pred_end = ([map_token_loc[self._space_token]] * (self.token_each_label - 1)
                               + [map_token_loc[self._comma_token]])

        self.empty_pred = [map_token_loc[self._space_token]] * (self.token_each_label - 1) + [map_token_loc[self._eos_token]]

        return label_tokens_map, label_tokens_map_normalize, interested_tokens

    def forward(self, _, cat_input_ids, cat_encoded_label):
        if self.train_t5_mode:
            return self._train_forward(cat_input_ids, cat_encoded_label)
        return self._inference_forward(cat_input_ids)

    def _train_forward(self, cat_input_ids, cat_encoded_label):
        input_ids = cat_input_ids[0, :, :]
        attention_mask = cat_input_ids[1, :, :]
        label_input_ids = cat_encoded_label[0, :, :]
        label_attention_mask = cat_encoded_label[1, :, :]
        # Need label to generate
        logits = self.model(input_ids, attention_mask=attention_mask,
                            labels=label_input_ids, decoder_attention_mask=label_attention_mask).logits
        return self.transform_logits(logits)

    def _inference_forward(self, cat_input_ids):
        input_ids = cat_input_ids[0, :, :]
        attention_mask = cat_input_ids[1, :, :]

        seq = self.model.generate(
            **{'input_ids': input_ids, 'attention_mask': attention_mask, 'min_new_tokens': self.output_length,
               'max_new_tokens': self.output_length + 1})

        logits = self.model(input_ids, attention_mask=attention_mask,
                            decoder_input_ids=seq).logits

        return self.transform_logits(logits)

    def transform_logits(self, logit):
        logits = logit[:, :self.output_length, self.interested_tokens]
        return logits

    def train(self: T, mode: bool = True) -> T:
        return_val = super().train(mode)
        print("Setting Training on T5 to {:}".format(mode))
        self.train_t5_mode = mode
        return return_val


class T5LocationClassification(nn.Module):
    def __init__(self, token_loc, candidate_output_token, device="cpu"):
        super().__init__()
        self.st_token, self.ed_token = token_loc
        self.candidate_output_token = candidate_output_token
        print(self.candidate_output_token)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, _, logits):
        logits = logits[:, self.st_token:self.ed_token, :]
        all_prob = torch.Tensor().requires_grad_().to(self.device)
        for token_label in self.candidate_output_token:
            label_prob = logits[:, 0, token_label[0]]
            for i, label_token in enumerate(token_label):
                if i == 0:
                    continue
                label_prob = torch.mul(label_prob, logits[:, i, token_label[i]])

            label_prob = label_prob.reshape(-1, 1)
            all_prob = torch.concat((all_prob, label_prob), dim=-1)

        all_prob = self.softmax(all_prob)
        return all_prob


class LabelClassification(nn.Module):
    def __init__(self, index_label):
        super().__init__()
        self.index_label = index_label

    def forward(self, _, all_prob):
        label_prob = all_prob[:, self.index_label].reshape(-1, 1)
        prob = torch.concat((torch.ones_like(label_prob) - label_prob, label_prob), dim=-1)
        return prob
