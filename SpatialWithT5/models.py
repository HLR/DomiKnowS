from torch.nn.modules.module import T
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, RobertaTokenizer, RobertaModel, \
    RobertaPreTrainedModel, AutoTokenizer, T5ForConditionalGeneration, T5PreTrainedModel, AutoModelForSeq2SeqLM, T5Tokenizer
from torch import nn
import torch
from torch.autograd import Variable
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


class T5WithLoraGenerativeCLF(nn.Module):
    def __init__(self, model_name, tokenizer, max_length, device="cpu", adapter=False):
        super().__init__()

        self.cur_device = device
        if adapter:
            print("Using Lora")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(tokenizer))

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
            self.model.resize_token_embeddings(len(tokenizer))

        self.train_t5_mode = True
        self.token_each_label = 0

        token_labels = tokenizer.token_labels
        self.interested_tokens = list(token_labels.values())
        self.output_length = max_length
        # self._space_token = tokenizer(" ")["input_ids"][0]
        # self._comma_token = tokenizer(",")["input_ids"][1]
        # self._eos_token = tokenizer(" ")["input_ids"][-1]
        # self.empty_pred_end = None
        # self.empty_pred = None
        # self.label_token_map, self.label_token_map_normalize, self.interested_tokens = self.tokenize_label(label,
        #                                                                                                    tokenizer)
        # self.output_length = self.max_group * self.token_each_label + 1  # (+ End of sentence)

    def forward(self, _, cat_input_ids, encoded_label):
        if self.train_t5_mode:
            return self._train_forward(cat_input_ids, encoded_label)
        return self._inference_forward(cat_input_ids)

    def _train_forward(self, cat_input_ids, encoded_label):
        input_ids = cat_input_ids[0, :, :]
        attention_mask = cat_input_ids[1, :, :]
        label_input_ids = encoded_label
        # print(input_ids, attention_mask, label_input_ids)
        # Need label to generate
        logits = self.model(input_ids, attention_mask=attention_mask,
                            labels=label_input_ids).logits
        # print(logits[:, :self.output_length, self.interested_tokens])
        return logits[:, :self.output_length, self.interested_tokens]

    def _inference_forward(self, cat_input_ids):
        input_ids = cat_input_ids[0, :, :]
        attention_mask = cat_input_ids[1, :, :]

        seq = self.model.generate(
            **{'input_ids': input_ids, 'attention_mask': attention_mask, 'min_new_tokens': self.output_length,
               'max_new_tokens': self.output_length})
        logits = self.model(input_ids, attention_mask=attention_mask,
                            decoder_input_ids=seq).logits
        return logits[:, :self.output_length, self.interested_tokens]

    def train(self: T, mode: bool = True) -> T:
        return_val = super().train(mode)
        self.train_t5_mode = mode
        return return_val


class Tokenizer:
    def __init__(self, model_id, label):
        self.token_labels = {}
        self.tokenizer = T5Tokenizer.from_pretrained(model_id)
        token_labels = self.tokenizer(label)["input_ids"]
        for i, label_tokens in enumerate(token_labels):
            # If label use one token, use the token
            if len(label_tokens) == 2:
                self.token_labels[label[i]] = label_tokens[0]
            else:
                # create the new token
                new_token = "<{:}>".format(label[i])
                self.tokenizer.add_tokens(new_token)
                self.token_labels[label[i]] = self.tokenizer(new_token)["input_ids"][0]

        # Adding EOS token and padding
        self.token_labels["eos"] = 1
        self.token_labels["pad"] = 0

    def __call__(self, _, questions, stories, return_long=True):
        prompts = []
        for ind, question in enumerate(questions):
            prompts.append("Answer based on the context:\n\n" + stories[ind] + "\n\n" + question)
        encoded_input = self.tokenizer(prompts, max_length=512, padding="max_length", truncation=True)

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        input_ids = torch.Tensor(input_ids) if not return_long else torch.LongTensor(input_ids)
        attention_mask = torch.Tensor(attention_mask) if not return_long else torch.LongTensor(attention_mask)
        return torch.stack((input_ids, attention_mask))

    def __len__(self):
        return len(self.tokenizer)
