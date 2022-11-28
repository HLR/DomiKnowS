from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForTokenClassification


class BIO_Model(BertForTokenClassification):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, label_masks=None,
                position_ids=None, head_mask=None, labels=None):
        # print('input_ids:', input_ids.size())
        # print('attention_mask:', attention_mask.size())
        # print('token_type_ids:', token_type_ids.size())
        # print('label_masks:', label_masks.size())
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]  # (b, MAX_LEN, 768)

        # token_reprs = [embedding[mask] for mask, embedding in zip(label_masks, sequence_output)]
        # token_reprs = pad_sequence(sequences=token_reprs, batch_first=True,
        #                            padding_value=-1)  # (b, local_max_len, 768)
        # sequence_output = self.dropout(token_reprs)
        logits = self.classifier(sequence_output)  # (b, local_max_len, num_labels)

        # outputs = (logits,)
        print(logits.size())
        outputs = logits
        # if labels is not None:
        #     labels = [label[mask] for mask, label in zip(label_masks, labels)]
        #     labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
        #     loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
        #     mask = labels != -1
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     loss /= mask.float().sum()
        #     outputs = (loss,) + outputs + (labels,)
        print('chen: model go to the last step!!')
        return outputs  # (loss), scores, (hidden_states), (attentions)
