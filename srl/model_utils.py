import torch

def pad(*sequences):
    padded_sequences = []
    
    def pad_indiv(s):
        if isinstance(s, list):
            s = torch.tensor(s)
        
        seq_shape = list(s.shape)
        seq_shape[0] = max_len
        
        padded_tensor = torch.zeros(seq_shape)
        
        padded_tensor[:s.shape[0]] = s
        
        return padded_tensor
    
    max_len = None
    for i, seq in enumerate(sequences):
        # get max sequence length
        m = max([len(s) for s in seq])
        if max_len != None:
            assert max_len == m
        else:
            max_len = m
        
        # pad to max sequence length
        padded_res = [(len(s), pad_indiv(s)) for s in seq]
        
        seq_lens, p_seq = zip(*padded_res)
        
        padded_sequences.append((torch.stack(p_seq, dim=0), torch.tensor(seq_lens)))

    return padded_sequences

def forward_pass(sentence_batch, predicates_batch, args_batch, lstm):
    (sentence_batch, token_lens), (predicates_batch, _), (args_batch, _) = pad(sentence_batch,
                                                                               predicates_batch,
                                                                               args_batch)
    
    sentence_batch = sentence_batch.permute(1, 0, 2)
    predicates_batch = predicates_batch.permute(1, 0)
    args_batch = args_batch.permute(1, 0)
    
    predicates_batch = predicates_batch.long()
    
    logits = lstm(sentence_batch.to('cpu'), predicates_batch.to('cpu'), token_lens)
    
    return logits, (sentence_batch, predicates_batch, args_batch, token_lens)

def make_mask(seq_length, token_lens):
    batch_size = len(token_lens)
    
    mask = torch.zeros((seq_length, batch_size), dtype=torch.long)
    
    for i, l in enumerate(token_lens):
        mask[:l, i] = 1
    
    return mask

def select_tokens(tokens, mask, mask_id, include_index=False):
    assert len(tokens) == len(mask)
    
    if include_index:
        result_tokens = [(t, i) for i, (t, m) in enumerate(zip(tokens, mask)) if m == mask_id]
    else:
        result_tokens = [t for t, m in zip(tokens, mask) if m == mask_id]
    
    return result_tokens

def select_batch(data, i_start, batch_size):
    return {
        k: v[i_start : i_start + batch_size] for k, v in data.items()
    }

def shuffle_data(data):
    data_len = len(data[next(iter(data.keys()))])
    
    rand_indices = torch.randperm(data_len)

    for k, v in data.items():
        if torch.is_tensor(v):
          data[k] = v[rand_indices]
        elif isinstance(v, list):
          data[k] = [v[i] for i in rand_indices]
        else:
          raise Exception('input not list or tensor')

from sklearn.metrics import accuracy_score, f1_score

def get_loss(logits, args_batch, token_lens):
    # loss calculation
    logits_flat = logits.reshape(-1, logits.shape[-1])
    args_batch_flat = args_batch.reshape(-1)
    
    loss = loss_func(logits_flat.cpu(), args_batch_flat.long()) # seq_length * batch_size, 1
    
    loss = loss.view(*args_batch.shape) # seq_length, batch_size
    
    loss *= make_mask(args_batch.shape[0], token_lens)
    
    # average across sequence
    loss = torch.sum(loss, dim=0) # batch_size
    loss = loss / token_lens
    
    # average across batch
    loss = torch.mean(loss)
    
    return loss

def subtoken_to_tokens(predictions, subtoken_spans):
    aligned_predictions = []

    #print(predictions.shape, sum(subtoken_spans))

    ptr = 0
    for size in subtoken_spans:
        group = predictions[ptr:ptr + size].tolist()

        aligned_predictions.append(max(set(group), key=group.count))
        ptr += size

    return aligned_predictions

def subtoken_to_tokens_batch(preds_max, subtoken_batch, token_lens):
    tokens_preds = []
    #print(preds_max.shape)
    for preds, subtoken_spans, l in zip(preds_max, subtoken_batch, token_lens):
        #print(len(preds))
        tokens_preds.append(subtoken_to_tokens(preds[:sum(subtoken_spans)], subtoken_spans))
    return tokens_preds

def get_metrics(logits, token_lens, sentence_raw_batch, args_batch, label_space):
    # metric calculation
    preds_max = torch.argmax(logits.cpu(), dim=-1)
    
    preds_max = preds_max.permute(1, 0)
    args_batch = args_batch.permute(1, 0)

    batch_size = len(preds_max)

    preds_all = []
    targets_all = []
    
    preds_bio = []
    targets_bio = []
    
    for j_batch in range(batch_size):
        seq_length = token_lens[j_batch]
        
        raw_input = sentence_raw_batch[j_batch]
        
        target_seq = args_batch[j_batch]
        pred_seq = preds_max[j_batch]
        
        targets_all.extend(target_seq)
        preds_all.extend(pred_seq)

        preds_bio.append([label_space[int(t)] for t in target_seq[:seq_length]])
        targets_bio.append([label_space[int(t)] for t in pred_seq[:seq_length]])
    
    acc = accuracy_score(targets_all, preds_all)
    
    return (preds_bio, targets_bio), acc
