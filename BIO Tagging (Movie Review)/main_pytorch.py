from reader import parse_and_return_dataset
from transformers import T5ForConditionalGeneration, AdamW
from transformers import T5Tokenizer
import torch
from torch.nn import CrossEntropyLoss
from model import PytorchFilteredT5Model
model_name='t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
new_tokens = ['<B>', '<I>', '<O>']
tokenizer.add_tokens(new_tokens)
train_inputs, train_targets, val_inputs, val_targets=parse_and_return_dataset(tokenizer)

device='cuda:1'
model=PytorchFilteredT5Model(model_name, tokenizer)
optimizer = AdamW(model.parameters(), lr=5e-4)
loss_fct = CrossEntropyLoss(ignore_index=-100)

epochs = 100
model.to(device)
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for batch_idx in range(len(train_inputs)):
        inputs = train_inputs[batch_idx].to(device)
        labels = train_targets[batch_idx].squeeze(-1).to(device)  

        outputs = model(input_ids=inputs['input_ids'].to(device), 
                        attention_mask=inputs['attention_mask'].to(device),
                        labels=labels[:,:-1])
        
        loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_inputs)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")
    if epoch%10==9:
        model.eval()
        with torch.no_grad():
            for batch_idx in range(2):
                inputs = val_inputs[batch_idx].to(device)
                labels = val_targets[batch_idx].squeeze(-1).to(device)  # Assuming labels are already tensors

                outputs = model.generate(input_ids=inputs['input_ids'].to(device), 
                                        attention_mask=inputs['attention_mask'].to(device),
                                        max_length=labels.shape[1])

                pred_tags = outputs
                actual_tags = labels
                print(f"Predicted Tags: {pred_tags}")
                print(f"Actual Tags: {actual_tags}\n")
