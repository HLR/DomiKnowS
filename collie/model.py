import torch
from tokens import TokenMap
from transformers import PreTrainedModel, PreTrainedTokenizer


class TinyModel(torch.nn.Module):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            label_map: TokenMap
        ):

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.mode = 'generate'
        assert self.mode in {'tf', 'generate'}

        self.lmap = label_map

    def forward(
            self,
            _,
            input_ids: torch.Tensor,
            target_tokens: torch.Tensor,
            test_tokens: torch.Tensor
        ) -> torch.Tensor:

        if self.mode == 'tf':
            input_vals = torch.cat([input_ids[0], target_tokens], dim=0).unsqueeze(0)

            logits = self.model(input_vals).logits[0]

            start_pos = input_ids.shape[1] - 1
            target_logits = logits[start_pos : start_pos + target_tokens.shape[0], :]

            target_logits_subset = target_logits[:, self.lmap.label_list]

            return target_logits_subset
        
        elif self.mode == 'generate':
            input_ids = input_ids[0].tolist()
            generated_logits = []

            for _ in range(len(target_tokens)):
                logits = self.model(torch.tensor(input_ids).unsqueeze(0)).logits[0]
                target_logits = logits[-1, :]

                target_logits_subset = target_logits[self.lmap.label_list]
                generated_logits.append(target_logits_subset.detach())

                input_ids.append(self.lmap.inv_label_map[torch.argmax(target_logits_subset).item()])

            return torch.stack(generated_logits)
