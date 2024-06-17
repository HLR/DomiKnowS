import torch
from tokens import TokenMap
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Literal


class TinyModel(torch.nn.Module):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            label_map: TokenMap,
            eos_idx: int = 50256,
            pad_size: int = 48,
            mode: Literal['tf', 'generate'] = 'generate'
        ):

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.mode = mode
        assert self.mode in {'tf', 'generate'}

        self.lmap = label_map

        self.eos_idx = eos_idx
        self.pad_size = pad_size

    def forward(
            self,
            _,
            input_ids: torch.Tensor,
            target_tokens: torch.Tensor,
            test_tokens: torch.Tensor # for debugging only
        ) -> tuple[torch.Tensor, torch.Tensor]:

        assert target_tokens.shape[0] == self.pad_size, 'ground truth tokens must have size (pad_size,)'

        if self.mode == 'tf':
            input_vals = torch.cat([input_ids[0], target_tokens.long()], dim=0).unsqueeze(0)

            logits = self.model(input_vals).logits[0]

            start_pos = input_ids.shape[1] - 1
            target_logits = logits[start_pos : start_pos + target_tokens.shape[0], :]

            target_logits_subset = target_logits[:, self.lmap.label_list]

            return target_logits_subset
        
        elif self.mode == 'generate':
            input_ids = input_ids[0].tolist()
            generated_logits = []

            for i in range(self.pad_size):
                logits = self.model(torch.tensor(input_ids).unsqueeze(0)).logits[0]
                target_logits = logits[-1, :]

                target_logits_subset = target_logits[self.lmap.label_list]
                generated_logits.append(target_logits_subset.detach())

                # generated id is the argmax within the subset of the vocabulary
                next_id = self.lmap.inv_label_map[torch.argmax(target_logits_subset).item()]
                input_ids.append(next_id)

                # if next_id == self.eos_idx:
                #     print('Model: hit EOS, breaking')
                #     eos_pos = i
                #     break

            gen_logits = torch.stack(generated_logits)
            gen_ids = torch.argmax(gen_logits, dim=-1)

            # pad to pad_size
            if gen_logits.shape[0] < self.pad_size:
                eos_oh = torch.ones((self.pad_size - gen_logits.shape[0], gen_logits.shape[1])) * -100
                eos_oh[:, self.lmap.label_map[self.eos_idx]] = 100

                gen_logits = torch.cat([
                    gen_logits,
                    eos_oh
                ], dim=0)
                
                gen_ids = torch.cat([
                    gen_ids,
                    torch.ones((self.pad_size - gen_ids.shape[0],), dtype=torch.long) * self.lmap.label_map[self.eos_idx]
                ], dim=0)

            return gen_logits
