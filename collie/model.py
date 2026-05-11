import torch
from tokens import TokenMap
from typing import Any, Literal, TYPE_CHECKING
from domiknows.generation import mask_logits_for_dfa

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
else:
    PreTrainedModel = Any
    PreTrainedTokenizer = Any


class TinyModel(torch.nn.Module):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            label_map: TokenMap,
            vocab: list[str],
            eos_idx: int = 50256,
            pad_size: int = 48,
            mode: Literal['tf', 'generate'] = 'generate',
            token_vocabulary=None,
            constrained_dfa=None,
        ):

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = vocab
        
        self.vocab_ids = [self.tokenizer.encode(v)[0] for v in self.vocab]

        self.mode = mode
        assert self.mode in {'tf', 'generate'}

        self.lmap = label_map

        self.eos_idx = eos_idx
        self.pad_size = pad_size
        self.token_vocabulary = token_vocabulary
        self.constrained_dfa = constrained_dfa

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
            dfa_state = self.constrained_dfa.start_state if self.constrained_dfa is not None else None

            for i in range(self.pad_size):
                logits = self.model(torch.tensor(input_ids).unsqueeze(0)).logits[0]
                target_logits = logits[-1, :]

                if self.constrained_dfa is not None:
                    if self.token_vocabulary is None:
                        raise ValueError("token_vocabulary is required for constrained decoding")
                    remaining_steps = self.pad_size - i
                    allowed = {
                        int(label)
                        for label in self.constrained_dfa.allowed_tokens(
                            dfa_state,
                            remaining_steps=remaining_steps,
                        )
                    }
                    # mask out disallowed tokens by setting their logits to a large negative value
                    target_logits = mask_logits_for_dfa(target_logits, allowed, self.token_vocabulary)

                target_logits_subset = target_logits[self.lmap.label_list]
                generated_logits.append(target_logits_subset.detach())

                # generated id is the argmax within the subset of the vocabulary
                if self.constrained_dfa is None:
                    next_id = self.lmap.inv_label_map[torch.argmax(target_logits_subset).item()]
                else:
                    next_id = int(torch.argmax(target_logits).item())
                    next_label = self.token_vocabulary.label_for_token_id(next_id)
                    dfa_state = self.constrained_dfa.step(dfa_state, next_label)
                    if dfa_state is None:
                        raise RuntimeError("constrained decoder selected a token with no DFA transition")
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
            
            gen_probs = torch.softmax(gen_logits, dim=-1)
            
            # condense token probs
            gen_probs_new = torch.zeros((gen_probs.shape[0], len(self.vocab) + 1))
            prob_sum = 0
            for i, token_id in enumerate(self.vocab_ids):
                next_prob = gen_probs[:, self.lmap.label_map[token_id]]
                gen_probs_new[:, i] = next_prob
                prob_sum += next_prob
            
            gen_probs_new[:, -1] = 1 - prob_sum

            gen_logprobs = torch.log(gen_probs_new.clamp_min(1e-12))

            return gen_logprobs
