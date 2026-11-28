"""Mirrored ES reward estimates for the source-only Joint Stage 2 pilot."""

from dataclasses import dataclass
import math

import torch

from .protocol import digest


def _seed(*values):
    return int(digest(values)[:15], 16)


def _noise(parameters, seed):
    cpu = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in parameters:
        yield torch.randn(parameter.shape, dtype=torch.float32, device="cpu", generator=cpu).to(parameter.device)


@dataclass(frozen=True)
class Population:
    noise_seeds: tuple[int, ...]
    sampling_seeds: tuple[int, ...]

    @classmethod
    def from_budget(cls, draws, *seed_parts):
        if draws < 2 or draws % 2:
            raise ValueError("mirrored ES requires a positive even rollout budget of at least two")
        return cls(tuple(_seed(*seed_parts, "noise", pair) for pair in range(draws // 2)),
                   tuple(_seed(*seed_parts, "sample", pair) for pair in range(draws // 2)))

    @torch.no_grad()
    def collect(self, parameters, sigma, draw):
        """Sample both signs of each direction from the same center and RNG state."""
        if not parameters or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("ES requires active parameters and positive finite noise")
        center = [parameter.detach().clone() for parameter in parameters]
        devices = sorted({parameter.device.index for parameter in parameters if parameter.is_cuda})
        variants = []
        for pair, (noise_seed, sample_seed) in enumerate(zip(self.noise_seeds, self.sampling_seeds, strict=True)):
            for sign_index, sign in enumerate((1, -1)):
                try:
                    for parameter, base, noise in zip(parameters, center, _noise(parameters, noise_seed), strict=True):
                        parameter.copy_(base + sign * sigma * noise)
                    with torch.random.fork_rng(devices=devices):
                        torch.random.default_generator.manual_seed(sample_seed)
                        for device in devices:
                            torch.cuda.default_generators[device].manual_seed(sample_seed)
                        variants.append(draw(2 * pair + sign_index))
                finally:
                    for parameter, base in zip(parameters, center, strict=True):
                        parameter.copy_(base)
        return variants

    def reward_differences(self, records):
        cases = {}
        for row in records:
            case = cases.setdefault(row["episode_id"], {})
            candidate = row["es_candidate"]
            if candidate in case:
                raise ValueError("duplicate ES candidate for one reset")
            case[candidate] = row
        expected = set(range(2 * len(self.noise_seeds)))
        fitness = [0] * len(expected)
        if not cases:
            raise ValueError("ES requires at least one scored reset")
        for case in cases.values():
            if set(case) != expected or len({(r["input_hash"], r.get("policy_hash")) for r in case.values()}) != 1:
                raise ValueError("ES candidates must cover identical reset inputs and constraints")
            for candidate, row in case.items():
                fitness[candidate] += int(row["success"])
        counts = [positive - negative for positive, negative in zip(fitness[::2], fitness[1::2], strict=True)]
        return {"episodes_per_candidate": len(cases), "candidate_successes": fitness,
                "pair_success_differences": counts, "informative_pairs": sum(value != 0 for value in counts),
                "noise_seeds": self.noise_seeds, "sampling_seeds": self.sampling_seeds}

    def capture_reward(self, manager, domain, parameters, sigma, metrics):
        """Inject the signed reward estimate at the restored center via a linear loss."""
        scale = -1 / (2 * sigma * metrics["episodes_per_candidate"] * len(self.noise_seeds))
        loss = None
        for seed, difference in zip(self.noise_seeds, metrics["pair_success_differences"], strict=True):
            if not difference:
                continue
            for parameter, noise in zip(parameters, _noise(parameters, seed), strict=True):
                contribution = (parameter * noise).sum() * (scale * difference)
                loss = contribution if loss is None else loss + contribution
        if loss is not None:
            manager.capture(domain, loss, parameters=parameters)
