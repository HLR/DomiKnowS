# Application Adapters and Controllers

`domiknows.generation.applications` contains application-level orchestration
built on top of the lower-level DFA, latent, and learner utilities.

This package is where backend generation calls, constraint verification,
reranking, and planning-domain graph adaptation are wired into practical flows.

## Modules

| Module | Purpose |
| --- | --- |
| `adapters.py` | Backend adapters with one unified result shape (`GenerationResult`) for HuggingFace and OpenAI-style generation paths. |
| `hybrid.py` | Hybrid candidate generation, verification, and reranking (`HybridController`, score dataclasses, and constraint-bundle selectors). |
| `planning.py` | Graph-to-planning adapters that derive planning bundles, compile hard DFAs, and build HMM masks from declarative graph schemas. |
| `__init__.py` | Public re-exports for the package API. |

## What This Layer Does

- abstracts backend-specific generation APIs behind common adapter interfaces;
- supports constrained decoding for compatible backends;
- verifies outputs against DFAs and explains rejection causes when requested;
- ranks candidates with compact-head likelihood, hard validity, latent
  preference terms, and risk penalties;
- converts declarative planning graphs into executable artifacts used by demos
  and tests.

## Quick Import Example

```python
from domiknows.generation.applications import (
    GenerationResult,
    HuggingFaceGenerationAdapter,
    HybridController,
    PlanningBundle,
    planning_bundle_from_graph,
    planning_dfa_from_graph,
)
```

## Typical Flow

1. create or load a graph and compile/discover constraints into a DFA;
2. generate candidates through an adapter (`HuggingFaceGenerationAdapter` or
   `OpenAIResponsesAdapter`);
3. verify each candidate against the DFA;
4. rerank accepted candidates using `HybridController` scoring;
5. for planning tasks, derive a `PlanningBundle` and planning DFA directly from
   graph declarations.

## Notes

- hard token-level constrained decoding is available for backends that expose
  token logits during generation;
- OpenAI Responses integration is generate-then-verify;
- planning utilities are domain-generic and rely on graph schema naming rather
  than a specific task implementation.
