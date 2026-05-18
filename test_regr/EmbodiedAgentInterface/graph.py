from dataset import ACTION_VOCAB, EOS_TOKEN


def create_generation_graph(max_steps=8, required_tokens=None, forbidden_tokens=None):
    from domiknows.generation import GenerationEncoder, default_generation_constraints

    constraints = default_generation_constraints(
        max_non_eos_count=max_steps - 1,
        required_tokens=required_tokens or {},
        forbidden_tokens=forbidden_tokens or [],
    )
    encoder = GenerationEncoder(
        ACTION_VOCAB,
        eos_token=EOS_TOKEN,
        graph_name="eai_generation_graph",
    )
    return encoder.build_graph(constraints)


# Backward-compatible alias for older imports in this folder.
def create_graph(max_steps=8):
    graph, bundle = create_generation_graph(max_steps=max_steps)
    return graph, bundle
