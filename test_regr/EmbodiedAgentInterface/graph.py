from dataset import ACTION_VOCAB, EOS_TOKEN


def create_generation_graph(max_steps=8, required_tokens=None, forbidden_tokens=None):
    from domiknows.generation import GenerationEncoder, apply_all_constraints

    encoder = GenerationEncoder(
        ACTION_VOCAB,
        eos_token=EOS_TOKEN,
        graph_name="eai_generation_graph",
    )
    graph, bundle = encoder.build_graph()
    with graph:
        apply_all_constraints(
            bundle.context,
            max_non_eos_count=max_steps - 1,
            required_tokens=required_tokens or {},
            forbidden_tokens=forbidden_tokens or [],
        )
    return graph, bundle


# Backward-compatible alias for older imports in this folder.
def create_graph(max_steps=8):
    graph, bundle = create_generation_graph(max_steps=max_steps)
    return graph, bundle
