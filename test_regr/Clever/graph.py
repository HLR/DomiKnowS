try:
    from .dataset import g_attribute_concepts, g_relational_concepts
    from .execution import create_execution_for_question
except ImportError:
    from dataset import g_attribute_concepts, g_relational_concepts
    from execution import create_execution_for_question

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.visual.visual_constraints import (
    apply_opposite_constraints,
    apply_inverse_constraints,
)


# Map CLEVR relation names to visual_constraints ctx keys
_CLEVR_TO_CTX_FWD = {
    "left": "left_of",
    "right": "right_of",
    "front": "in_front_of",
    "behind": "behind",
}

_CLEVR_TO_CTX_REV = {
    "left_rev": "left_of_rev",
    "right_rev": "right_of_rev",
    "front_rev": "in_front_of_rev",
    "behind_rev": "behind_rev",
}


def create_graph(
    dataset,
    return_graph_text=False,
    include_query_questions=False,
    apply_constraints=True,
    relation_syntax="legacy",
):
    """
    Create the DomiKnows graph for CLEVR dataset.

    Builds the graph programmatically (no exec/eval) and applies
    spatial constraints via the visual_constraints library.

    Args:
        dataset: The CLEVR dataset (list of dicts).
        return_graph_text: If True, also return a text representation.
        include_query_questions: Whether to include parent attribute concepts for queryL.
        apply_constraints: Whether to apply visual reasoning constraints.
        relation_syntax: Relation translation mode ('legacy' or 'binary').
    """
    # ------------------------------------------------------------------
    # 0.  Clear any previous graph state
    # ------------------------------------------------------------------
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("image_graph") as graph:

        # --------------------------------------------------------------
        # 1.  Image / Object
        # --------------------------------------------------------------
        image = Concept(name="image")
        obj = Concept(name="obj")
        (image_object_contains,) = image.contains(obj)

        # --------------------------------------------------------------
        # 2.  Attribute concepts
        # --------------------------------------------------------------
        attribute_names_dict = {}

        if include_query_questions:
            # Create parent attribute concepts and their subclasses for queryL
            for attr, values in g_attribute_concepts.items():
                parent = obj(name=attr)
                attribute_names_dict[attr] = parent
                for val in values:
                    child = parent(name=val)
                    attribute_names_dict[val] = child
        else:
            # Flat attribute concepts
            for attr, values in g_attribute_concepts.items():
                for val in values:
                    concept = obj(name=val)
                    attribute_names_dict[val] = concept

        # --------------------------------------------------------------
        # 3.  Forward pair node  (pair_forward)
        # --------------------------------------------------------------
        pair_forward = Concept("pair_forward")
        (obj1, obj2) = pair_forward.has_a(arg1=obj, arg2=obj)

        spatial_fwd = {}
        for val in g_relational_concepts.get("spatial_relation", []):
            c = pair_forward(name=val)
            attribute_names_dict[val] = c
            spatial_fwd[val] = c

        # --------------------------------------------------------------
        # 4.  Reverse pair node  (pair_reverse)
        # --------------------------------------------------------------
        pair_reverse = Concept("pair_reverse")
        (obj1_rev, obj2_rev) = pair_reverse.has_a(arg1=obj, arg2=obj)

        spatial_rev = {}
        for val in g_relational_concepts.get("spatial_relation", []):
            rev_name = f"{val}_rev"
            c = pair_reverse(name=rev_name)
            attribute_names_dict[rev_name] = c
            spatial_rev[rev_name] = c

        # --------------------------------------------------------------
        # 5.  Same-attribute relations (on pair_forward) for query mode
        # --------------------------------------------------------------
        if include_query_questions:
            for attr in ["size", "color", "material", "shape"]:
                key = f"same_{attr}"
                c = pair_forward(name=key)
                attribute_names_dict[key] = c

        # --------------------------------------------------------------
        # 6.  Constraints via visual_constraints library
        #
        # §0 opposite:   nandL(left, right) — mutual exclusion, allows
        #                both to be false (non-exhaustive)
        # §1 inverse:    equivalenceL(left_fwd, right_rev) — true
        #                semantic identity across pair nodes
        # --------------------------------------------------------------
        if apply_constraints:
            ctx = _build_constraint_context(
                graph=graph,
                image=image,
                obj=obj,
                image_object_contains=image_object_contains,
                pair_forward=pair_forward,
                obj1=obj1,
                obj2=obj2,
                pair_reverse=pair_reverse,
                obj1_rev=obj1_rev,
                obj2_rev=obj2_rev,
                spatial_fwd=spatial_fwd,
                spatial_rev=spatial_rev,
                attribute_names_dict=attribute_names_dict,
            )
            apply_opposite_constraints(ctx)
            if relation_syntax == "binary":
                apply_inverse_constraints(ctx)

    # ------------------------------------------------------------------
    # 6b. Register all concept variable names in graph.varNameReversedMap
    #     so that compile_executable can resolve names in constraint strings.
    #     Graph.__exit__ only captures named local variables from the caller's
    #     frame; concepts built in loops/dicts are missed.
    # ------------------------------------------------------------------
    _register_concepts_in_graph(
        graph,
        image=image,
        obj=obj,
        image_object_contains=image_object_contains,
        pair_forward=pair_forward,
        obj1=obj1,
        obj2=obj2,
        pair_reverse=pair_reverse,
        obj1_rev=obj1_rev,
        obj2_rev=obj2_rev,
        attribute_names_dict=attribute_names_dict,
    )

    # ------------------------------------------------------------------
    # 7.  Compile executable constraints per question
    # ------------------------------------------------------------------
    executions = []
    query_types = []

    for i in range(len(dataset)):
        current_instance = dataset[i]
        program = current_instance.get("program", [])
        question_raw = current_instance.get("question_raw", "")

        execution, query_type = create_execution_for_question(
            program,
            i,
            relation_syntax=relation_syntax,
        )

        if " or " in question_raw:
            print(f"Found 'or' in question {i}")

        executions.append(execution)
        query_types.append(query_type)

    # ------------------------------------------------------------------
    # 8.  Build graph text for debugging / inspection
    # ------------------------------------------------------------------
    graph_text = _generate_graph_text(include_query_questions) if return_graph_text else None

    # ------------------------------------------------------------------
    # 9.  Return
    # ------------------------------------------------------------------
    if return_graph_text:
        return (
            executions,
            graph,
            image,
            obj,
            image_object_contains,
            obj1,
            obj2,
            pair_forward,
            attribute_names_dict,
            graph_text,
            query_types,
            obj1_rev,
            obj2_rev,
            pair_reverse,
        )

    return (
        executions,
        graph,
        image,
        obj,
        image_object_contains,
        obj1,
        obj2,
        pair_forward,
        attribute_names_dict,
        query_types,
        obj1_rev,
        obj2_rev,
        pair_reverse,
    )


def _register_concepts_in_graph(
    graph,
    *,
    image,
    obj,
    image_object_contains,
    pair_forward,
    obj1,
    obj2,
    pair_reverse,
    obj1_rev,
    obj2_rev,
    attribute_names_dict,
):
    """
    Manually populate graph.varNameReversedMap with every concept so that
    compile_executable() can resolve names in constraint strings.

    Graph.__exit__ captures variable names via inspect of the caller's locals,
    but concepts created in loops or dicts are not named locals and get missed.
    """
    var_map = graph.varNameReversedMap

    # Structural concepts
    var_map["image"] = image
    var_map["obj"] = obj
    var_map["image_object_contains"] = image_object_contains
    var_map["pair_forward"] = pair_forward
    var_map["obj1"] = obj1
    var_map["obj2"] = obj2
    var_map["pair_reverse"] = pair_reverse
    var_map["obj1_rev"] = obj1_rev
    var_map["obj2_rev"] = obj2_rev

    # .reversed relations (needed by legacy constraint syntax)
    if hasattr(obj1, "reversed"):
        var_map["obj1.reversed"] = obj1.reversed
    if hasattr(obj2, "reversed"):
        var_map["obj2.reversed"] = obj2.reversed
    if hasattr(obj1_rev, "reversed"):
        var_map["obj1_rev.reversed"] = obj1_rev.reversed
    if hasattr(obj2_rev, "reversed"):
        var_map["obj2_rev.reversed"] = obj2_rev.reversed

    # All attribute and relation concepts — keyed by their dict key
    # which matches the name used in constraint expressions
    for name, concept in attribute_names_dict.items():
        var_map[name] = concept


def _build_constraint_context(
    *,
    graph,
    image,
    obj,
    image_object_contains,
    pair_forward,
    obj1,
    obj2,
    pair_reverse,
    obj1_rev,
    obj2_rev,
    spatial_fwd,
    spatial_rev,
    attribute_names_dict,
):
    """
    Build a context dict compatible with the visual_constraints library
    (apply_opposite_constraints, apply_inverse_constraints, etc.).

    """
    # Extract per-category dicts for completeness
    colors_dict = {}
    shapes_dict = {}
    sizes_dict = {}
    for attr, values in g_attribute_concepts.items():
        for val in values:
            c = attribute_names_dict.get(val)
            if c is None:
                continue
            if attr == "color":
                colors_dict[val] = c
            elif attr == "shape":
                shapes_dict[val] = c
            elif attr == "size":
                sizes_dict[val] = c

    ctx = {
        "graph": graph,
        "image": image,
        "object": obj,
        "image_contains_object": image_object_contains,
        # forward pair
        "pair_forward": pair_forward,
        "rel_arg1_fwd": obj1,
        "rel_arg2_fwd": obj2,
        # reverse pair
        "pair_reverse": pair_reverse,
        "rel_arg1_rev": obj1_rev,
        "rel_arg2_rev": obj2_rev,
        # attributes
        "colors": colors_dict,
        "shapes": shapes_dict,
        "sizes": sizes_dict,
        "material": attribute_names_dict.get("material"),
    }

    # Map CLEVR forward spatial names → visual_constraints expected keys
    for clevr_name, ctx_key in _CLEVR_TO_CTX_FWD.items():
        ctx[ctx_key] = spatial_fwd.get(clevr_name)

    # Map CLEVR reverse spatial names → visual_constraints expected keys
    for clevr_name, ctx_key in _CLEVR_TO_CTX_REV.items():
        ctx[ctx_key] = spatial_rev.get(clevr_name)

    return ctx


def _generate_graph_text(include_query_questions):
    """Generate a textual representation of the graph for debugging."""
    lines = [
        "from domiknows.graph import Graph, Concept",
        "from domiknows.graph.logicalConstrain import ifL, andL, existsL, iotaL, queryL",
        "",
        "with Graph('image_graph') as graph:",
        "",
        "\timage = Concept(name='image')",
        "\tobj = Concept(name='obj')",
        "\timage_object_contains, = image.contains(obj)",
        "",
    ]

    if include_query_questions:
        for attr, values in g_attribute_concepts.items():
            lines.append(f"\t{attr} = obj(name='{attr}')")
            for val in values:
                lines.append(f"\t{val} = {attr}(name='{val}')")
            lines.append("")
    else:
        for attr, values in g_attribute_concepts.items():
            for val in values:
                lines.append(f"\t{val} = obj(name='{val}')")
            lines.append("")

    for attr, values in g_relational_concepts.items():
        lines.append("\tpair_forward = Concept('pair_forward')")
        lines.append("\t(obj1, obj2) = pair_forward.has_a(arg1=obj, arg2=obj)")
        for val in values:
            lines.append(f"\t{val} = pair_forward(name='{val}')")
        lines.append("\tpair_reverse = Concept('pair_reverse')")
        lines.append("\t(obj1_rev, obj2_rev) = pair_reverse.has_a(arg1=obj, arg2=obj)")
        for val in values:
            lines.append(f"\t{val}_rev = pair_reverse(name='{val}_rev')")
        lines.append("")

    if include_query_questions:
        for a in ["size", "color", "material", "shape"]:
            lines.append(f"\tsame_{a} = pair_forward(name='same_{a}')")
        lines.append("")

    return "\n".join(lines)


def _demo_program():
    """A small CLEVR-style program for smoke-testing the graph."""
    return [
        {"inputs": [], "function": "scene", "value_inputs": []},
        {"inputs": [0], "function": "filter_shape", "value_inputs": ["cube"]},
        {"inputs": [1], "function": "filter_color", "value_inputs": ["red"]},
        {"inputs": [2], "function": "unique", "value_inputs": []},
        {"inputs": [3], "function": "relate", "value_inputs": ["left"]},
        {"inputs": [4], "function": "filter_size", "value_inputs": ["large"]},
        {"inputs": [5], "function": "exist", "value_inputs": []},
    ]


if __name__ == "__main__":
    import json, sys, argparse, traceback

    print("[graph.py] Starting debug runner...")
    sys.stdout.flush()

    parser = argparse.ArgumentParser(description="Debug / inspect the CLEVR DomiKnows graph")
    parser.add_argument("--no-constraints", action="store_true",
                        help="Skip applying visual constraints")
    parser.add_argument("--include-query", action="store_true",
                        help="Include parent attribute concepts for queryL")
    parser.add_argument("--relation-syntax", choices=["legacy", "binary"], default="legacy")
    parser.add_argument("--program-json", type=str, default=None,
                        help="Path to a JSON file with a list of CLEVR programs")
    parser.add_argument("--questions-json", type=str, default=None,
                        help="Path to a JSON file with question/answer pairs (no programs); "
                             "prints questions only, no constraint compilation")
    parser.add_argument("--visualize", type=str, default=None,
                        help="If set, save graph visualization to this filename (requires graphviz)")
    args = parser.parse_args()

    # Build a tiny dataset from either a file or the built-in demo program
    try:
        if args.questions_json:
            # --questions-json: load question/answer pairs (no CLEVR programs)
            # Just print questions — no constraint compilation possible
            with open(args.questions_json, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                entries = raw.get("questions", raw.get("data", []))
            elif isinstance(raw, list):
                entries = raw
            else:
                entries = []

            print(f"[graph.py] Loaded {len(entries)} questions from {args.questions_json}")
            print()
            print("=" * 60)
            print("QUESTIONS (no programs — constraints cannot be compiled)")
            print("=" * 60)
            for i, entry in enumerate(entries):
                q = entry.get("question", str(entry)) if isinstance(entry, dict) else str(entry)
                a = entry.get("answer", "?") if isinstance(entry, dict) else "?"
                print(f"  [{i}] Q: {q}")
                print(f"       A: {a}")
                print()
            print("[graph.py] Done (questions-only mode).")
            sys.exit(0)

        if args.program_json:
            with open(args.program_json, "r") as f:
                programs = json.load(f)
            if isinstance(programs, dict) and "questions" in programs:
                programs = programs["questions"]
            dataset = []
            for i, entry in enumerate(programs):
                prog = entry if isinstance(entry, list) else entry.get("program", [])
                dataset.append({
                    "program": prog,
                    "question_raw": entry.get("question", f"<question {i}>"),
                    "answer": entry.get("answer", "yes"),
                })
        else:
            dataset = [
                {"program": _demo_program(), "question_raw": "Is there a large thing left of the red cube?", "answer": True},
            ]

        print(f"[graph.py] Dataset size: {len(dataset)}")
        sys.stdout.flush()

        print("[graph.py] Building graph...")
        sys.stdout.flush()

        results = create_graph(
            dataset,
            return_graph_text=True,
            include_query_questions=args.include_query,
            apply_constraints=not args.no_constraints,
            relation_syntax=args.relation_syntax,
        )

        executions = results[0]
        graph = results[1]
        attribute_names_dict = results[8]
        graph_text = results[9]
        query_types = results[10]
        obj1_rev = results[11]
        obj2_rev = results[12]
        pair_reverse = results[13]

        # --- Graph text ---
        print()
        print("=" * 60)
        print("GRAPH TEXT")
        print("=" * 60)
        print(graph_text)
        print()

        # --- Concepts ---
        print("=" * 60)
        print("ATTRIBUTE / RELATION CONCEPTS")
        print("=" * 60)
        for name, concept in attribute_names_dict.items():
            print(f"  {name:20s}  ->  {concept.name}  (type: {type(concept).__name__})")
        print()

        # --- Reverse pair info ---
        print("=" * 60)
        print("PAIR NODES")
        print("=" * 60)
        print(f"  pair_forward:  {results[7]}")
        print(f"  obj1:          {results[5]}")
        print(f"  obj2:          {results[6]}")
        print(f"  pair_reverse:  {pair_reverse}")
        print(f"  obj1_rev:      {obj1_rev}")
        print(f"  obj2_rev:      {obj2_rev}")
        print()

        # --- Constraints ---
        print("=" * 60)
        print("LOGICAL CONSTRAINTS ON GRAPH")
        print("=" * 60)
        lcs = getattr(graph, "logicalConstrains", {})
        if lcs:
            for lc_name, lc in lcs.items():
                active = getattr(lc, "active", "?")
                lc_type = type(lc).__name__
                print(f"  {lc_name:40s}  type={lc_type:20s}  active={active}")
        else:
            print("  (none)")
        print()

        # --- Graph concepts registered ---
        print("=" * 60)
        print("GRAPH varNameReversedMap (concept variable names)")
        print("=" * 60)
        var_map = getattr(graph, "varNameReversedMap", {})
        if var_map:
            for var_name, concept_obj in sorted(var_map.items()):
                c_name = getattr(concept_obj, "name", str(concept_obj))
                print(f"  {var_name:25s}  ->  {c_name}")
        else:
            print("  (empty — variable names not captured)")
        print()

        # --- Executions ---
        print("=" * 60)
        print("COMPILED EXECUTIONS (per question)")
        print("=" * 60)
        for i, (exc, qt) in enumerate(zip(executions, query_types)):
            q = dataset[i].get("question_raw", "")
            a = dataset[i].get("answer", "")
            print(f"  [{i}] Q: {q}")
            print(f"       A: {a}   query_type={qt}")
            print(f"       Exec: {exc}")
            print()

        # --- Optional visualization ---
        if args.visualize:
            try:
                graph.visualize(args.visualize, open_image=False)
                print(f"Graph visualization saved to {args.visualize}.png")
            except Exception as e:
                print(f"Visualization failed: {e}")

        print("[graph.py] Done.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


def create_graph_for_query_questions(
    dataset,
    return_graph_text=False,
    apply_constraints=True,
    relation_syntax="legacy",
):
    """
    Specialized graph creation for query-type questions only.
    Uses queryL with iotaL for selecting unique objects and querying attributes.
    """
    return create_graph(
        dataset,
        return_graph_text=return_graph_text,
        include_query_questions=True,
        apply_constraints=apply_constraints,
        relation_syntax=relation_syntax,
    )