# type: ignore

with Graph(name='global') as graph:
    image = Concept(name='image')

    # 1) Digit concept
    #   Previous: specify concept with pre-specified categorical values
    # digit = image(
    #     name='digits',
    #     ConceptClass=EnumConcept,
    #     values=digits
    # )

    # Now: we specify a numerical concept
    #   numerical concept indicates:
    #       (1) how we should treat it in the graph (we can't directly specify logical
    #           constraints over it now)
    #       (2) how we should treat sensor values assigned to it (previously, we
    #           expect vectors of probabilities; now we expect regression outputs)
    digit = image(
        name='digits',
        ConceptClass=NumericalConcept,
    )

    # 1.5) Specifying image pair (unchanged from before)
    image_pair = Concept(name='pair')
    pair_d0, pair_d1 = image_pair.has_a(digit0=image, digit1=image)

    # 2) Performing summation
    #   summation is a Concept
    summation = ifP(
        # `Condition` must be a binary Concept (i.e., not NumericalConcept)
        #   OR a logical expression (e.g., andL)

        # Interfacing between numerical and categorical (regular) Concepts
        # Using numerical concepts with >, <, ==, etc. gives us regular binary concepts
        #   that we can e.g., use to specify logical constraints or use as conditions
        condition = andL(
            # NumericalConcept % 2 -> NumericalConcept
            # (NumericalConcept == 0) -> regular Concept
            digit(path = (pair_d0, 'x')) % 2 == 0,
            digit(path = (pair_d1, 'y')) % 2 == 0
        ),

        # NumericalConcept {+, -, /, *} NumericalConcept -> NumericalConcept
        then_ret = (digit(path=(pair_d0, 'x')) + digit(path=(pair_d1, 'y'))),

        else_ret = (digit(path=(pair_d0, 'x')) * digit(path=(pair_d1, 'y')))
    )
    # aside: maybe we should be treating the outputs of logical expressions
    #   as concepts as well?

# 1) Load data (unchanged)
image['pixels'] = ReaderSensor(keyword='pixels')

image_pair[pair_d0.reversed, pair_d1.reversed] = JointSensor(
    image['pixels'],
    forward=make_pairs
)

# 2) image[digit] now expects numerical values from Net.forward(...)
#   during inference, the +, -, /, *, % operations from above will be
#   applied to these numeric values
image[digit] = ModuleLearner('pixels', module=Net())

# 3) We retrieve the summation label from the ReaderSensor
#   Also, similar to how we specify program inference outputs for binary values
#   we store the labels in the graph.constraint Concept
graph.constraint[summation] = ReaderSensor(keyword='summation', label=True)
