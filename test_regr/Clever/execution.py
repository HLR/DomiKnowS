

def create_execution_existL(program, question_index, parent_img="img",property_prefix="prop", indent="\t"):

    values = [v for step in program for v in step.get("value_inputs", [])]

    preds = [
        f'is_{val}("{property_prefix}{i}", '
        f'path=("{parent_img}", image_object_contains))'
        for i, val in enumerate(values)
    ]

    def nest(pred_list, level):
        if len(pred_list) == 1:                         # base case
            return indent * level + pred_list[0]

        first, rest = pred_list[0], pred_list[1:]
        return (
            indent * level + "andL(\n\t"
            + indent * (level + 1) + first + ",\n\t"
            + nest(rest, level + 1) + "\n\t"
            + indent * level + ")"
        )

    constraint = (
        f'ifL(image("{parent_img}"),\n'
        f'\t{indent}existsL(\n\t'
        + nest(preds, 2) + "\n\t"
        f'{indent})\n\t'
        f')'
    )
    return constraint