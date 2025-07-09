from dataset import g_relational_concepts

def create_execution_existL(program, question_index, parent_img="img",property_prefix="prop", indent="\t"):

    values = [v for step in program for v in step.get("value_inputs", [])]
    
    define_character = ord("a")
    check_use = False
    check_relation = None

    preds = []
    for i, val in enumerate(values):
        suffix = ""
        if check_relation:
            suffix = ",obj2"
        if val in g_relational_concepts["spatial_relation"]:
            preds.append(f'is_{val}("{chr(define_character+1)}", path=("{chr(define_character)}", obj1.reversed))')
            define_character += 1
            check_use = True
            check_relation = True
        elif not check_use:
            preds.append(f'is_{val}("{chr(define_character)}"{suffix})')
            check_use = True
        else:
            preds.append(f'is_{val}(path=("{chr(define_character)}"{suffix}))')

    if len(preds)==0:
        constraint = (
                f'existsL(obj("prop"))\n\t'
        )
        return constraint
    
    def nest(pred_list, level):
        if len(pred_list) == 1:                         # base case
            return indent * level + pred_list[0]

        first, rest = pred_list[0], pred_list[1:]
        return (
            indent * level + "\n\t"
            + indent * (level + 1) + first + ",\n\t"
            + nest(rest, level + 1) + "\n\t"
            + indent * level + ""
        )

    constraint = (
        f'existsL(\n\t'
        + nest(preds, 2) + "\n\t"
        f'{indent})\n\t'
    )
    return constraint