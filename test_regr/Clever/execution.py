from convert_CLEVR_domiKnowS import translate_left_domiknows
try:
    from .dataset import g_relational_concepts
except ImportError:
    from dataset import g_relational_concepts

def create_execution_existL(program, question_index, parent_img="img", property_prefix="prop", indent="\t"):
    """Original existL execution creator for existence/count questions."""
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
            preds.append(f'{val}("{chr(define_character+1)}", path=("{chr(define_character)}", obj1.reversed))')
            define_character += 1
            check_use = True
            check_relation = True
        elif not check_use:
            preds.append(f'{val}("{chr(define_character)}"{suffix})')
            check_use = True
        else:
            preds.append(f'{val}(path=("{chr(define_character)}"{suffix}))')

    if len(preds) == 0:
        constraint = f'existsL(obj("prop"))\n\t'
        return constraint
    
    def flat_andL(pred_list, base_indent_level):
        """Create a flat andL with all predicates as arguments."""
        if len(pred_list) == 1:
            return indent * base_indent_level + pred_list[0]
        
        lines = [indent * base_indent_level + "andL("]
        for i, pred in enumerate(pred_list):
            comma = "," if i < len(pred_list) - 1 else ""
            lines.append(indent * (base_indent_level + 1) + pred + comma)
        lines.append(indent * base_indent_level + ")")
        return "\n".join(lines)

    constraint = (
        f'existsL(\n'
        + flat_andL(preds, 1) + "\n"
        f')\n'
    )
    return constraint


def create_execution_queryL(program, question_index, parent_img="img", property_prefix="prop", indent="\t"):
    """
    Create queryL execution for query-type questions.
    
    queryL wraps iotaL to query a multiclass attribute from the selected entity.
    
    For a question like "What material is the big object that is right of the brown cylinder 
    and left of the large brown sphere?":
    
    queryL(
        material,
        iotaL(
            andL(
                big('z'),
                right_of('r1', path=('z', rel_arg1.reversed)),
                brown('x'),
                cylinder(path='x'),
                left_of('r2', path=('z', rel_arg1.reversed)),
                large('y'),
                brown(path='y'),
                sphere(path='y')
            )
        )
    )
    """
    query_type = None
    
    parsed_steps = []
    for step in program:
        func = step.get('type', step.get('function', ''))
        inputs = step.get('inputs', [])
        value_inputs = step.get('value_inputs', [])
        parsed_steps.append({
            'function': func,
            'inputs': inputs,
            'value_inputs': value_inputs,
            'index': len(parsed_steps)
        })
    
    if parsed_steps:
        last_step = parsed_steps[-1]
        if last_step['function'].startswith('query_'):
            query_type = last_step['function'].replace('query_', '')
    
    def build_predicates_from_program(steps):
        """
        Build predicate list from program steps.
        Returns a flat list of all predicates.
        """
        preds = []
        var_counter = ord('a')
        
        current_obj_var = chr(var_counter)
        current_obj_path = None
        
        first_predicate = True
        
        for step in steps:
            func = step['function']
            value_inputs = step['value_inputs']
            
            if func == 'scene':
                continue
            
            elif func.startswith('filter_'):
                if value_inputs:
                    val = value_inputs[0]
                    if first_predicate:
                        preds.append(f'{val}("{current_obj_var}")')
                        first_predicate = False
                    elif current_obj_path:
                        path_str = ", ".join(current_obj_path)
                        preds.append(f'{val}(path=({path_str}))')
                    else:
                        preds.append(f'{val}(path=("{current_obj_var}"))')
            
            elif func == 'relate':
                if value_inputs:
                    relation = value_inputs[0]
                    var_counter += 1
                    rel_var = chr(var_counter)
                    
                    if current_obj_path:
                        path_parts = list(current_obj_path) + ["obj1.reversed"]
                        path_str = ", ".join(path_parts)
                        preds.append(f'{relation}("{rel_var}", path=({path_str}))')
                    else:
                        preds.append(f'{relation}("{rel_var}", path=("{current_obj_var}", obj1.reversed))')
                    
                    current_obj_path = [f'"{rel_var}"', "obj2"]
                    first_predicate = False
            
            elif func == 'unique':
                continue
            
            elif func.startswith('query_'):
                continue
            
            elif func.startswith('same_'):
                attr_type = func.replace('same_', '')
                var_counter += 1
                rel_var = chr(var_counter)
                
                if current_obj_path:
                    path_parts = list(current_obj_path) + ["obj1.reversed"]
                    path_str = ", ".join(path_parts)
                    preds.append(f'same_{attr_type}("{rel_var}", path=({path_str}))')
                else:
                    preds.append(f'same_{attr_type}("{rel_var}", path=("{current_obj_var}", obj1.reversed))')
                
                current_obj_path = [f'"{rel_var}"', "obj2"]
                first_predicate = False
        
        return preds
    
    preds = build_predicates_from_program(parsed_steps)
    
    def flat_andL(pred_list, base_indent_level):
        """Create a flat andL with all predicates as arguments."""
        if len(pred_list) == 1:
            return indent * base_indent_level + pred_list[0]
        
        lines = [indent * base_indent_level + "andL("]
        for i, pred in enumerate(pred_list):
            comma = "," if i < len(pred_list) - 1 else ""
            lines.append(indent * (base_indent_level + 1) + pred + comma)
        lines.append(indent * base_indent_level + ")")
        return "\n".join(lines)

    # Build the iotaL part
    if len(preds) == 0:
        iota_content = indent * 3 + f'obj("{property_prefix}0")'
    else:
        iota_content = flat_andL(preds, 3)
    
    # Wrap with queryL(query_type, iotaL(...))
    if query_type:
        constraint = (
            f'queryL(\n'
            f'{indent}{query_type},\n'
            f'{indent}iotaL(\n'
            + iota_content + "\n"
            f'{indent})\n'
            f')\n'
        )
    else:
        # Fallback to just iotaL if no query_type
        if len(preds) == 0:
            fallback_content = indent * 2 + f'obj("{property_prefix}0")'
        else:
            fallback_content = flat_andL(preds, 2)
        constraint = (
            f'iotaL(\n'
            + fallback_content + "\n"
            f')\n'
        )
    
    return constraint, query_type


def create_execution_for_question(program, question_index, question_type=None):
    """
    Factory function to create the appropriate execution based on question type.
    """
    if not program:
        return 'existsL(obj("prop"))\n', None
    
    last_op = program[-1]
    op_type = last_op.get('type', last_op.get('function', ''))

    return translate_left_domiknows(program, len(program) - 1, first_initial=True)
    # if op_type.startswith('query_'):
    #     return create_execution_queryL(program, question_index)
    # elif op_type == 'exist':
    #     return create_execution_existL(program, question_index), None
    #
    # elif op_type == 'count':
    #     return create_execution_existL(program, question_index), None
    #
    # elif op_type in ['equal_integer', 'less_than', 'greater_than']:
    #     return create_execution_existL(program, question_index), None
    #
    # elif op_type.startswith('equal_'):
    #     return create_execution_existL(program, question_index), None
    #
    # else:
    #     return create_execution_existL(program, question_index), None