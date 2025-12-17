from dataset import g_relational_concepts, g_attribute_concepts


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
    
    def nest(pred_list, level):
        if len(pred_list) == 1:
            return indent * level + pred_list[0]
        first, rest = pred_list[0], pred_list[1:]
        return (
            indent * level + "andL(\n\t"
            + indent * (level + 1) + first + ",\n\t"
            + nest(rest, level + 1) + "\n\t"
            + indent * level + ")"
        )

    constraint = (
        f'existsL(\n\t'
        + nest(preds, 2) + "\n\t"
        f'{indent})\n\t'
    )
    return constraint


def create_execution_iotaL_simple(program, question_index, parent_img="img", property_prefix="prop", indent="\t"):
    """
    Create iotaL execution for query-type questions.
    iotaL selects THE unique entity satisfying a condition.
    
    For a question like "What color is the cube to the right of the yellow sphere?":
    - We use iotaL to select the unique cube that is right of the yellow sphere
    - The query_color tells us what attribute we're asking about (handled separately)
    - Does not parse complex nested relations
    """
    query_type = None
    if program:
        last_op = program[-1]
        op_type = last_op.get('type', last_op.get('function', ''))
        if op_type.startswith('query_'):
            query_type = op_type.replace('query_', '')
    
    values = []
    for step in program:
        func = step.get('type', step.get('function', ''))
        if func.startswith('query_'):
            continue
        for v in step.get("value_inputs", []):
            values.append((v, func))
    
    define_character = ord("a")
    check_use = False
    current_relation_var = None
    
    preds = []
    
    for i, (val, func) in enumerate(values):
        if val in g_relational_concepts.get("spatial_relation", []):
            new_var = chr(define_character + 1)
            if check_use:
                preds.append(f'{val}("{new_var}", path=("{chr(define_character)}", obj1.reversed))')
            else:
                preds.append(f'{val}("{new_var}", path=("{chr(define_character)}", obj1.reversed))')
                check_use = True
            define_character += 1
            current_relation_var = new_var
        else:
            if not check_use:
                preds.append(f'{val}("{chr(define_character)}")')
                check_use = True
            elif current_relation_var:
                preds.append(f'{val}(path=("{current_relation_var}", obj2))')
            else:
                preds.append(f'{val}(path=("{chr(define_character)}"))')
    
    if len(preds) == 0:
        constraint = f'iotaL(obj("{property_prefix}0"))\n\t'
        return constraint, query_type
    
    def nest_andL(pred_list, level):
        if len(pred_list) == 1:
            return indent * level + pred_list[0]
        first, rest = pred_list[0], pred_list[1:]
        return (
            indent * level + "andL(\n\t"
            + indent * (level + 1) + first + ",\n\t"
            + nest_andL(rest, level + 1) + "\n\t"
            + indent * level + ")"
        )

    constraint = (
        f'iotaL(\n\t'
        + nest_andL(preds, 2) + "\n\t"
        f'{indent})\n\t'
    )
    
    return constraint, query_type


def create_execution_iotaL(program, question_index, parent_img="img", property_prefix="prop", indent="\t"):
    """
    Create iotaL execution for query-type questions.
    iotaL selects THE unique entity satisfying a condition.
    
    For a question like "What color is the cube to the right of the yellow sphere?":
    - We use iotaL to select the unique cube that is right of the yellow sphere
    - The query_color tells us what attribute we're asking about (handled separately)
    
    Handles complex nested relations.
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
        
        Path structure:
        - First object: defined directly with variable "a"
        - After relation: access via (relation_var, obj2)
        - Chained relations: build compound path (prev_rel, obj2, obj1.reversed)
        """
        preds = []
        var_counter = ord('a')
        
        # Track current path to reach the "current object"
        # None means direct variable, otherwise it's a tuple like ("b", obj2)
        current_obj_var = chr(var_counter)
        current_obj_path = None  # Path to current object (None = direct var)
        
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
                        # First predicate - define the object variable
                        preds.append(f'{val}("{current_obj_var}")')
                        first_predicate = False
                    elif current_obj_path:
                        # After a relation - use path to reach object
                        path_str = ", ".join(current_obj_path)
                        preds.append(f'{val}(path=({path_str}))')
                    else:
                        # Continuation of same object description
                        preds.append(f'{val}(path=("{current_obj_var}"))')
            
            elif func == 'relate':
                if value_inputs:
                    relation = value_inputs[0]
                    var_counter += 1
                    rel_var = chr(var_counter)
                    
                    if current_obj_path:
                        # Chained relation - build path through previous relation
                        # From prev relation's obj2, follow obj1.reversed
                        path_parts = list(current_obj_path) + ["obj1.reversed"]
                        path_str = ", ".join(path_parts)
                        preds.append(f'{relation}("{rel_var}", path=({path_str}))')
                    else:
                        # First relation - from direct object variable
                        preds.append(f'{relation}("{rel_var}", path=("{current_obj_var}", obj1.reversed))')
                    
                    # Update path: current object is now at (rel_var, obj2)
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
    
    if len(preds) == 0:
        constraint = f'iotaL(obj("{property_prefix}0"))\n\t'
        return constraint, query_type
    
    def nest_andL(pred_list, level):
        if len(pred_list) == 1:
            return indent * level + pred_list[0]
        first, rest = pred_list[0], pred_list[1:]
        return (
            indent * level + "andL(\n\t"
            + indent * (level + 1) + first + ",\n\t"
            + nest_andL(rest, level + 1) + "\n\t"
            + indent * level + ")"
        )

    constraint = (
        f'iotaL(\n\t'
        + nest_andL(preds, 2) + "\n\t"
        f'{indent})\n\t'
    )
    
    return constraint, query_type


def create_execution_for_question(program, question_index, question_type=None):
    """
    Factory function to create the appropriate execution based on question type.
    """
    if not program:
        return 'existsL(obj("prop"))\n\t', None
    
    last_op = program[-1]
    op_type = last_op.get('type', last_op.get('function', ''))
    
    if op_type.startswith('query_'):
        return create_execution_iotaL(program, question_index)
    
    elif op_type == 'exist':
        return create_execution_existL(program, question_index), None
    
    elif op_type == 'count':
        return create_execution_existL(program, question_index), None
    
    elif op_type in ['equal_integer', 'less_than', 'greater_than']:
        return create_execution_existL(program, question_index), None
    
    elif op_type.startswith('equal_'):
        return create_execution_existL(program, question_index), None
    
    else:
        return create_execution_existL(program, question_index), None