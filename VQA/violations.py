import torch


def get_constraints_satisfaction(node):
    """
    Get constraint satisfaction from datanode
    Returns number of satisfied constraints and total number of constraints
    """
    verifyResult = node.verifyResultsLC(key = "/local/argmax")

    assert verifyResult

    satisfied_constraints = []
    for lc_idx, lc in enumerate(verifyResult):
        satisfied_constraints.append(verifyResult[lc]['satisfied'])

    num_constraints = len(verifyResult)
    num_satisfied = sum(satisfied_constraints) // 100

    return num_satisfied, num_constraints


def violation_num(adj_mat, labels):
    # adj_mat is V*V, and labels is a 1*V vector , which V is number of image labels.
    V_vec = torch.matmul(labels, adj_mat) # B*E vector with {positive or zero}=no_violaton, {negative} = violation
    V_num = torch.sum(V_vec<0, 1)
    #V_avg = torch.mean(V_num.float())
    #print(f'Violation scores, and Violation average: {V_vec},  {V_avg}')
    return V_num


def labels_to_oh(l1, l2, l3, l4):
    oh = [
        torch.zeros((1, 274)),
        torch.zeros([1, 157]),
        torch.zeros([1, 62]),
        torch.zeros([1, 7])
    ]

    if l1 < oh[0].shape[1]:
        oh[0][0, l1] = 1
    
    if l2 < oh[1].shape[1]:
        oh[1][0, l2] = 1
    
    if l3 < oh[2].shape[1]:
        oh[2][0, l3] = 1
    
    if l4 < oh[3].shape[1]:
        oh[3][0, l4] = 1

    oh_cat = torch.cat(oh, dim=-1)
    
    return oh_cat


def adj_matrix_violations(adj_matrix, preds):
    return violation_num(adj_matrix, labels_to_oh(*preds))
