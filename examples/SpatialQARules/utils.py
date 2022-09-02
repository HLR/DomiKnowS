def check_symmetric(arg1, arg2):
    if arg1 == arg2:
        return False
    relation_arg2 = arg2.getAttribute("relation")
    if relation_arg2 == "":
        return False
    relation_describe = relation_arg2.split(',')
    if relation_describe[0] == "symmetric":
        story1 = arg1.getAttribute("story")
        story2 = arg2.getAttribute("story")
        if story1 == story2:
            qid1 = arg1.getAttribute("id").item()
            if qid1 == int(relation_describe[1]):
                return True
    return False


def check_reverse(arg10, arg20):
    if arg10 == arg20:
        return False
    relation_arg2 = arg20.getAttribute("relation")
    if relation_arg2 == "":
        return False
    relation_describe = relation_arg2.split(',')
    if relation_describe[0] == "reverse":
        story1 = arg10.getAttribute("story")
        story2 = arg20.getAttribute("story")
        if story1 == story2:
            qid1 = arg10.getAttribute("id").item()
            if qid1 == int(relation_describe[1]):
                return True
    return False


def check_transitive(arg11, arg22, arg33):
    if arg11 == arg22 or arg11 == arg33 or arg22 == arg33:
        return False
    relation_arg3 = arg33.getAttribute("relation")
    if relation_arg3 == "":
        return False
    relation_describe = relation_arg3.split(',')
    if relation_describe[0] == "transitive":
        story1 = arg11.getAttribute("story")
        story2 = arg22.getAttribute("story")
        story3 = arg33.getAttribute("story")
        if story1 == story2 and story2 == story3:
            qid1 = arg11.getAttribute("id").item()
            qid2 = arg22.getAttribute("id").item()
            if qid1 == int(relation_describe[1]) and qid2 == int(relation_describe[2]):
                return True
    return False


def check_transitive_topo(arg111, arg222, arg333, arg444):
    if arg111 == arg222 or arg111 == arg222 or arg111 == arg333 or arg111 == arg444 or arg222 == arg333 or arg222 == arg444 or arg333 == arg444:
        return False
    relation_arg4 = arg444.getAttribute("relation")
    if relation_arg4 == "":
        return False
    relation_describe = relation_arg4.split(',')
    if relation_describe[0] == "transitive_topo":
        story1 = arg111.getAttribute("story")
        story2 = arg222.getAttribute("story")
        story3 = arg333.getAttribute("story")
        story4 = arg444.getAttribute("story")
        if story1 == story2 and story2 == story3 and story3 == story4:
            qid1 = arg111.getAttribute("id").item()
            qid2 = arg222.getAttribute("id").item()
            qid3 = arg333.getAttribute("id").item()
            if qid1 == int(relation_describe[1]) and qid2 == int(relation_describe[2]) and qid3 == int(relation_describe[3]):
                return True
    return False
