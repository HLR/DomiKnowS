def check_symmetric(arg1, arg2):
    if arg1 == arg2:
        return False
    h1, p1 = arg1.getAttribute("hypothesis"), arg1.getAttribute("premise")
    h2, p2 = arg2.getAttribute("hypothesis"), arg2.getAttribute("premise")
    return h1 == p2 and h2 == p1
