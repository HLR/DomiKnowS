def check_symmetric(arg1, arg2):
    """
    Define the symmetric of given arguments
    Parameters
    ----------
    arg1 - the first sentence
    arg2 - the second sentence

    Returns
    -------
    Boolean - Whether these arguments are symmetric or not
    """
    if arg1 == arg2:
        return False
    h1, p1 = arg1.getAttribute("hypothesis"), arg1.getAttribute("premise")
    h2, p2 = arg2.getAttribute("hypothesis"), arg2.getAttribute("premise")
    return h1 == p2 and h2 == p1


def check_transitive(arg11, arg22, arg33):
    """
    Define the transitive of given arguments
    Parameters
    ----------
    arg1 - the first sentence
    arg2 - the second sentence
    arg3 - the third sentence

    Returns
    -------
    Boolean - Whether these arguments are transitive or not
    """
    if arg11 == arg22 or arg22 == arg33 or arg11 == arg33:
        return False
    h1, p1 = arg11.getAttribute("hypothesis"), arg11.getAttribute("premise")
    h2, p2 = arg22.getAttribute("hypothesis"), arg22.getAttribute("premise")
    h3, p3 = arg33.getAttribute("hypothesis"), arg33.getAttribute("premise")
    #Ent(X1, X2) + Ent(X2, X3) => Ent(X1, X3)
    return p1 == h2 and h1 == h3 and p2 == p3
