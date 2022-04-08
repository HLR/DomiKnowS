def check_symmetric(sentence1, sentence2):
    if sentence1 == sentence2:
        return False
    h1, p1 = sentence1.getAttribute("hypothesis"), sentence1.getAttribute("premise")
    h2, p2 = sentence2.getAttribute("hypothesis"), sentence2.getAttribute("premise")
    return h1 == p2 and h2 == p1
