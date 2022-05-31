

def print_tags(sentence):
    tokens = sentence.split()
    tags = tagger.tag([tokens])[0]
    for token, tag in zip(tokens, tags):
        print(f'{token:12s}{tag}')


print_tags('John Johnson was born in Moscow , lives in Gothenburg , and works for Chalmers Technical University and the University of Gothenburg .')