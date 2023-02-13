from regr.graph import Graph, Concept, Relation
from regr.graph.concept import EnumConcept
from regr.graph.logicalConstrain import nandL, exactL, ifL, orL
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR100') as graph:
    image_group = Concept(name='image_group')
    image = Concept(name='image')
    image_group_contains, = image_group.contains(image)

    category = image(name="category", ConceptClass=EnumConcept,
                     values=['aquaticmammals', 'fish', 'flowers', 'foodcontainers', 'fruitandvegetables',
                             'householdelectricaldevices', 'householdfurniture', 'insects', 'largecarnivores',
                             'largeman-madeoutdoorthings', 'largenaturaloutdoorscenes', 'largeomnivoresandherbivores',
                             'mediummammals', 'non-insectinvertebrates', 'people', 'reptiles', 'smallmammals', 'trees',
                             'vehicles1', 'vehicles2'])
    Label = image(name="tag", ConceptClass=EnumConcept,
                  values=['apple', 'aquariumfish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                          'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                          'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
                          'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
                          'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawnmower', 'leopard', 'lion',
                          'lizard', 'lobster', 'man', 'mapletree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
                          'oaktree', 'orange', 'orchid', 'otter', 'palmtree', 'pear', 'pickuptruck', 'pinetree',
                          'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                          'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                          'spider', 'squirrel', 'streetcar', 'sunflower', 'sweetpepper', 'table', 'tank', 'telephone',
                          'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
                          'willowtree', 'wolf', 'woman', 'worm'])
    parent_names = {j:i for j, i in
                    enumerate(['aquaticmammals', 'fish', 'flowers', 'foodcontainers', 'fruitandvegetables',
                               'householdelectricaldevices', 'householdfurniture', 'insects', 'largecarnivores',
                               'largeman-madeoutdoorthings', 'largenaturaloutdoorscenes', 'largeomnivoresandherbivores',
                               'mediummammals', 'non-insectinvertebrates', 'people', 'reptiles', 'smallmammals',
                               'trees',
                               'vehicles1', 'vehicles2'])}

    children_names = {j:i for j, i in
                      enumerate(['apple', 'aquariumfish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                                 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                                 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
                                 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
                                 'fox',
                                 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawnmower', 'leopard',
                                 'lion',
                                 'lizard', 'lobster', 'man', 'mapletree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
                                 'oaktree', 'orange', 'orchid', 'otter', 'palmtree', 'pear', 'pickuptruck', 'pinetree',
                                 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                                 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                                 'snake',
                                 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweetpepper', 'table', 'tank',
                                 'telephone',
                                 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                                 'whale',
                                 'willowtree', 'wolf', 'woman', 'worm'])}
    structure = {'largeomnivoresandherbivores': {'cattle', 'kangaroo', 'chimpanzee', 'elephant', 'camel'},
                 'reptiles': {'dinosaur', 'lizard', 'crocodile', 'turtle', 'snake'},
                 'fruitandvegetables': {'pear', 'orange', 'apple', 'mushroom', 'sweetpepper'},
                 'people': {'boy', 'baby', 'girl', 'woman', 'man'},
                 'fish': {'flatfish', 'shark', 'trout', 'ray', 'aquariumfish'},
                 'householdelectricaldevices': {'keyboard', 'clock', 'telephone', 'television', 'lamp'},
                 'vehicles1': {'pickuptruck', 'bicycle', 'bus', 'train', 'motorcycle'},
                 'foodcontainers': {'can', 'bowl', 'bottle', 'cup', 'plate'},
                 'largenaturaloutdoorscenes': {'mountain', 'cloud', 'sea', 'plain', 'forest'},
                 'trees': {'palmtree', 'willowtree', 'mapletree', 'pinetree', 'oaktree'},
                 'flowers': {'tulip', 'sunflower', 'orchid', 'poppy', 'rose'},
                 'largeman-madeoutdoorthings': {'bridge', 'skyscraper', 'road', 'house', 'castle'},
                 'largecarnivores': {'wolf', 'lion', 'bear', 'leopard', 'tiger'},
                 'smallmammals': {'squirrel', 'shrew', 'rabbit', 'hamster', 'mouse'},
                 'householdfurniture': {'table', 'couch', 'bed', 'wardrobe', 'chair'},
                 'mediummammals': {'skunk', 'raccoon', 'fox', 'porcupine', 'possum'},
                 'vehicles2': {'lawnmower', 'tank', 'streetcar', 'tractor', 'rocket'},
                 'insects': {'bee', 'beetle', 'caterpillar', 'butterfly', 'cockroach'},
                 'non-insectinvertebrates': {'spider', 'worm', 'snail', 'lobster', 'crab'},
                 'aquaticmammals': {'seal', 'beaver', 'whale', 'otter', 'dolphin'}}

    NEW_LC = True

    if NEW_LC:
        counter = 0
        # exactL(*[Label.__getattr__(i[1]) for i in Label.attributes])
        exactL(*[category.__getattr__(i[1]) for i in category.attributes])
        for i in category.attributes:
            lj = [Label.get_concept(l) for l in structure[i[1]]]
            ifL(category.__getattr__(i[1]), orL(*[Label.__getattr__(ii[1]) for ii in lj]))
            ifL(orL(*[Label.__getattr__(ii[1]) for ii in lj]), category.__getattr__(i[1]))
    else:
        relations = 0
        for i in category.attributes:
            for j in Label.attributes:
                if not j[1] in structure[i[1]]:
                    nandL(i, j)
                else:
                    relations += 1

        print("number of relations: ", relations)