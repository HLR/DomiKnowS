from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL, atMostL, existsL

def setup_graph(fix_constraint=False):
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('CIFAR10Graph') as graph:
        image_batch = Concept(name='image_batch')
        image = Concept(name='image')
        rgb = Concept(name='RGB')
        class_label = Concept(name='class_label')

        (rel_image_batch_contains_image, rel_image_batch_contains_rgb) = image_batch.contains(image, rgb)
        (rel_image_has_class_label,) = image.has_a(arg1=class_label)

        airplane = class_label(name='airplane')
        automobile = class_label(name='automobile')
        bird = class_label(name='bird')
        cat = class_label(name='cat')
        deer = class_label(name='deer')
        dog = class_label(name='dog')
        frog = class_label(name='frog')
        horse = class_label(name='horse')
        ship = class_label(name='ship')
        truck = class_label(name='truck')

        animal = class_label(name='animal')
        vehicle = class_label(name='vehicle')
        fine_grained_animal = class_label(name='fine_grained_animal')
        fine_grained_vehicle = class_label(name='fine_grained_vehicle') 

    return graph
