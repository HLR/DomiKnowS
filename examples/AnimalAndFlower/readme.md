# Animals vs Flowers example

in this example we examine the effect using hierarchical constraints on multi-class dataset

## Dataset

The dataset consists of 2 main class of `Animal` and `Flower`. Animal class itself has four subclasses named `cat`
, `dog`, `monkey` and `squirrel`. The flower class has five subclasses of `daisy`,`dandelion`,`rose`,`sunflower`
and `tuilp`.\
you can download this dataset from Google Drive
using [this link](https://drive.google.com/drive/folders/18BhzbaE4ykJ5ntSqefs_bKMehyKxUuma)

|superclass| class  | frequency |
|----|----|---|
|Animal| monkey | 299  |
|Animal   |  cat  |289|
|Animal    |  squirrel  |386|
|Animal    |  dog  |297|
|Flower  |  daisy  |577|
|Flower    |  rose  |588|
|Flower    |  tulip  |744|
|Flower    |  sunflower  |552|
|Flower    |  dandelion  |789|

|superclass|total frequency |
|----|----|
|Animal| 1271|
|Flower  |  3250  |

As you can see in the above tables the class frequencies are fairly imbalanced; hence, it could be a relatively hard
task for a neural network.

## Graph

To make constraints on this model first we define the following graph for our model

```python 
with Graph('AnimalAndFlower') as graph:
    image = Concept(name='image')

    animal = image(name='animal')
    cat = image(name='cat')
    dog = image(name='dog')
    monkey = image(name='monkey')
    squirrel = image(name='squirrel')

    flower = image(name='flower')
    daisy = image(name='daisy')
    dandelion = image(name='dandelion')
    rose = image(name='rose')
    sunflower = image(name='sunflower')
    tulip = image(name='tulip')
```

We defined three types of constraints in our model

### ifL

With this constraint we state that if a `subclass` is `true` then its `superclass` must be `true` too.

``` python
    ifL(cat('x'), animal('x'))
    ifL(dog('x'), animal('x'))
    ifL(monkey('x'), animal('x'))
    ifL(squirrel('x'), animal('x'))

    ifL(daisy('x'), flower('x'))
    ifL(dandelion('x'), flower('x'))
    ifL(rose('x'), flower('x'))
    ifL(sunflower('x'), flower('x'))
    ifL(tulip('x'), flower('x'))
```

### disjoint

with this constraint we state that at most one of the subclasses can be true at the same time

```python
    disjoint(cat, dog, monkey, squirrel, daisy, dandelion, rose, sunflower, tulip)
```

### nandL

With this constraint we state that any two-member combination of our subclasses could not be true at the same time.

```python

for l1, l2 in combinations([daisy, dandelion, rose, sunflower, tulip, cat, dog, monkey, squirrel], 2):
    nandL(l1, l2)

for l1, l2 in combinations([animal, flower], 2):
    nandL(l1, l2)
```

### Sub-class

```python
image = Concept(name='image')

animal = image(name='animal')
cat = animal(name='cat')
dog = animal(name='dog')
monkey = animal(name='monkey')
squirrel = animal(name='squirrel')

flower = image(name='flower')
daisy = flower(name='daisy')
dandelion = flower(name='dandelion')
rose = flower(name='rose')
sunflower = flower(name='sunflower')
tulip = flower(name='tulip')
```

## Experiments

### without constraints

```python
with Graph('AnimalAndFlower') as graph:
    image = Concept(name='image')

    animal = image(name='animal')
    cat = image(name='cat')
    dog = image(name='dog')
    monkey = image(name='monkey')
    squirrel = image(name='squirrel')

    flower = image(name='flower')
    daisy = image(name='daisy')
    dandelion = image(name='dandelion')
    rose = image(name='rose')
    sunflower = image(name='sunflower')
    tulip = image(name='tulip')
```

#### result

```python
(0.7129727272727273, {'animal': {'P': 1, 'R': 0.482, 'F1': 0.6505}, 'cat': {'P': 1, 'R': 0.55, 'F1': 0.7097}, 'dog': {'P': 1, 'R': 0.5, 'F1': 0.6667}, 'monkey': {'P': 1, 'R': 0.556, 'F1': 0.7147}, 'squirrel': {'P': 1, 'R': 0.572, 'F1': 0.7277}, 'flower': {'P': 1, 'R': 0.638, 'F1': 0.779}, 'daisy': {'P': 1, 'R': 0.542, 'F1': 0.703}, 'dandelion': {'P': 1, 'R': 0.526, 'F1': 0.6894}, 'rose': {'P': 1, 'R': 0.602, 'F1': 0.7516}, 'sunflower': {'P': 1, 'R': 0.584, 'F1': 0.7374}, 'tulip': {'P': 1, 'R': 0.554, 'F1': 0.713}})
```

### with constraints using PrimalDualProgram with IML loss

#### Program

```python
program = PrimalDualProgram(graph, IMLModel, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                            loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=0.5)),
                            metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                    'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
```
#### result

```python
(0.9436818181818182, {'animal': {'P': 1, 'R': 0.634, 'F1': 0.776}, 'cat': {'P': 1, 'R': 0.996, 'F1': 0.998}, 'dog': {'P': 1, 'R': 0.98, 'F1': 0.9899}, 'monkey': {'P': 1, 'R': 0.984, 'F1': 0.9919}, 'squirrel': {'P': 1, 'R': 0.944, 'F1': 0.9712}, 'flower': {'P': 1, 'R': 0.582, 'F1': 0.7358}, 'daisy': {'P': 1, 'R': 0.98, 'F1': 0.9899}, 'dandelion': {'P': 1, 'R': 0.938, 'F1': 0.968}, 'rose': {'P': 1, 'R': 1, 'F1': 1}, 'sunflower': {'P': 1, 'R': 0.932, 'F1': 0.9648}, 'tulip': {'P': 1, 'R': 0.99, 'F1': 0.995}})

 ```
### with constraints using IMLProgram

#### Program

```python
program = IMLProgram(graph, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                     loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=0.5)),
                     metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                             'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
```

#### result with ifL constraints

The best `Macro-Averaged F1` score of our model `0.9762`. It shows a great performance improvement from no-constraint model.

```python
(0.9762727272727273, {'animal': {'P': 1, 'R': 0.834, 'F1': 0.9095}, 'cat': {'P': 1, 'R': 0.986, 'F1': 0.993}, 'dog': {'P': 1, 'R': 0.996, 'F1': 0.998}, 'monkey': {'P': 1, 'R': 0.982, 'F1': 0.9909}, 'squirrel': {'P': 1, 'R': 0.998, 'F1': 0.999}, 'flower': {'P': 1, 'R': 0.754, 'F1': 0.8597}, 'daisy': {'P': 1, 'R': 0.982, 'F1': 0.9909}, 'dandelion': {'P': 1, 'R': 1, 'F1': 1}, 'rose': {'P': 1, 'R': 1, 'F1': 1}, 'sunflower': {'P': 1, 'R': 0.998, 'F1': 0.999}, 'tulip': {'P': 1, 'R': 0.998, 'F1': 0.999}})
```

#### result with Sub-class constraints

```python
(0.9676636363636363, {'animal': {'P': 1, 'R': 0.736, 'F1': 0.8479}, 'cat': {'P': 1, 'R': 0.99, 'F1': 0.995}, 'dog': {'P': 1, 'R': 0.97, 'F1': 0.9848}, 'monkey': {'P': 1, 'R': 0.958, 'F1': 0.9785}, 'squirrel': {'P': 1, 'R': 1, 'F1': 1}, 'flower': {'P': 1, 'R': 0.744, 'F1': 0.8532}, 'daisy': {'P': 1, 'R': 0.992, 'F1': 0.996}, 'dandelion': {'P': 1, 'R': 1, 'F1': 1}, 'rose': {'P': 1, 'R': 0.994, 'F1': 0.997}, 'sunflower': {'P': 1, 'R': 0.984, 'F1': 0.9919}, 'tulip': {'P': 1, 'R': 1, 'F1': 1}})
```
