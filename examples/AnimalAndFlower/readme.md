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
```

### with constraints using PrimeDealProgram with IML

#### Program

```python
program = PrimalDualProgram(graph, IMLModel, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                            loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=0.5)),
                            metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                    'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
```

#### result

```python
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

The best `Macro-Averaged F1` score of our model `0.905`. It shows a great improvement in the performance of our model

```python
```

#### result with Sub-class constraints

```python
```
