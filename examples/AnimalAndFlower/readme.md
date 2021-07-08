# Animals vs Flowers example

in this example we examine the effect using hierarchical constraints on multi-class dataset

## Dataset

The dataset consists of 2 main class of `Animal` and `Flower`. Animal class itself has four subclasses named `cat`
, `dog`, `monkey` and `squirrel`. The flower class has five subclasses of `daisy`,`dandelion`,`rose`,`sunflower`
and `tuilp`.\
you can download this dataset from Google Drive using [this link](https://drive.google.com/drive/folders/18BhzbaE4ykJ5ntSqefs_bKMehyKxUuma)

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

As you can see in the above tables the class frequencies are fairly imbalanced; hence, it could be a relatively hard task for a neural network.

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

## Result
In this section we 
if we run our mode without the constraint we get the following result
### without constraints
```python
(0.6514909090909091, {'animal': {'P': 1, 'R': 0.466, 'F1': 0.6357}, 'cat': {'P': 1, 'R': 0.566, 'F1': 0.7229}, 'dog': {'P': 1, 'R': 0.512, 'F1': 0.6772}, 'monkey': {'P': 1, 'R': 0.532, 'F1': 0.6945}, 'squirrel': {'P': 1, 'R': 0.444, 'F1': 0.615}, 'flower': {'P': 1, 'R': 0.762, 'F1': 0.8649}, 'daisy': {'P': 1, 'R': 0.822, 'F1': 0.9023}, 'dandelion': {'P': 1, 'R': 0.442, 'F1': 0.613}, 'rose': {'P': 1, 'R': 0.698, 'F1': 0.8221}, 'sunflower': {'P': 1, 'R': 0.448, 'F1': 0.6188}})
```

### with constraints using IMLProgram
```python
 (0.905, {'animal': {'P': 1, 'R': 0.956, 'F1': 0.9775}, 'cat': {'P': 1, 'R': 1, 'F1': 1}, 'dog': {'P': 1, 'R': 1, 'F1': 1}, 'monkey': {'P': 1, 'R': 1, 'F1': 1}, 'squirrel': {'P': 1, 'R': 1, 'F1': 1}, 'flower': {'P': 1, 'R': 0.956, 'F1': 0.9775}, 'daisy': {'P': 1, 'R': 1, 'F1': 1}, 'dandelion': {'P': 1, 'R': 1, 'F1': 1}, 'rose': {'P': 1, 'R': 1, 'F1': 1}, 'sunflower': {'P': 1, 'R': 1, 'F1': 1}})
```

The `Macro-Averaged F1` score of our mode without constraints is `0.651`. while its score in the constrained mode is `0.905`.
It shows a great improvement in the performance of our model

### with constraints using PrimeDealProgram with IML
```python
 (0.8712, {'animal': {'P': 1, 'R': 0.792, 'F1': 0.8839}, 'cat': {'P': 1, 'R': 0.844, 'F1': 0.9154}, 'dog': {'P': 1, 'R': 0.972, 'F1': 0.9858}, 'monkey': {'P': 1, 'R': 0.988, 'F1': 0.994}, 'squirrel': {'P': 1, 'R': 0.97, 'F1': 0.9848}, 'flower': {'P': 1, 'R': 0.796, 'F1': 0.8864}, 'daisy': {'P': 1, 'R': 0.894, 'F1': 0.944}, 'dandelion': {'P': 1, 'R': 1, 'F1': 1}, 'rose': {'P': 1, 'R': 0.998, 'F1': 0.999}, 'sunflower': {'P': 1, 'R': 0.98, 'F1': 0.9899}})
 ```

