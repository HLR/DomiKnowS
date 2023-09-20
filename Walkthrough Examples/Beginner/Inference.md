# Walkthrough Example

The followings are the user's steps to using our framework.

- Dataset
- Knowledge Declaration
- Model Declaration
- Training and Testing
- **Inference**

## Inference



## Inference

One feature of our framework is an automatic inference based on domain knowledge.
To try this out, the user must first create `Datanode`.

```python
    for datanode in lbp.populate(test_dataset):
        print('datanode:', datanode)
        print('Spam:', datanode.getAttribute(Spam))
        print('Regular:', datanode.getAttribute(Regular))
        print('inference spam:', datanode.getAttribute(Spam, 'ILP'))
        print('inference regular:', datanode.getAttribute(Regular, 'ILP'))
```
`program.populate` given the reader, will create a datagraph of `Datanode`s and returns a list of "Root" concepts. the "Root" concept here is the `email` concept. each `email` is an instance of `Datanode` class. `ILP` enforces our constraints and ensures that only one label is correct.

You can run this full example in [jupytor](https://colab.research.google.com/drive/17TAMCNBfyzJeAoc90epbWlJx2Ndr9g0u?usp=sharing).