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
for datanode in program.populate(test_reader):
    print(datanode.getAttribute("text"))
    print(datanode.getChildDataNodes(conceptName=phrase))
    datanode.getChildDataNodes(conceptName=phrase)[0].visualize(filename="./datanode_image", inference_mode="ILP")
    break
```
`program.populate` given the reader, will create a datagraph of `Datanode`s and returns a list of "Root" concepts.

You can run this full example in [jupytor](https://colab.research.google.com/drive/1FvdePHv3h3NDSTkBw1VKwAmaZFWuGgTi#scrollTo=xy8cicGARIyN).