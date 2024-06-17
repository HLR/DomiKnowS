# Demo of complete DomiKnowS program

This example show all the basic steps in configuring learning program in the [DomiKnowS environment](https://hlr.github.io/domiknows-nlp/pipeline).

First the [domain graph](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L30) is defined including all the [concepts](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L34) and [relations](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L39) the program is going to learn about the data.
https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L30

This follows by [definition of logical constraints](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/docs/developer/KNOWLEDGE.md#logical-constraints-lc), which use concepts from the graph and encode the knowledge about the domain. https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L46 https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L49 

The [sample](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L55) for the example and the [reader](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L72) of the data is presented next.

After that the model [sensors](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L81)https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L86 and [learner](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L92)https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L92 for the graph concepts are configured from the sets of sensors available in the DomiKnowS environment.
 
The demo uses [POIProgram](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L106)https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L106 to first [train](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L118) and then [test](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L129) the model.

As the next step ILP solver is called to [infer](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L146)https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L146 the best solution from the learnet model based on the encoded knowledge in logical constraints.

The demo show how to access the [learned parameters](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L152)https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L152 and [ILP solver solutions](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L160)https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L160 from [Datanode](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L147).

It concludes with [verification](https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L171) of logical constraints compliance. https://github.com/HLR/DomiKnowS/blob/16e4f93a050eb412afc0eee16c9f239198da081c/examples/demo/main.py#L171

## How To Run
```
!python main.py
```
