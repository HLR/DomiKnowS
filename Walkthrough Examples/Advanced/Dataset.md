# Walkthrough Example

The followings are the user's steps to using our framework for the advanced example.

- **Dataset**
- Knowledge Declaration
- Model Declaration
- Training and Testing
- Inference

## Dataset


### Dataset Introduction

The WIQA dataset is designed for "What if..." questions over procedural text. It has three core components:

Paragraphs that depict a process, such as beach erosion.
Influence Graphs associated with each paragraph, illustrating how one alteration impacts another.
A collection of 40,000 "What if...?" multiple-choice questions that stem from the influence graphs.


### Task

 The primary objective is to evaluate a paragraph (like one on beach erosion) and respond to a "What if...?" query. For instance, one might ask, "Would stormy weather lead to increased or decreased erosion (or no effect at all)?" The dataset features questions that:

Discuss modifications highlighted in the paragraph.
Mention external perturbations not covered in the paragraph but necessitating commonsense awareness.
Reference irrelevant (yielding no effect) perturbations.
Generating Questions: Each path within the influence graph can spawn a “change-effect?” question. The questions can be classified based on:

The number of edges traversed in the graph, e.g., a single edge would be a "1-hop."
The origin of the question — whether it's rooted in details inside the paragraph (in-para) or outside it (out-of-para).
Some questions are formulated to have a "no effect" outcome. These are crafted by extracting alterations from unrelated paragraphs and gauging their influence on the current graph.

### Explanations

 The main WIQA task does not mandate explanations. However, since every question is derived from an influence graph (IG), it inherently carries a potential explanation. To capitalize on this, an explanation database was devised, keeping in mind the prospect of incorporating an explanatory task in the future.

### Refrence

See [here](https://allenai.org/data/wiqa) for more information

____
[Goto next section (Knowledge Declaration)](Knowledge%20Declaration.md)



