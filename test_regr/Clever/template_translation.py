# TODO: Edit this to match DomiKnowS
#

prompt_clevr = """
I am going to ask you to write some programs to answer some questions based on a scene. You will be using the Python language, but you should only use the following features:

1. To classify whether an object has a certain property, use "is_cat('x')", where `cat` is the property you want to test, and 'x' is a variable.
2. You can use and, or, not Boolean operations based on these classification results. For example, AndL(is_cat('x'), is_dog('y'))
4. To link two objects are the same objects, you need to use path=('x') where 'x' is variable For example, "andL(is_cat('x'), is_lawn(path=('x'))".
5. When you want to check whether there exists a cat in the scene, use "existsL(is_cat('x'))".
6. When you want to count the number of cats in a scene exactly equal k, use "countL(is_cat('x'), k)", k is numerical number
7. When there is relation between two objects, use following "is_left('y', path=('x', obj1.reversed))". After this relation, every time referring 'y' need to be in this follow "is_red(path=('y', obj2))" where 'red' is the property.
8. Possible relations are left, right, front, behide, near, on
9. Other than these specified functions and variables in lambda functions, you should not invent new variables or constants. You also shouldn't use built-in comparison primitives, such as == and >=.
10. You should make your invented functions as primitive as possible. For example, instead of using "white_cat('x')", use "andL(white(x), cat(path=('x')))"
11. Please make sure that your expression has balanced ( and ).
12. before you write the programs, first translate the language into a more simple form, removing excess information, for example, simplify <text>Is the color of the towel hanging draped over the side of the tub red?</text> to <simplified>exist red color of the towel on the tub</simplified>, before writing the correct program, <code>existL(andL(is_red('x'), is_towel(path=('x')), is_on('y', path=('x', obj1.reversed)), is_bathtub(path=('y', obj2))))</code>.

----

Now I first give you a few examples:

<text>Is there an apple next to the cake that is front and center of the room?</text>
<simplified>exists apple next to the cake</simplified>
<code>existsL(apple('x'), next('y', path=('x', obj1.reversed)))
<code>existsL(Object, lambda x: apple(x) and next(x, iota(Object, lambda y: and cake(y))))</code>

<text>What is the size of the apple?</text>
<simplified>size of apple</simplified>
<code>describe(Size, iota(Object, lambda y: apple(y)))</code>

<text>What is the shape of the cake that's not pink by itself in the corner?</text>
<simplified>shape of cake not pink</simplified>
<code>describe(Shape, iota(Object, lambda y: cake(y) and not(pink(y))))</code>

<text>What is the color of the cake in the room?</text>
<simplified>color of the cake</simplified>
<code>describe(Color, iota(Object, lambda y: cake(y)))</code>

<text>What is the material of the cat without an apple on itself?</text>
<simplified>material of the cat that does not have an apple</simplified>
<code>describe(Material, iota(Object, lambda y: cat(y) and not(have(y, iota(Object, lambda z: apple(z))))))</code>

<text>How many apples does the cat have in its paws?</text>
<simplified>count apples the cat have</simplified>
<code>count(Object, lambda x: have(iota(Object, lambda y: cat(y)), x) and apple(x))</code>

<text>Are there more apples over cakes at the end of the room?</text>
<simplified>greater count of apples than cakes</simplified>
<code>greater_than(count(Object, lambda x: apple(x)), count(Object, lambda x: cake(x)))</code>


<text>Is there a same number of cakes and apples or is there less cakes than apples?</text>
<simplified>equal count of cakes and apples or less than count of cakes than apples</simplified>
<code>equal(count(Object, lambda x: cake(x)), count(Object, lambda x: apple(x))) or less_than(count(Object, lambda x: cake(x)), count(Object, lambda x: apple(x)))</code>


Now please translate the following text into a program. First output a simplified text version inside <simplified></simplified>, with a shortened version of <text>. Then, using ONLY the simplified text, without looking at the original text, output a short translated program inside <code></code>.
"""


import ast
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import time
from tqdm import tqdm
import transformers
import torch
import json
import os
import argparse
from sklearn.metrics import confusion_matrix
import torch
import copy



os.environ["HF_HOME"] = "/egr/research-hlr2/premsrit/transformer_cache"
os.environ["HF_DATASETS_CACHE"] = "/egr/research-hlr2/premsrit/transformer_cache"
os.environ["TRANSFORMERS_CACHE"] = "/egr/research-hlr2/premsrit/transformer_cache"


class Qwen3Model():

    def __init__(self, model_size=None, device=None, port=None, enable_reasoning=True):
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8307/v1" if enable_reasoning else "http://localhost:8307/v1"

        if port is not None and port != "":
            openai_api_base = f"http://localhost:{port}/v1"
        self.enable_reasoning = True

        print(openai_api_base)
        self.client = OpenAI(
                            api_key=openai_api_key,
                            base_url=openai_api_base,
                        )

        models = self.client.models.list()
        self.model = models.data[0].id

    def __call__(self, message, enable_thinking=False):
        response = self.client.chat.completions.create(model=self.model, 
                                                    messages=message,
                                                    extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_reasoning}},
                                                    timeout=3000,
                                                    max_tokens=8196)
            
        thinking = response.choices[0].message.reasoning_content
        raw_response = response.choices[0].message.content
        # print(thinking, raw_response)
        return thinking, raw_response