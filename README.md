# DomiKnowS Examples Branch
This branch contains variety of examples using the DomiKnowS library.
To run a specific example, follow the steps.
- create virtual environment for the example:  
<code>python -m venv --upgrade-deps domiknowsExample</code>
- sparsely clone DomiKnowS Examples branch:  
<code>git clone --branch Examples --filter=blob:none --sparse https://github.com/HLR/DomiKnowS </code>
- sparsely checkout the example to run, for instance the demo example:  
<code>git sparse-checkout add demo</code>
- change directory to the example folder:  
<code> cd demo </code>
- install the example requirements:  
<code> pip install --no-cache-dir -r requirements.txt</code>
- execute example, for instance in the case of the demo example:  
<code>python main.py</code>
