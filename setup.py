from setuptools import setup, find_packages
import os

DomiKnowS_version = '0.402'
DomiKnowS_version_env = os.getenv('DomiKnowS_VERSION')
DomiKnowS_VERSION_Postfix = os.getenv('DomiKnowS_VERSION_Postfix')

# Override version if set in environment
if DomiKnowS_version_env is not None:
    DomiKnowS_version = DomiKnowS_version_env[1:]
    if DomiKnowS_VERSION_Postfix is not None:
        DomiKnowS_version = DomiKnowS_version + "." + DomiKnowS_VERSION_Postfix + "0"
    
print("Using DomiKnowS version: " + DomiKnowS_version)
    
setup(
    name='DomiKnowS',
    version=DomiKnowS_version,
    description='A library provides integration between Domain Knowledge and Deep Learning.',

    long_description ='The library allows to specify a problem domain with a conceptual [graph](https://github.com/HLR/DomiKnowS/blob/main/docs/developer/KNOWLEDGE.md#graph) including declarations of edges and nodes, as well as [logical constraints](https://github.com/HLR/DomiKnowS/blob/main/docs/developer/KNOWLEDGE.md#constraints) on the graph concepts and relations. [Neural network](https://github.com/HLR/DomiKnowS/blob/main/docs/developer/MODEL.md#model-declaration) outputs bounded to the graph edges and nodes. The logical constraints are converted to ILP and Gurobi Solver is used for inferring. This adds a relational overlay over elements in a network that relates physical concepts in applications. <br> <br> The example running in Google CoLab environment, presenting the usage of the library is [here](https://colab.research.google.com/drive/1FvdePHv3h3NDSTkBw1VKwAmaZFWuGgTi).',
    long_description_content_type='text/markdown',
    
    url='https://github.com/HLR/DomiKnowS',
    author='Andrzej Uszok',
    author_email='auszok@ihmc.org',

    packages=find_packages(include=['domiknows', 'domiknows.*', 'README.md']),
    
    install_requires=[
       'acls>=1.0.2',
       'Owlready2>=0.30',
       'gurobipy',
       'pandas>=1.1.5',
       'torch>=1.8.1',
       'ordered-set',
       'graphviz',
       'pymongo[tls]',
       'dnspython',
       'scikit-learn',
       'tqdm'
    ],
    
    license='MIT',
    
    classifiers=[
        'Intended Audience :: Developers',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)