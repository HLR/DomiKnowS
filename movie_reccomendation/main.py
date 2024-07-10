import torch, logging, random
from transformers import AdamW

from utils import generate_test_case
from domiknows.graph.concept import EnumConcept
from domiknows.graph import Graph, Concept, Relation
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from domiknows.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL, exactAL,atLeastAL
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateSensor
from domiknows.sensor.pytorch.learners import TorchLearner
logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:

    description = Concept(name='description')
    movie = Concept(name='movie')
    selected_movie = movie(name='selected_movie')

    similarity = Concept(name='similarity')
    selected_similarity = similarity(name='selected_similarity')

    description_contains_similarity, =description.contains(similarity)
    (movie1, movie2) = similarity.has_a(edge1=movie, edge2=movie)
    #(selected_movie1, selected_movie2) = selected_similarity.has_a(edge1=selected_movie, edge2=selected_movie)

    atLeastAL(selected_movie,2)
    exactAL(selected_movie,4)
    ifL(selected_similarity("x"),
        andL(selected_movie("y1",path=("x",movie1)),selected_movie("y2",path=("x",movie2))),
    )

from openai import OpenAI
client = OpenAI(api_key="")
openai_messages=[{"role": "system", "content": "give me a list of 10 movies seprated by ,"}]
response = client.chat.completions.create(model="gpt-4-0125-preview", messages=openai_messages, max_tokens=300, temperature=0.1)
response_text=response.choices[0].message.content

reader=[{"LLMtext":response_text}]
description["text"]=ReaderSensor(keyword='LLMtext')
movie["name"]=FunctionalSensor(description["text"], forward=lambda x:x.split(","))
movie[description_contains_similarity] = EdgeSensor(movie["name"],description["text"], relation=description_contains_similarity, forward=lambda x,_: torch.ones(len(x)).unsqueeze(-1))
def connect_movies(name, edge1, edge2):
    return True

similarity[movie1.reversed, movie2.reversed] = CompositionCandidateSensor(movie['name'], relations=(movie1.reversed, movie2.reversed), forward=connect_movies)

class DummyLearner(TorchLearner):
    def forward(self, x):
        result = torch.zeros(len(x), 2)
        result[:, 1] = -1000
        return result

class RandomLearner(TorchLearner):
    def forward(self, x,_):
        result = torch.zeros(len(x)*len(x), 2)
        result[:, 1] = -1000
        #for i in range(len(x)*len(x)):
        #    if random.random()>0.1:
        #        result[i, 1] = 10000
        result[2, 1] = 10000
        result[3, 1] = 10000
        return result

movie[selected_movie]=DummyLearner("name")
similarity[selected_similarity]=RandomLearner(movie["name"],movie["name"])
program = SolverPOIProgram(graph, poi=[description, movie, movie[selected_movie], similarity,similarity[selected_similarity]])

for datanode in program.populate(dataset=reader):
    
    print("before inference")
    print("movie index   :",list(range(1,10,1)))
    print("movie selected:", [int(child_node.getAttribute('<' + selected_movie.name + '>',"local/softmax")[1].item()) for child_node in datanode.getChildDataNodes()])
    
    datanode.inferILPResults() 
    
    print("\nafter inference")
    print("movie index   :",list(range(1,10,1)))
    print("movie selected:", [int(child_node.getAttribute('<' + selected_movie.name + '>',"ILP").item()) for child_node in datanode.getChildDataNodes()])
