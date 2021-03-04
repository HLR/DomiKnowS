
import networkx as nx
import re
import matplotlib.pyplot as plt
def extract_cause_effect(question):
    # a helper method to extract events for cause and effect given a WIQA question.
    #print(question)
    q_cause = re.findall('suppose (.*) happens', question)[0]
    q_effect = re.findall('how will it affect (.*).', question)[0]
    return q_cause, q_effect

def make_graph_darius(paragraph,questions,answers):
    G=nx.DiGraph()
    nodes=set()
    for i in questions:
        c,a=extract_cause_effect(i)
        nodes.add(c)
        nodes.add(a)
    G.add_nodes_from(list(nodes))
    get_weight={"more":1,"less":-1,"no_effect":0}.get
    G.add_edges_from([tuple([*extract_cause_effect(i),{'weight':get_weight(j)}]) for i,j in zip(questions,answers)])
    print("number of nodes and edges are: ",G.number_of_nodes(),G.number_of_edges())

    nx.draw_networkx_edge_labels(G,pos=nx.circular_layout(G),edge_labels=dict([((u,v),d['weight']) for u,v,d in G.edges(data=True)]))

    get_color={1:"green",0:'black',-1:'red'}.get

    nx.draw_circular(G,node_color='white',with_labels=True,edge_color=[get_color(G[edge[0]][edge[1]]['weight']) for edge in G.edges()] )
    plt.show()
