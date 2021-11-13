import json

def read_data():
    f = open("data/calibration_facts.json")
    f = json.load(f)

    calibration_data = []
    for i in f.keys():
        facts=[]
        labels=[]
        for j in f[i]:
            facts.append(j)
            labels.append(f[i][j])
        calibration_data.append({"name":i,"facts":facts,"labels":labels})

    f = open("data/silver_facts.json")
    f = json.load(f)

    silver_data = []
    for i in f.keys():
        facts = []
        labels = []
        for j in f[i]:
            facts.append(j)
            labels.append(f[i][j])
        silver_data.append({"name":i,"facts":facts})

    f = open("data/constraints_v2.json")
    f = json.load(f)


    constraints=dict()
    for i in f["nodes"]:
        constraints[i["id"]]=set()

    for i in f["links"]:
        if i["weight"]=="yes_yes":
            if i["direction"]=="forward":
                constraints[i["source"]].add(i["target"])
            else:
                constraints[i["target"]].add(i["source"])

    return calibration_data,silver_data,constraints