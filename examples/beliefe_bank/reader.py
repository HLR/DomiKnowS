import json

def read_data(batch_size=16,sample_size=10000000):
    f = open("data/calibration_facts.json")
    f = json.load(f)

    sample_counter=0
    calibration_data = []
    for i in f.keys():
        facts=[]
        labels=[]
        for j in f[i]:
            facts.append(j)
            labels.append(f[i][j])
            if len(facts)==batch_size:
                calibration_data.append({"name":i,"facts":[facts],"labels":[labels]})
                facts = []
                labels = []
                sample_counter+=1
                if sample_counter>=sample_size:
                    break
        if sample_counter >= sample_size:
            break
        if not len(facts)==0:
            calibration_data.append({"name": i, "facts": [facts], "labels": [labels]})
            sample_counter += 1


    f = open("data/silver_facts.json")
    f = json.load(f)

    sample_counter=0
    silver_data = []
    for i in f.keys():
        facts = []
        labels = []
        for j in f[i]:
            facts.append(j)
            labels.append(f[i][j])
            if len(facts)==batch_size:
                silver_data.append({"name":i,"facts":[facts],"labels":[labels]})
                facts = []
                labels = []
                sample_counter+=1
                if sample_counter>=sample_size:
                    break
        if sample_counter >= sample_size:
            break
        if not len(facts) == 0:
            silver_data.append({"name": i, "facts": [facts], "labels": [labels]})
            sample_counter += 1

    f = open("data/constraints_v2.json")
    f = json.load(f)


    constraints=dict()
    for i in f["nodes"]:
        constraints[i["id"]]=set()
    print(len(f["links"]))
    for i in f["links"]:
        if i["weight"]=="yes_yes":
            if i["direction"]=="forward":
                constraints[i["source"]].add(i["target"])
            else:
                constraints[i["target"]].add(i["source"])
    print(len(calibration_data),len(silver_data),len(constraints))
    return calibration_data,silver_data,constraints