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

        if not len(facts) == 0:
            silver_data.append({"name": i, "facts": [facts], "labels": [labels]})

    f = open("data/constraints_v2.json")
    f = json.load(f)


    constraints_yes = dict()
    constraints_no = dict()
    for i in f["nodes"]:
        constraints_yes[i["id"]] = set()
        constraints_no[i["id"]] = set()

    print("number of links:",len(f["links"]))

    for i in f["links"]:
        if i["weight"]=="yes_yes":
            if i["direction"]=="forward":
                constraints_yes[i["source"]].add(i["target"])
            else:
                constraints_yes[i["target"]].add(i["source"])
        else:
            if (i["direction"]=="forward" and i["weight"]=="yes_no") or (i["direction"]=="back" and i["weight"]=="no_yes"):
                constraints_no[i["source"]].add(i["target"])
            else:
                constraints_no[i["target"]].add(i["source"])

    print("data sizes:",len(calibration_data),len(silver_data),len(constraints_yes),len(constraints_no))
    return calibration_data,silver_data,constraints_yes,constraints_no