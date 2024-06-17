import json

def read_data(sample_size=100,file_name="data/rule-reasoning-dataset-V2020.2.5.0/original/depth-5/train.jsonl",\
              meta_file="data/rule-reasoning-dataset-V2020.2.5.0/original/depth-5/meta-train.jsonl"):

    reader=[]
    counter = 0

    import json, random, re
    with open(file_name, 'r') as json_file:
        json_list = list(json_file)
    with open(meta_file, 'r') as json_file:
        json_list_meta = list(json_file)


    for json_str, meta_str in zip(json_list, json_list_meta):
        contexts = []
        questions = []
        labels = []
        proofs = []
        strategies = []
        result = json.loads(json_str)
        result_meta = json.loads(meta_str)

        for i in range(1, 20):
            try:
                rrrr = result_meta['questions']["Q" + str(i)]
            except:
                continue
            #if " not " in result_meta['questions']["Q" + str(i)]['question']:
            #    continue
            if int(result_meta['questions']["Q" + str(i)]["QDep"]) > 5:  # or int(result_meta['questions']["Q"+str(i)]["QDep"])==0:
                continue

            c = result["context"]
            contexts.append(c)
            questions.append(result_meta['questions']["Q" + str(i)]['question'])
            labels.append(result_meta['questions']["Q" + str(i)]['answer'])
            proofs.append(result_meta['questions']["Q" + str(i)]['proofs'])
            strategies.append(result_meta['questions']["Q" + str(i)]['strategy']) #proof #inv-proof

        counter += 1
        reader.append({"context":c,"questionslist":[questions],"labelslist":[labels],"proofslist":[proofs],"strategieslist":[strategies]})
        if counter>sample_size:
            break

    return reader