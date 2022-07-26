import torch

def create_readers(dataset,sample_size,batch_size):
    Numbers = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four",\
               5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}
    reader=[]
    instance = {}
    for number,i in enumerate(dataset):
        if not "pixels" in instance:
            instance["pixels"]=[i[0]]
        else:
            instance["pixels"].append(i[0])
        for j in range(10):
            if not Numbers[j] in instance:
                instance[Numbers[j]]=""

        for j in range(10):
            if not j==i[1]:
                instance[Numbers[j]] += "0"
            else:
                instance[Numbers[j]] += "1"

        if number%batch_size==batch_size-1:
            instance['pixels']=torch.stack(instance['pixels'], dim=0).unsqueeze(dim=0)
            reader.append(instance)
            instance={}
    if instance:
        instance['pixels'] = torch.stack(instance['pixels'], dim=0).unsqueeze(dim=0)
        reader.append(instance)
    return reader[:sample_size]