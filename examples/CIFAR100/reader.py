import pickle,torch
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_readers(train_num=50000,test_num=10000,batch_size=64):
    structure={}
    def unpickle(file_name):
        with open(file_name , 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    train_data = unpickle("cifar-100-python/train")
    test_data = unpickle("cifar-100-python/test")
    meta = unpickle("cifar-100-python/meta")

    train_reader,test_reader=[],[]
    batch_train_reader, batch_test_reader = [], []
    coarse_names=meta[b'coarse_label_names']
    fine_names = meta[b'fine_label_names']
    for i in range(train_num):
        instance={}
        instance['pixels']=preprocess(train_data[b'data'][i].reshape((32,32,3)))
        instance['corase_label'] = train_data[b'coarse_labels'][i]
        instance['fine_label'] = train_data[b'fine_labels'][i]
        instance['corase_label_name'] = coarse_names[(train_data[b'coarse_labels'][i])].decode('ascii')
        instance['fine_label_name'] = fine_names[(train_data[b'fine_labels'][i])].decode('ascii')
        if instance['corase_label_name'] in structure:
            structure[instance['corase_label_name']].add(instance['fine_label_name'])
        else:
            structure[instance['corase_label_name']]=set()
            structure[instance['corase_label_name']].add(instance['fine_label_name'])
        train_reader.append(instance)

    for i in range(0,train_num,batch_size):
        pixels_group,coarse_names_group,fine_names_group=[],[],[]
        for i_ in range(i,min(i+batch_size,train_num)):
            pixels_group.append(train_reader[i_]['pixels'])
            coarse_names_group.append(train_reader[i_]['corase_label'])
            fine_names_group.append(train_reader[i_]['fine_label'])
        intance={}
        intance['pixels']=torch.stack(pixels_group, dim=0).unsqueeze(dim=0)
        intance['corase_label']="@@".join([str(j) for j in coarse_names_group])
        intance['fine_label']="@@".join([str(j) for j in fine_names_group])
        batch_train_reader.append(intance)

    for i in range(test_num):
        instance={}
        instance['pixels']=preprocess(test_data[b'data'][i].reshape((32,32,3)))
        instance['corase_label'] = test_data[b'coarse_labels'][i]
        instance['fine_label'] = test_data[b'fine_labels'][i]
        instance['corase_label_name'] = coarse_names[(test_data[b'coarse_labels'][i])].decode('ascii')
        instance['fine_label_name'] = fine_names[(test_data[b'fine_labels'][i])].decode('ascii')
        test_reader.append(instance)

    for i in range(0,test_num,batch_size):
        pixels_group,coarse_names_group,fine_names_group=[],[],[]
        for i_ in range(i,min(i+batch_size,test_num)):
            pixels_group.append(test_reader[i_]['pixels'])
            coarse_names_group.append(test_reader[i_]['corase_label'])
            fine_names_group.append(test_reader[i_]['fine_label'])
        intance={}
        intance['pixels']=torch.stack(pixels_group, dim=0).unsqueeze(dim=0)
        intance['corase_label']="@@".join([str(j) for j in coarse_names_group])
        intance['fine_label']="@@".join([str(j) for j in fine_names_group])
        batch_test_reader.append(intance)

    return batch_train_reader,batch_test_reader