import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader



def generate_people_and_locations():
    def is_real(tensor):
        return (tensor[0] > 0 and tensor[1] > 0) or (tensor[0] < 0 and tensor[1] < 0)

    def generate_entity(label_prefix):
        tensor = torch.randn(2)
        label = "real" if is_real(tensor) else "fake"
        return {
            f"{label_prefix}_tensor": tensor,
            f"{label_prefix}_label": label
        }

    def determine_work_relationship(people, locations):
        relationships = []
        for person in people:
            person_tensor = person[f"person_{people.index(person)+1}_tensor"]
            for location in locations:
                location_tensor = location[f"location_{locations.index(location)+1}_tensor"]
                works = (person_tensor[0]>0 and location_tensor[0]>0) or (person_tensor[0]<0 and location_tensor[0]<0)
                relationships.append({
                    "person": person,
                    "location": location,
                    "works": works
                })
        return relationships


    people = [generate_entity(f"person_{i+1}") for i in range(3)]
    locations = [generate_entity(f"location_{i+1}") for i in range(3)]
    work_relationships = determine_work_relationship(people, locations)

    condition_1 = (
        work_relationships[0]['works'] and work_relationships[1]['works']
        and people[0]["person_1_label"] == "real"
        and people[1]["person_2_label"] == "real"
    )
    condition_2 = (
        (work_relationships[1]['works'] and people[1]["person_2_label"] == "real")
        or (work_relationships[2]['works'] and people[2]["person_3_label"] == "real")
    )

    return {
        "people": people,
        "locations": locations,
        "work_relationships": work_relationships,
        "condition_1": condition_1,
        "condition_2": condition_2
    }


class PeopleLocationsDataset(Dataset):

    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]

        p0 = data_item["people"][0]["person_1_tensor"]
        p1 = data_item["people"][1]["person_2_tensor"]
        p2 = data_item["people"][2]["person_3_tensor"]
        l0 = data_item["locations"][0]["location_1_tensor"]
        l1 = data_item["locations"][1]["location_2_tensor"]
        l2 = data_item["locations"][2]["location_3_tensor"]

        cond1_true = float(data_item["condition_1"])
        cond2_true = float(data_item["condition_2"])

        work_for0 = data_item["work_relationships"][0]['works']
        work_for1 = data_item["work_relationships"][1]['works']
        work_for2 = data_item["work_relationships"][2]['works']

        p0_is_real = float(data_item["people"][0]["person_1_label"] == "real")
        p1_is_real = float(data_item["people"][1]["person_2_label"] == "real")
        p2_is_real = float(data_item["people"][2]["person_3_label"] == "real")

        return (p0, p1, p2, l0, l1, l2, cond1_true, cond2_true,work_for0 ,work_for1,work_for2, p0_is_real, p1_is_real, p2_is_real)

def collate_fn(batch):

    p0_list, p1_list, p2_list = [], [], []
    l0_list, l1_list, l2_list = [], [], []
    cond1_list, cond2_list = [], []
    p0_is_real_list, p1_is_real_list, p2_is_real_list = [], [], []
    wf0_list, wf1_list, wf2_list = [], [], []

    for b in batch:
        p0_list.append(b[0])
        p1_list.append(b[1])
        p2_list.append(b[2])
        l0_list.append(b[3])
        l1_list.append(b[4])
        l2_list.append(b[5])
        cond1_list.append(b[6])
        cond2_list.append(b[7])

        wf0_list.append(b[8])
        wf1_list.append(b[9])
        wf2_list.append(b[10])

        p0_is_real_list.append(b[11])
        p1_is_real_list.append(b[12])
        p2_is_real_list.append(b[13])

    p0_batch = torch.stack(p0_list, dim=0)
    p1_batch = torch.stack(p1_list, dim=0)
    p2_batch = torch.stack(p2_list, dim=0)
    l0_batch = torch.stack(l0_list, dim=0)
    l1_batch = torch.stack(l1_list, dim=0)
    l2_batch = torch.stack(l2_list, dim=0)

    cond1_batch = torch.tensor(cond1_list, dtype=torch.float)
    cond2_batch = torch.tensor(cond2_list, dtype=torch.float)


    wf0_batch = torch.tensor(wf0_list, dtype=torch.float)
    wf1_batch = torch.tensor(wf1_list, dtype=torch.float)
    wf2_batch = torch.tensor(wf2_list, dtype=torch.float)

    p0_is_real_batch = torch.tensor(p0_is_real_list, dtype=torch.float)
    p1_is_real_batch = torch.tensor(p1_is_real_list, dtype=torch.float)
    p2_is_real_batch = torch.tensor(p2_is_real_list, dtype=torch.float)

    return (
        p0_batch, p1_batch, p2_batch,
        l0_batch, l1_batch, l2_batch,
        cond1_batch, cond2_batch,
        wf0_batch, wf1_batch, wf2_batch,
        p0_is_real_batch, p1_is_real_batch, p2_is_real_batch
    )


def generate_dataset(sample_num=1000):
    data_list = []
    for i in range(sample_num):
        data = generate_people_and_locations()
        while not data["condition_1"] == True:
            data = generate_people_and_locations()
        data_list.append(data)
    for i in range(sample_num):
        data = generate_people_and_locations()
        while not data["condition_1"] == False:
            data = generate_people_and_locations()
        data_list.append(data)
    for i in range(sample_num):
        data = generate_people_and_locations()
        while not data["condition_2"] == True:
            data = generate_people_and_locations()
        data_list.append(data)
    for i in range(sample_num):
        data = generate_people_and_locations()
        while not data["condition_2"] == False:
            data = generate_people_and_locations()
        data_list.append(data)

    random.shuffle(data_list)
    return data_list


def reader_format(data_list):
    reader_list = []
    for data in data_list:
        reader_item = {
            "person1": [data["people"][0]['person_1_tensor'].tolist()],
            "person2": [data["people"][1]['person_2_tensor'].tolist()],
            "person3": [data["people"][2]['person_3_tensor'].tolist()],
            "person1_label": [data["people"][0]['person_1_label']=="real"],
            "person2_label": [data["people"][1]['person_2_label']=="real"],
            "person3_label": [data["people"][2]['person_3_label']=="real"],
            "location1": [data["locations"][0]['location_1_tensor'].tolist()],
            "location2": [data["locations"][1]['location_2_tensor'].tolist()],
            "location3": [data["locations"][2]['location_3_tensor'].tolist()],
            "condition_1": [data["condition_1"].item() if isinstance(data["condition_1"], torch.Tensor) else data["condition_1"]],
            "condition_2": [data["condition_2"].item() if isinstance(data["condition_2"], torch.Tensor) else data["condition_2"]],
        }
        reader_list.append(reader_item)
    return reader_list

is_real_person = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

work_for = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)