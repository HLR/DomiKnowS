import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from NER_utils import PeopleLocationsDataset, collate_fn, is_real_person, work_for, generate_dataset

criterion = nn.BCELoss()


def forward_conditions(p0_batch, p1_batch, p2_batch, l0_batch, l1_batch, l2_batch):
    real_p0 = is_real_person(p0_batch)
    real_p1 = is_real_person(p1_batch)
    real_p2 = is_real_person(p2_batch)

    wf_0 = work_for(torch.cat([p0_batch, l0_batch], dim=1))
    wf_1 = work_for(torch.cat([p1_batch, l1_batch], dim=1))
    wf_2 = work_for(torch.cat([p2_batch, l2_batch], dim=1))

    cond1_pred = torch.min(
        torch.min(wf_0, wf_1),
        torch.min(real_p0, real_p1)
    )

    left = torch.min(wf_1, real_p1)
    right = torch.min(wf_2, real_p2)
    cond2_pred = torch.max(left, right)

    return real_p0.view(-1), real_p1.view(-1), real_p2.view(-1), cond1_pred.view(-1), cond2_pred.view(-1)


@torch.no_grad()
def evaluate(is_real_person_model, work_for_model, loader, device):
    is_real_person_model.eval()
    work_for_model.eval()

    total_examples = 0
    correct_cond1 = 0
    correct_cond2 = 0

    correct_wf = 0
    total_wf = 0

    for batch in loader:
        (
            p0, p1, p2,
            l0, l1, l2,
            cond1_true, cond2_true,
            wf0_true, wf1_true, wf2_true,
            p0_is_real, p1_is_real, p2_is_real
        ) = batch

        p0 = p0.to(device)
        p1 = p1.to(device)
        p2 = p2.to(device)
        l0 = l0.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        cond1_true = cond1_true.to(device)
        cond2_true = cond2_true.to(device)

        wf0_true = wf0_true.to(device)
        wf1_true = wf1_true.to(device)
        wf2_true = wf2_true.to(device)

        _, _, _, cond1_pred, cond2_pred = forward_conditions(p0, p1, p2, l0, l1, l2)

        cond1_label = (cond1_pred >= 0.5).float()
        cond2_label = (cond2_pred >= 0.5).float()

        correct_cond1 += (cond1_label == cond1_true).sum().item()
        correct_cond2 += (cond2_label == cond2_true).sum().item()
        total_examples += cond1_true.size(0)

        wf0_pred = work_for_model(torch.cat([p0, l0], dim=1))
        wf1_pred = work_for_model(torch.cat([p1, l1], dim=1))
        wf2_pred = work_for_model(torch.cat([p2, l2], dim=1))

        wf0_label = (wf0_pred >= 0.5).float()
        wf1_label = (wf1_pred >= 0.5).float()
        wf2_label = (wf2_pred >= 0.5).float()

        correct_wf += (wf0_label == wf0_true).sum().item()
        correct_wf += (wf1_label == wf1_true).sum().item()
        correct_wf += (wf2_label == wf2_true).sum().item()

        total_wf += (wf0_true.size(0) * 3)

    acc_cond1 = correct_cond1 / total_examples
    acc_cond2 = correct_cond2 / total_examples
    acc_work_for = correct_wf / total_wf

    return acc_cond1, acc_cond2, acc_work_for


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dataset():
    sample_num = 1000
    data_list = generate_dataset(sample_num=sample_num)
    
    train_ratio = 0.8
    train_size = int(train_ratio * sample_num)
    train_data = data_list[:train_size]
    dev_data = data_list[train_size:]
    
    return train_data, dev_data


@pytest.fixture
def data_loaders(dataset):
    train_data, dev_data = dataset
    
    train_dataset = PeopleLocationsDataset(train_data)
    dev_dataset = PeopleLocationsDataset(dev_data)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, dev_loader


def test_training(device, data_loaders):
    print("\nUsing device:", device)

    is_real_person.to(device)
    work_for.to(device)

    train_loader, dev_loader = data_loaders

    learning_rate = 8e-4
    num_epochs = 200
    optimizer = optim.Adam(
        list(is_real_person.parameters()) + list(work_for.parameters()),
        lr=learning_rate
    )

    for epoch in range(num_epochs):
        is_real_person.train()
        work_for.train()

        total_loss = 0.0
        for batch in train_loader:
            (
                p0, p1, p2,
                l0, l1, l2,
                cond1_true, cond2_true,
                wf0_true, wf1_true, wf2_true,
                p0_is_real, p1_is_real, p2_is_real
            ) = batch

            p0 = p0.to(device)
            p1 = p1.to(device)
            p2 = p2.to(device)
            l0 = l0.to(device)
            l1 = l1.to(device)
            l2 = l2.to(device)
            cond1_true = cond1_true.to(device)
            cond2_true = cond2_true.to(device)
            wf0_true = wf0_true.to(device)
            wf1_true = wf1_true.to(device)
            wf2_true = wf2_true.to(device)

            p0_is_real = p0_is_real.to(device)
            p1_is_real = p1_is_real.to(device)
            p2_is_real = p2_is_real.to(device)

            p0_pred, p1_pred, p2_pred, cond1_pred, cond2_pred = forward_conditions(p0, p1, p2, l0, l1, l2)

            loss1 = criterion(cond1_pred, cond1_true)
            loss2 = criterion(cond2_pred, cond2_true)

            loss3 = criterion(p0_pred, p0_is_real)
            loss4 = criterion(p1_pred, p1_is_real)
            loss5 = criterion(p2_pred, p2_is_real)

            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        cond1_acc, cond2_acc, work_for_acc = evaluate(
            is_real_person, work_for, dev_loader, device
        )

        print(f"Epoch {epoch + 1}/{num_epochs} "
              f"| Train Loss = {total_loss:.4f} "
              f"| Dev Cond1 Acc = {cond1_acc:.3f} "
              f"| Dev Cond2 Acc = {cond2_acc:.3f} ")

    assert cond1_acc > 0.5, f"Condition 1 accuracy too low: {cond1_acc}"
    assert cond2_acc > 0.5, f"Condition 2 accuracy too low: {cond2_acc}"