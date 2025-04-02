import torch
from model import Net
from utils import batch_iterator
from dataclasses import dataclass, field
from typing import Callable
from tqdm import tqdm


@dataclass
class TrainingConfig:
    approx_select: Callable[[torch.Tensor, str], torch.Tensor]
    device: str
    
    operations: list[str] = field(default_factory=lambda: ['summation', 'subtraction', 'multiplication', 'division'])
    batch_size: int = 64
    num_epochs: int = 10
    lr: float = 1e-3
    temp: float = 1.0

@torch.no_grad()
def get_accuracy_iter(
        dataloader: torch.utils.data.DataLoader,
        net: torch.nn.Module,
        device: str
    ) -> tuple[float, int]:
    '''
    Get digit classification accuracy using samples from a DataLoader.

    Input:
        dataloader: DataLoader object.
        net: model to be evaluated.
        device: str indicating device to use (e.g., cpu).
    
    Output:
        tuple of accuracy and total number of samples.
    '''

    correctness_all = []

    for batch_data in batch_iterator(dataloader, batch_size=32):
        # collate batch data
        digits_batch = torch.stack([x['pixels'] for x in batch_data], dim=0).to(device)
        digit_labels_batch = torch.stack([x['digit'] for x in batch_data], dim=0).to(device) # (batch_size, 2)

        # forward pass for all digit predictions
        digits_batch_flat = digits_batch.view(-1, 28*28)
        logits_batch_flat = net(digits_batch_flat)

        logits_batch = logits_batch_flat.view(-1, 2, 10)
        
        # get accuracies
        digit_preds = logits_batch.argmax(dim=-1) # (batch_size, 2)
        
        batch_correctness = (digit_preds == digit_labels_batch).flatten() # (batch_size * 2,)

        correctness_all.extend(batch_correctness.cpu().tolist())
    
    return (sum(correctness_all) / len(correctness_all), len(correctness_all))

def loop(
        config: TrainingConfig,
        trainloader: torch.utils.data.DataLoader,
        validloader: torch.utils.data.DataLoader
    ) -> list[dict[str, list]]:
    '''
    Trains and collects metrics across all operations (specified in config.operations).

    Input:
        config: TrainingConfig object specifying training hyperparameters.
        trainloader: DataLoader for training data.
        validloader: DataLoader for validation data.
    
    Output:
        list of metrics across all epochs & operations.
        
        Each dictionary in the list corresponds to metrics for a single epoch.
        The `operation` key indicates the operation (e.g., 'summation', 'subtraction', etc.)
        being evaluated.
    '''

    loss = torch.nn.MSELoss()

    batch_size = config.batch_size
    num_epochs = config.num_epochs
    all_metrics = []

    for operation in config.operations:
        # perform a training run for this operation
        net = Net().to(config.device)
        optim = torch.optim.Adam(net.parameters(), lr=config.lr)

        for epoch_idx in range(num_epochs):
            metrics_epoch = {
                'operation': [], # summation, subtraction, multiplication, division
                'epoch_idx': [],
                'iter_idx': [],
                'loss_summation': [], # training loss across iters
                'correct_digit0': [], # training accuracy across iters
                'correct_digit1': [] # training accuracy across iters
            }

            total = len(trainloader) // batch_size
            total += 1 if len(trainloader) % batch_size != 0 else 0

            for iter_idx, batch_data in tqdm(
                    enumerate(batch_iterator(
                        trainloader,
                        batch_size=batch_size,
                        filter_f=(lambda x: x['division'] is not None) if operation == 'division' else None
                    )),
                    total=total
                ):
                optim.zero_grad()

                # collate batch data
                digits_batch = torch.stack([x['pixels'] for x in batch_data], dim=0).to(config.device)
                digit_labels_batch = torch.stack([x['digit'] for x in batch_data], dim=0).to(config.device)
                summation_labels_batch = torch.tensor([x[operation][0][0] for x in batch_data]).to(config.device)

                # forward pass for all digit predictions
                digits_batch_flat = digits_batch.view(-1, 28*28)
                logits_batch_flat = net(digits_batch_flat)

                logits_batch = logits_batch_flat.view(-1, 2, 10) # (batch_size, 2, 10)

                logits_digit0_batch = logits_batch[:, 0, :] # (batch_size, 10)
                logits_digit1_batch = logits_batch[:, 1, :]

                selected_digit0_batch = config.approx_select(logits_digit0_batch, config.device, temp=config.temp)
                selected_digit1_batch = config.approx_select(logits_digit1_batch, config.device, temp=config.temp)

                # perform summation over soft-selected digit values
                if operation == 'summation':
                    pred_summation_batch = selected_digit0_batch + selected_digit1_batch # (batch_size,)
                elif operation == 'division':
                    pred_summation_batch = selected_digit0_batch / (selected_digit1_batch + 1e-6)
                elif operation == 'multiplication':
                    pred_summation_batch = selected_digit0_batch * selected_digit1_batch
                elif operation == 'subtraction':
                    pred_summation_batch = selected_digit0_batch - selected_digit1_batch
                else:
                    raise ValueError(f'invalid operation: {operation}')

                # compute summation loss
                loss_summation = loss(pred_summation_batch, summation_labels_batch.float())

                # backwards
                loss_summation.backward()
                optim.step()

                # metrics
                pred_digit0_batch = logits_digit0_batch.argmax(dim=-1) # (batch_size,)
                pred_digit1_batch = logits_digit1_batch.argmax(dim=-1)

                acc_digit0 = (pred_digit0_batch == digit_labels_batch[:, 0]).float().mean().item()
                acc_digit1 = (pred_digit1_batch == digit_labels_batch[:, 1]).float().mean().item()

                metrics_epoch['operation'].append(operation)
                metrics_epoch['epoch_idx'].append(epoch_idx)
                metrics_epoch['iter_idx'].append(iter_idx)
                metrics_epoch['loss_summation'].append(loss_summation.item())
                metrics_epoch['correct_digit0'].append(acc_digit0)
                metrics_epoch['correct_digit1'].append(acc_digit1)
            
            epoch_acc_digit0 = sum(metrics_epoch['correct_digit0']) / len(metrics_epoch['correct_digit0'])
            epoch_acc_digit1 = sum(metrics_epoch['correct_digit1']) / len(metrics_epoch['correct_digit1'])
            loss_summation = sum(metrics_epoch['loss_summation']) / len(metrics_epoch['loss_summation'])
            print(f'epoch {epoch_idx}:\tloss={loss_summation:.3f}\tacc_digit0={epoch_acc_digit0:.3f}\tacc_digit1={epoch_acc_digit1:.3f}')

            # validation
            valid_digit_acc, valid_total = get_accuracy_iter(validloader, net, config.device)
            print(f'\tvalid acc={valid_digit_acc:.3f} (n={valid_total})')
            metrics_epoch['valid_digit_acc'] = valid_digit_acc

            all_metrics.append(metrics_epoch)
    
    return all_metrics
