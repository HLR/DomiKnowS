import torch
from model import Net
from utils import batch_iterator
from typing import Callable
from tqdm import tqdm

from training import TrainingConfig, get_loss_fn, apply_operation, get_accuracy_iter


def expected_sat_logprobs(
        probs_0: torch.Tensor, # (10,)
        probs_1: torch.Tensor, # (10,)
        n_samples: int = 100,
        satisfies_func: Callable[[torch.Tensor, torch.Tensor], bool] = None
    ) -> torch.Tensor:
    '''
    Computes: -E_{x ~ p_0, y ~ p_1}[log(p_0(x) * p_1(y)) I[operation(x, y) = label]]
    '''
    
    samples_0 = torch.multinomial(probs_0, n_samples, replacement=True) # (n_samples,)
    samples_1 = torch.multinomial(probs_1, n_samples, replacement=True) # (n_samples,)

    satisfying_logprobs = 0
    for i in range(n_samples):
        sample_0 = samples_0[i]
        sample_1 = samples_1[i]

        if satisfies_func(sample_0, sample_1):
            satisfying_logprobs += torch.log(probs_0[sample_0] * probs_1[sample_1])

    return satisfying_logprobs / n_samples

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

    loss = get_loss_fn(config.loss)

    batch_size = config.batch_size
    num_epochs = config.num_epochs
    all_metrics = []

    for operation in config.operations:
        # perform a training run for this operation
        net = Net().to(config.device)
        optim = torch.optim.Adam(net.parameters(), lr=config.lr)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim,
        #     T_max=config.num_epochs,
        #     eta_min=0
        # )

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

                probs_digit0_batch = torch.nn.functional.softmax(logits_digit0_batch, dim=-1) # (batch_size, 10)
                probs_digit1_batch = torch.nn.functional.softmax(logits_digit1_batch, dim=-1)
                
                sat_f_batch = lambda i: (
                    lambda x, y: apply_operation(operation, x, y) == summation_labels_batch[i]
                )

                # loss
                # for each instance in the batch, compute:
                # -E_{x ~ p_0, y ~ p_1}[log(p_0(x) * p_1(y)) I[operation(x, y) = label]]
                loss_summation = -sum([
                    expected_sat_logprobs(
                        probs_digit0_batch[i],
                        probs_digit1_batch[i],
                        satisfies_func=sat_f_batch(i)
                    )
                    for i in range(len(batch_data))
                ]) / len(batch_data)

                # backwards
                loss_summation.backward()

                optim.step()
                # scheduler.step()

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
            valid_stats = get_accuracy_iter(
                validloader,
                net,
                config.device,
                operation,
                loss_fn=loss
            )

            valid_digit_acc = valid_stats['accuracy']
            valid_total = valid_stats['num_samples']
            valid_loss = valid_stats['loss']

            print(f'\tvalid acc={valid_digit_acc:.3f}\tvalid_loss={valid_loss:.3f} (n={valid_total})')

            metrics_epoch['valid_digit_acc'] = valid_digit_acc
            metrics_epoch['valid_loss'] = valid_loss

            all_metrics.append(metrics_epoch)
    
    return all_metrics
