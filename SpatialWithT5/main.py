import argparse

from program_declaration import program_declaration
from reader import DomiKnowS_reader
import torch
import numpy as np
import random
import transformers
from tqdm import tqdm


def train(program, train_set, epoch, lr, cur_device, pmd=True, warmup_epochs=2):
    # optimizer = lambda param: transformers.optimization.Adafactor(param, lr=lr, scale_parameter=False,
    #                                                               relative_step=False)
    optimizer = lambda param: torch.optim.AdamW(param, lr=lr)
    if pmd:
        program.train(train_set, c_warmup_iters=warmup_epochs, train_epoch_num=epoch,
                      Optim=optimizer,
                      device=cur_device)
    else:
        program.train(train_set, train_epoch_num=epoch,
                      Optim=optimizer,
                      device=cur_device)

    program.save("Models/trained_model.pt")


def eval(program, testing_set, cur_device, key='local/argmax'):
    from graph import answer_relations
    count_questions = 0
    count_correct = 0
    for datanode in tqdm(program.populate(testing_set, device=cur_device), "Checking accuracy"):
        preds = []
        labels = []
        for question in datanode.getChildDataNodes():
            count_questions += 1
            check_generate_answer = True
            pred_each = []
            labels_each = []
            expected_answer = set()
            for answer in question.getChildDataNodes():
                label = answer.getAttribute(answer_relations, 'label').item()
                label = ["left", "right", "above", "below", "behind", "front",
                         "near", "far", "disconnected", "touch", "overlap", "coveredby",
                         "inside", "cover", "contain", "<eos>", "<pad>"][label]
                if label == "<eos>":
                    break
                expected_answer.add(label)
                labels_each.append(label)
            for answer in question.getChildDataNodes():
                pred = answer.getAttribute(answer_relations, 'local/argmax').argmax().item()
                pred = ["left", "right", "above", "below", "behind", "front",
                        "near", "far", "disconnected", "touch", "overlap", "coveredby",
                        "inside", "cover", "contain", "<eos>", "<pad>"][pred]
                pred_each.append(pred)
                if pred == "<eos>":
                    break
                if pred == "<pad>":
                    continue
                if pred not in expected_answer:
                    check_generate_answer = False
                    break
                expected_answer.remove(pred)

            if len(expected_answer) > 0:
                check_generate_answer = False

            count_correct += int(check_generate_answer)
            preds.append(pred_each)
            labels.append(labels_each)
        # print("preds: {:}, label: {:}".format(preds, labels))

    # print("Acc:", , "%")
    return count_correct / count_questions * 100


def set_seed():
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)


def main(args):
    set_seed()

    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'

    train_data = DomiKnowS_reader("DataSet/train_FR_v3.json", "FR",
                                  size=args.train_size,
                                  upward_level=12,
                                  augmented=True,
                                  batch_size=args.batch_size,
                                  STEPGAME_status=None)

    test_data = DomiKnowS_reader("DataSet/test.json", "FR",
                                 size=args.test_size,
                                 augmented=False,
                                 batch_size=args.batch_size,
                                 STEPGAME_status=None,
                                 )

    dev_data = DomiKnowS_reader("DataSet/dev.json", "FR",
                                size=args.test_size,
                                augmented=False,
                                batch_size=args.batch_size,
                                STEPGAME_status=None,
                                )

    program = program_declaration(device=cur_device, pmd=args.pmd, beta=args.beta, constraints=args.constraints)
    if args.loaded_file is not None:
        program.load("Models/" + args.loaded_file,
                     map_location={'cuda:0': cur_device, 'cuda:1': cur_device, 'cuda:2': cur_device,
                                   'cuda:3': cur_device, 'cuda:4': cur_device, 'cuda:5': cur_device})

        # EVAL
    else:
        # print("Training with no constraint loss")
        train(program, train_data, epoch=args.epoch, lr=args.lr, cur_device=cur_device, warmup_epochs=0, pmd=args.pmd)
    print(f"Acc seen: {eval(program, train_data, cur_device=cur_device)}")
    print(f"Acc unseen: {eval(program, test_data, cur_device=cur_device)}")

    if args.compare_constraints:
        print("Loading the same setting without constraint learning")
        set_seed()
        program = program_declaration(device=cur_device, pmd=args.pmd, beta=0, constraints=True)
        train(program, train_data, epoch=args.epoch, lr=args.lr, cur_device=cur_device, warmup_epochs=0, pmd=args.pmd)
        print(f"Acc seen: {eval(program, train_data, cur_device=cur_device)}")
        print(f"Acc unseen: {eval(program, test_data, cur_device=cur_device)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SpaRTUN Rules Base")
    parser.add_argument("--epoch", dest="epoch", type=int, default=1)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-5)
    parser.add_argument("--cuda", dest="cuda", type=int, default=0)
    parser.add_argument("--test_size", dest="test_size", type=int, default=10)
    parser.add_argument("--train_size", dest="train_size", type=int, default=10)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--pmd", dest="pmd", type=bool, default=False)
    parser.add_argument("--beta", dest="beta", type=float, default=0.5)
    parser.add_argument("--loaded_file", dest="loaded_file", type=str, default=None)
    parser.add_argument("--constraints", dest="constraints", type=bool, default=False)
    parser.add_argument("--compare_constraints", dest="compare_constraints", type=bool, default=False)
    args = parser.parse_args()
    main(args)
