import random
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import argparse
from model import MLPClassifier
from dataset import PE_perc_dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os 
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import csv
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, f1_score

def setup(rank, world_size): # to set up the process group
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, world_size, dataset, batch_size=4, pin_memory = False, num_workers=4): # split the total indices of dataset into world_size parts
    sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
        drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

def cleanup(): # dismiss the process group after running
    dist.destroy_process_group()

def _safe_div(a, b, eps=1e-12): # to prevent nan
    return float(a) / float(b + eps)

amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def train(rank, world_size, args, ddp_model_parallel, training_Set, valid_Set, optimizer, loss_function):
    best_loss = 1000
    best_auc = 0
    trigger_times = 0
    trainloader = prepare(rank, world_size, training_Set, batch_size=args.batch_size)
    validloader = DataLoader(valid_Set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Rank {rank} Training")
        ddp_model_parallel.train()
        trainloader.sampler.set_epoch(epoch)
        loss_total = 0
        
        # Use tqdm only on the main process for clean output
        if rank == 0:
            pbar = tqdm(trainloader, desc=f"Epoch {epoch+1} Training")
        else:
            pbar = trainloader

        for i_batch, data in enumerate(pbar):
            input1, input2, label = data[0].to(rank), data[1].to(rank), data[2].to(rank) # this is for inhale + exhale
            label = label.to(torch.float32)
            input1 = input1.squeeze()
            input2 = input2.squeeze()
            if input1.dim() == 1:
                input1 = input1.unsqueeze(0)
                input2 = input2.unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)
            inputs = torch.cat((input1, input2), dim=1) # concatenate inhale and exhale inputs
            output = ddp_model_parallel(inputs)
            output = output.squeeze(1)
            loss = loss_function(output.float(), label.float())
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
        if rank == 0:
            print('training loss:', loss_total / len(trainloader)) 
        # get validation loss
        if dist.get_rank() == 0:
            current_loss, current_auc = validation(ddp_model_parallel, validloader, loss_function, rank)
            print('The validation Loss:', current_loss)
            print('The validation AUC:', current_auc)
            if current_auc < best_auc: # if auc not improved, increase patience counter
                trigger_times += 1

                if trigger_times >= args.patience:
                    print('Early stopping!\nStart to test process.')
                    break
            elif current_auc > best_auc: # if auc is improved, save the model
                    trigger_times=0
                    best_loss = current_loss
                    best_auc = current_auc
                    print('rank0, best update:', best_loss, ', best auc:', best_auc, 'model saved')
                    torch.save(ddp_model_parallel.state_dict(), args.save_path)
    
            else: # auc remains the same, depends on the loss
                if current_loss <= best_loss: # if loss is improved, save the model
                    trigger_times=0
                    best_loss = current_loss
                    best_auc = current_auc
                    print('rank0, best update:', best_loss, ', best auc:', best_auc, 'model saved')
                    torch.save(ddp_model_parallel.state_dict(), args.save_path)
                else: # if loss is not improved, increase patience counter
                    trigger_times += 1
                    if trigger_times >= args.patience:
                        print('Early stopping!\nStart to test process.')
                        break
    if rank == 0:
        print("[Rank 0] Training complete.", flush=True)


def validation(model, valid_loader, loss_function, device):
    print('validation process')
    model.eval()
    loss_total = 0
    labels = torch.tensor([]).to(device)
    outputs = torch.tensor([]).to(device)

    # Test validation data
    with torch.no_grad():
        for i_batch, data in enumerate(valid_loader, 0):
            input1, input2, label = data[0].to(device), data[1].to(device), data[2].to(device) # this is for inhale + exhale
            label = label.to(torch.float32)
            input1 = input1.squeeze()
            input2 = input2.squeeze()
            if input1.dim() == 1:
                input1 = input1.unsqueeze(0)
                input2 = input2.unsqueeze(0)
            inputs = torch.cat((input1, input2), dim=1) # concatenate inhale and exhale inputs
            output = model(inputs)
            output = output.squeeze(1)
            labels = torch.cat((labels, label))
            outputs = torch.cat((outputs, output))
            
            loss = loss_function(output, label) 
            loss_total += loss.item()
        print('outputs and labels:', outputs, '//////', labels)

    return loss_total / len(valid_loader), roc_auc_score(labels.cpu(), outputs.cpu())

def test(args, test_loader, loss_function, device):
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")] if args.hidden_dims else []
    model = MLPClassifier(input_dim=202, hidden_dims=hidden_dims, num_classes=1, dropout = args.dropout_rate)
    model_path = args.save_path
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    loss_total = 0
    labels = torch.tensor([]).to(device)
    outputs = torch.tensor([]).to(device)

    # Test validation data
    with torch.no_grad():
        for i_batch, data in enumerate(test_loader, 0):
            input1, input2, label = data[0].to(device), data[1].to(device), data[2].to(device) # this is for inhale + exhale
            label = label.to(torch.float32)
            input1 = input1.squeeze()
            input2 = input2.squeeze()
            if input1.dim() == 1:
                input1 = input1.unsqueeze(0)
                input2 = input2.unsqueeze(0)
            inputs = torch.cat((input1, input2), dim=1) # concatenate inhale and exhale inputs
            output = model(inputs)
            output = output.squeeze(1)
            labels = torch.cat((labels, label))
            outputs = torch.cat((outputs, output))
            
            loss = loss_function(output, label) 
            loss_total += loss.item()
        print('outputs and labels:', outputs, '//////', labels)
        test_loss = loss_total / len(test_loader)
        test_auc = roc_auc_score(labels.cpu(), outputs.cpu())

        # getting confusion matrix
        p_test = torch.sigmoid(outputs)
        pred = (p_test >= 0.5).astype(np.int64) # predictions

        cm = confusion_matrix(labels, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = _safe_div(tp, tp + fn)   # recall
        specificity = _safe_div(tn, tn + fp)
        ppv = _safe_div(tp, tp + fp)           # precision
        npv = _safe_div(tn, tn + fn)
        f1 = float(f1_score(labels, pred, zero_division=0))


        print('outputs and labels:', outputs, '//////', labels)
        print('testing loss:', test_loss)
        print('testing AUC:', test_auc)
        row = [
            "test", "MLP_IE", args.current_fold, test_auc, sensitivity, specificity, ppv, npv, f1,
        ]

        # without unsupervised learning
        with open('./results_10folds.csv','a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    return loss_total / len(test_loader), roc_auc_score(labels.cpu(), outputs.cpu())



def main(rank, world_size, args, rskf, dataset, labels):
    setup(rank, world_size)
    fold_count = args.N_rounds
    c = 0
    for train_fold_ind, test_fold_ind in rskf.split(np.zeros(len(dataset)), labels):
        if c < fold_count:
            c += 1  # already trained
            continue

        if rank == 0:
            print("-" * 50)
            print(f"--- Starting Fold {c} ---")
        args.save_path = os.path.join(
            './', 
            f'model_MLP_IE_fold{c}.pt'
        )

        if rank == 0:
            print(f"Model for this fold will be saved to: {args.save_path}")

        # Create subsets for this specific fold
        train_fold_subset = Subset(dataset, train_fold_ind)
        Valid_number = int(len(train_fold_subset) * 0.1)
        Train_number = len(train_fold_subset) - Valid_number
        training_Set, valid_Set = torch.utils.data.random_split(train_fold_subset, [Train_number, Valid_number], 
                    generator=torch.Generator().manual_seed(42))
        test_fold_subset = Subset(dataset, test_fold_ind)

        # set up the model
        hidden_dims = [int(x) for x in args.hidden_dims.split(",")] if args.hidden_dims else []
        model = MLPClassifier(input_dim=202, hidden_dims=hidden_dims, num_classes=1, dropout = args.dropout_rate)
        model_parallel = model.to(rank)
        ddp_model_parallel = DDP(model_parallel, device_ids=[rank])

        # the training parameters
        optimizer = torch.optim.AdamW(ddp_model_parallel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()

        if rank == 0:
            print('Training start!')
        train(rank, world_size, args, ddp_model_parallel, training_Set, valid_Set, optimizer, criterion)
        print('training process finished!, rank:', rank)

        if rank == 0:
            print('Training complete!, moving to test process.')
            test(args, test_fold_subset, criterion, rank)
        # break # just run one fold for each time
        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP models")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="dropout rate")
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--patience", default=100, type=int, help="early stopping patience")
    parser.add_argument("--epochs", default=5000, type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size for training")
    parser.add_argument("--N_rounds", required=True, type=int, help="Current number for training")
    parser.add_argument("--hidden_dims", default="256, 128", type=str, help="hidden dimensions")
    args = parser.parse_args()


    # get the dataset
    img_dir = '/Data/PE_percentiles_normalized'
    label_file = '/Data/PE_percentiles_normalized/label.csv'
    csv_data = pd.read_csv(label_file, header=None)
    labels = csv_data.iloc[:,1].to_numpy()
    dataset = PE_perc_dataset(label_file=label_file, img_dir=img_dir)
    Data_Number = dataset.__len__()
    print(f"Total dataset size: {Data_Number} samples")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=seed)

    world_size = 4
    mp.spawn(
            main,
            args=(world_size, args, rskf, dataset, labels),
            nprocs = world_size,
            join=True
        )

