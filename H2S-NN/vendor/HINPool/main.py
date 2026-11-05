import argparse
import os
import random
import pickle
import time
from tqdm import tqdm
import datetime
import statistics
import csv
from pathlib import Path

import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import TUDataset

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import LambdaLR

from utils.utils import *
from utils.EarlyStopping import EarlyStopping
from model.my_gpool import HGPool
from HeteroDataset import HeteroDataset, myDataset

global_train_time_per_epoch = []
print(f'available: {torch.cuda.is_available()}')
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

def arg_parse():
    """
    argument parser
    """
    parser = argparse.ArgumentParser(description="HGPool arguments")
    ################# General Args #################
    parser.add_argument("--out-dir"     , dest="out_dir"                , help="Output Directory name")
    parser.add_argument("--seed"        , dest="seed"       , type=int  , help="training seed")
    parser.add_argument("--repeat"      , dest="repeat"     , type=int  , help="repeat runs")
    parser.add_argument("--train-ratio" , dest="train_ratio", type=float,)
    parser.add_argument("--test-ratio"  , dest="test_ratio" , type=float,)
    parser.add_argument("--num_workers" , dest="n_worker"   , type=int,)

    ################# Data Args #################
    parser.add_argument("--dataset"         , dest="dataset"                    , help="Input Dataset")
    parser.add_argument("--hetero-mode"     , dest="hetero_mode"    , type=int  , help="How heterogeneous will this dataset be")
    parser.add_argument("--num-atom-types"  , dest="num_atom_types" , type=int  , help="TopK node types with k highest number of atoms, 0 for all types")
    parser.add_argument("--hop"             , dest="hop", nargs='+' , type=int  ,)
    
    ################# Training Args #################
    parser.add_argument("--cuda"        , dest="cuda"           , type=int  , help="switch cuda")
    parser.add_argument("--lr"          , dest="lr"             , type=float,)
    parser.add_argument("--weight-decay", dest="weight_decay"   , type=float,)
    parser.add_argument("--warmup-epoch", dest="warmup_epoch"   , type=int,)
    parser.add_argument("--dropout"     , dest="dropout"        , type=float, help="dropout rate")
    parser.add_argument("--patience"    , dest="patience"       , type=float,)
    parser.add_argument("--batch-size"  , dest="batch_size"     , type=int, )
    parser.add_argument("--epochs"      , dest="epoch"          , type=int  , help="num-of-epoch")
    parser.add_argument("--joint"       , dest="joint"          , type=int  , help="Cross layer joint learning")

    ################# Model Args #################
    parser.add_argument("--model"       , dest="model"                   , help="Model")
    parser.add_argument("--backbone"    , dest="backbone"                , help="backbone Model")
    parser.add_argument('--num-heads'   , dest="num_heads"  , type=int,) 
    parser.add_argument('--num-layers'  , dest="num_layers" , type=int,) 
    parser.add_argument("--hidden-dim"  , dest="hidden_dim" , type=int)  
    parser.add_argument("--pool"        , dest="pool"                    , help="Pooling method, e.g. mean")
    parser.add_argument("--pool-ratio"  , dest="pool_ratio" , type=float , nargs='+',)
    parser.add_argument("--fusion"      , dest="fusion"                  , help="Feature fusion of several readouts")
    parser.add_argument("--skip-aggr"   , dest="skip_aggr"               , help="Aggregation type of skip-layer conncetion")
    parser.add_argument("--pool-by"     , dest="pool_by"                 , help="all/type")
    
    ################# Additional Args #################
    parser.add_argument("--use-mixed", dest="use_mixed", type=int, help="Whether to incorporate mixed-type readout in aggregation")
    parser.add_argument("--mlp-score", dest="mlp_score", type=int, help="Whether to use mlp to calculate score in HGPool")
    parser.add_argument("--mul-score", dest="mul_score", type=int, help="Whether to multiply the score and embedding in HGPool")
    parser.add_argument("--norm"     , dest="norm"     , type=int)
    
    ################# Ablation Study Args #################
    parser.add_argument("--do-TA-readout",  dest="do_TA_readout",  type=int, help="Use only mixed-type mean embedding")
    parser.add_argument("--do-cross-layer", dest="do_cross_layer", type=int, help="Use only last layer embedding")
    
    ################# Arguments Default #################
    parser.set_defaults(
        #### General Args ####
        repeat      = 1,
        train_ratio = 0.7,
        test_ratio  = 0.2,
        n_worker    = 0,
        #### Data Args ####
        dataset         = "MUTAG",
        num_atom_types  = 0,
        hetero_mode     = 0,
        hop             = [1],
        #### Training Args ####
        cuda            = 1,
        lr              = 1e-3,
        weight_decay    = 1e-2,
        warmup_epoch    = 60,
        dropout         = 0.3,
        patience        = 100,
        batch_size      = 50,
        epoch           = 1000,
        joint           = 1,
        #### Model Args ####
        model       = "HGPool",
        backbone    = 'Simple',
        num_heads   = 4,
        num_layers  = 3,
        hidden_dim  = 64,
        pool        = 'mean',
        pool_ratio  = [0.9, 0.8, 0.7],
        skip_aggr   = 'mean',
        fusion      = 'concat',
        #### Additional Args ####
        use_mixed   = 0,
        mlp_score   = 0,
        mul_score   = 1,
        norm        = 0,
        #### Ablation Study Args ####
        do_TA_readout  = 0,
        do_cross_layer = 0,
    )
    return parser.parse_args()


def prepare_data(dataset, prog_args, train=False):

    if train:
        shuffle = True
    else:
        shuffle = False
        
    return dgl.dataloading.GraphDataLoader(
        dataset,
        batch_size=prog_args.batch_size,
        shuffle=shuffle,
        num_workers=prog_args.n_worker,
        pin_memory=True,
        generator=torch.Generator(device=device),
    )


def graph_classify_task(prog_args, i, graphs, train_set, val_set, test_set, input_dim, labels, label_dim, full_path):
    print(f'device: {device}')
    sample_graph = graphs[0]  # A heterogeneous graph
    # initialize model
    model = HGPool(prog_args, 
                sample_graph, 
                input_dim, 
                prog_args.hidden_dim, 
                label_dim,
                prog_args.num_layers, 
                prog_args.num_heads, 
                use_norm = True)
        
    print(model)
    print("# of params:", sum(p.numel() for p in model.parameters()))

    model = model.to(device)

    dataset_train, dataset_val, dataset_test = train_set, val_set, test_set, 
    print(dataset_train[0])

    class_weights = None

    train_dataloader = prepare_data(
        dataset_train, prog_args, train=True
    )
    val_dataloader = prepare_data(
        dataset_val, prog_args, train=False
    )
    test_dataloader = prepare_data(
        dataset_test, prog_args, train=False
    )

    early_stopping = train(
        prog_args, i, full_path, sample_graph, train_dataloader, model, class_weights
        , val_dataset=val_dataloader, test_dataset=test_dataloader
    )
    # Find thresholds on train, Try on val, Apply on test
    # probs: List, targets: List
    train_probs, train_targets, _ = evaluate(train_dataloader, model, class_weights, prog_args, early_stopping)
    val_probs,   val_targets,   _ = evaluate(val_dataloader,   model, class_weights, prog_args, early_stopping)
    test_probs,  test_targets,  _ = evaluate(test_dataloader,  model, class_weights, prog_args, early_stopping)

    # Convertion
    train_probs = torch.tensor(train_probs).cpu()
    val_probs   = torch.tensor(val_probs).cpu()
    test_probs  = torch.tensor(test_probs).cpu()

    enz_flag = prog_args.dataset == 'ENZYMES'
    if not enz_flag:
        # ## Brute
        # best_thr = find_best_binary_threshold( \
        #         np.array(train_targets).reshape(-1, 1), np.array(train_probs), accuracy_score)
        # best_thr  = torch.tensor(best_thr).cpu()
        # ## N=50
        best_thrs    = find_best_thresholds( \
                        np.array(train_targets).reshape(-1, 1), np.array(train_probs), accuracy_score, 50)
        best_thrs    = torch.tensor(best_thrs).cpu()
        best_val_acc = 0.0
        best_thr     = None

        for thr in best_thrs:
            temp_acc, temp_thr = validate_thr(val_probs, val_targets, best_val_acc, thr)
            if temp_acc >= best_val_acc:
                best_val_acc = temp_acc
                best_thr     = temp_thr
                
        ## Final cut
        train_outputs = to_labels(train_probs, best_thr).long().cpu().numpy()
        val_outputs   = to_labels(val_probs,   best_thr).long().cpu().numpy()
        test_outputs  = to_labels(test_probs,  best_thr).long().cpu().numpy()
    else:
        train_probs = torch.tensor(train_probs)
        val_probs   = torch.tensor(val_probs)
        test_probs  = torch.tensor(test_probs)

        train_outputs = train_probs.argmax(1).long().cpu().numpy()
        val_outputs   = val_probs.argmax(1).long().cpu().numpy()
        test_outputs  = test_probs.argmax(1).long().cpu().numpy()
            
    train_acc = get_score(train_targets, train_outputs, 'acc')
    val_acc   = get_score(val_targets,   val_outputs,   'acc')
    test_acc  = get_score(test_targets,  test_outputs,  'acc')

    train_auroc = get_score(train_targets, train_probs.cpu(), 'AUROC', enz_flag)
    val_auroc   = get_score(val_targets,   val_probs.cpu(),   'AUROC', enz_flag)
    test_auroc  = get_score(test_targets,  test_probs.cpu(),  'AUROC', enz_flag)
        
    print("Final testing acc {:.2f}%, AUROC: {:.2f}"\
          .format(test_acc * 100, test_auroc * 100))
    
    # Clear space
    os.remove(early_stopping.filename)
    
    return train_acc * 100, train_auroc * 100, \
           val_acc * 100,   val_auroc * 100, \
           test_acc * 100,  test_auroc * 100

def train(args, i, full_path, sample_graph, dataset, model, class_weights
          , val_dataset=None, test_dataset=None):
    
    dataloader = dataset
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    # num_warmup_steps=0,
                    num_warmup_steps=len(dataloader)*args.warmup_epoch,
                    num_training_steps=len(dataloader)*args.epoch)
    
    # Loss by class weight
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    early_stopping = EarlyStopping(patience=args.patience, save_path='checkpoint/checkpoint.pt')
    

        
    # Epoch loss
    train_epoch_loss, train_epoch_acc = np.array([]), np.array([])
    val_epoch_loss, val_epoch_acc = np.array([]), np.array([])
    test_epoch_acc = np.array([])

    # 
    best_best_thr = None

    # Check LR
    lrs = []

    #####################
    for epoch in range(args.epoch):
        # For AUROC calculation
        train_targets = []
        train_outputs = []
        train_probs   = []
        
        begin_time = time.time()
        model.train()
        
        train_total_loss = 0
        print("\nEPOCH ###### {} ######".format(epoch))
        computation_time = 0.0
        # for _, (batch_graph, graph_labels) in enumerate(dataloader):
        for _, (batch_graph, graph_labels) in tqdm(enumerate(dataloader)):

            batch_graph = batch_graph.to(device)
            graph_labels = graph_labels.to(device)
            
            model.zero_grad()
            start = time.time()
            logits = model(batch_graph)
            
            # print(logits)
            loss = criterion(logits, graph_labels.flatten())
            train_total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            lrs.append(scheduler.get_lr()[0])

            # Append outputs part
            train_targets.extend(graph_labels.flatten().cpu().numpy())
            if args.dataset == 'ENZYMES':
                train_probs.extend(F.softmax(logits, dim=1).detach().cpu().numpy().tolist())
            else:
                train_probs.extend(F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1])

            batch_compute_time = time.time() - start
            computation_time += batch_compute_time

        best_thr = None
        # Validation part
        val_probs,  val_targets,  val_loss  = evaluate(val_dataset, model, class_weights, args)
        
        # Testing part
        test_probs, test_targets, test_loss = evaluate(test_dataset, model, class_weights, args)

        enz_flag = args.dataset == 'ENZYMES'

        # Convertion
        train_probs = torch.tensor(train_probs)
        val_probs = torch.tensor(val_probs)
        test_probs = torch.tensor(test_probs)
        
        train_auroc = get_score(train_targets, train_probs.cpu(), 'AUROC', enz_flag)
        val_auroc   = get_score(val_targets,   val_probs.cpu(),   'AUROC', enz_flag)
        test_auroc  = get_score(test_targets,  test_probs.cpu(),  'AUROC', enz_flag)
        
            
        total = len(train_targets)
        
        print(
            "train AUROC: {:.2f}".format(
                train_auroc * 100
            )
        )
        elapsed_time = time.time() - begin_time
        print(
            "loss {:.7f} with epoch time {:.4f} s & computation time {:.4f} s ".format(
                train_total_loss/total, elapsed_time, computation_time
            )
        )
        global_train_time_per_epoch.append(elapsed_time)

        train_epoch_loss = np.append(train_epoch_loss, train_total_loss/total)
        train_epoch_acc = np.append(train_epoch_acc, train_auroc * 100)
        
        # Validation part
        val_epoch_loss = np.append(val_epoch_loss, val_loss)
        val_epoch_acc = np.append(val_epoch_acc, val_auroc * 100)
        print("validation AUROC: {:.2f}%, loss: {:.8f}"\
              .format(val_auroc * 100, val_loss))
        
        # Testing part
        test_epoch_acc = np.append(test_epoch_acc, test_auroc * 100)
        print("Testing AUROC: {:.2f}%"\
              .format(test_auroc * 100))
        
        # Main metric AUROC, stop by AUROC
        # early_stop = early_stopping.step(val_loss, val_acc, model)
        early_stop = early_stopping.step(val_loss, val_auroc, model)
        if early_stop:
            break

    plot_lr(lrs, full_path)
    plot_loss(train_epoch_loss, val_epoch_loss, args, i, full_path)
    plot_acc(train_epoch_acc, val_epoch_acc, test_epoch_acc, args, i, full_path)
    return early_stopping

def evaluate(dataloader, model, class_weights, args, early_stopping=None):
    """
    evaluate function
    """
    if early_stopping is not None:
        early_stopping.load_checkpoint(model)
    
    targets = []
    outputs = []
    probs   = []
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.eval()
    correct_label = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (batch_graph, graph_labels) in tqdm(enumerate(dataloader)):
            batch_graph, graph_labels = batch_graph.to(device), graph_labels.to(device)

            logits = model(batch_graph)

            loss = criterion(logits, graph_labels.flatten())
            total_loss += loss.item()

            # Append outputs
            targets.extend(graph_labels.flatten().cpu().numpy())
            if args.dataset == 'ENZYMES':
                probs.extend(F.softmax(logits, dim=1).detach().cpu().numpy().tolist())
            else:
                probs.extend(F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1])

    total  = len(targets)
    
    return probs, targets, total_loss / total


def main():
    """
    main
    """
    prog_args = arg_parse()
    print(prog_args)

    # Write file ######################################
    filename = prog_args.out_dir
    full_path = os.path.join('results', filename)
    os.makedirs(full_path, exist_ok=True)
    ###################################################

    if prog_args.cuda > 0:
        torch.set_default_device('cuda')
    train_accs, val_accs, test_accs = [], [], []
    train_aucs, val_aucs, test_aucs = [], [], []
    total_times = []
    epoch_times = []
    
    graphs, input_dim, labels, label_dim = my_load_graph(prog_args.dataset, prog_args.num_atom_types)
    myData = myDataset(graphs.copy(), labels, prog_args)

    # File I/O
    index_path_np  = Path(f'dataset/{prog_args.dataset}/index.npz')
    index_path_txt = Path(f'dataset/{prog_args.dataset}/index.txt')

    if index_path_np.exists() and index_path_txt.exists():
        # If exist, load all indices and get subset
        print("Files exist.")
        all_idx     = np.load(index_path_np)
        train_set   = torch.utils.data.Subset(myData, all_idx['train'])
        val_set     = torch.utils.data.Subset(myData, all_idx['val'])
        test_set    = torch.utils.data.Subset(myData, all_idx['test'])
    else:
        # If do not exist, generate indices then write .npz/.txt
        print("Files do not exist.")

        train_size  = int(prog_args.train_ratio * len(graphs))
        test_size   = int(prog_args.test_ratio * len(graphs))
        val_size    = int(len(graphs) - train_size - test_size)
        train_set, val_set, test_set, orig_train_index, orig_val_index, orig_test_index \
            = stratified_split(myData, train_size, val_size, test_size, prog_args.seed
        )

        # Write npz file for fast load
        np.savez(index_path_np, \
                 train=orig_train_index, \
                 val  =orig_val_index, \
                 test =orig_test_index)
        
        # Write txt file for checking
        all_idx = np.load(index_path_np)
        for string in ['train', 'val', 'test']:
            with open(index_path_txt, 'a') as f:
                f.write(string)
                for idx in all_idx[string]:
                    f.write(f' {idx}')
                f.write('\n')
                f.close()
    
    if prog_args.repeat == 1:
        same_seeds(prog_args.seed)
        start_time = time.time()
        train_acc, train_auroc, val_acc, val_auroc, test_acc, test_auroc = \
            graph_classify_task(prog_args, 0, \
                graphs.copy(), train_set, val_set, test_set, \
                input_dim, labels, label_dim, full_path)
        time_per_epoch = sum(global_train_time_per_epoch) / len(global_train_time_per_epoch)
        f = open(full_path +'/' + prog_args.dataset + f'|' + prog_args.model + '.txt', 'a')
        dash_str = '-'*100
        f.write(f'\n{dash_str}\n')
        log = ( f'Backbone:{prog_args.backbone}\n'
                f'ratio:{prog_args.pool_ratio}, '
                f'lr:{prog_args.lr}, '
                f'head:{prog_args.num_heads}, '
                f'hidden:{prog_args.hidden_dim}, '
                f'batch:{prog_args.batch_size}, '
                f'pool by:{prog_args.pool_by}, '
                f'with {prog_args.fusion} of {prog_args.pool} readouts '
                f'and {prog_args.skip_aggr} over graphs.\n'
        )
        f.write(log)
        f.write("test_acc: {:.2f}%, test_AUROC: {:.2f}%, \nwith runtime {:.2f} sec in total and {:.2f} sec per epoch\n"\
                    .format(test_acc, test_auroc, time.time()-start_time, time_per_epoch))
        f.close()
    else:
        for i in range(prog_args.repeat):
            print(f'Running {i+1}/{prog_args.repeat}')
            same_seeds(prog_args.seed + i)
            start_time = time.time()
            train_acc, train_auroc, val_acc, val_auroc, test_acc, test_auroc = \
                graph_classify_task(prog_args, i, \
                    graphs.copy(), train_set, val_set, test_set, \
                    input_dim, labels, label_dim, full_path)
            time_per_epoch = sum(global_train_time_per_epoch) / len(global_train_time_per_epoch)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            train_aucs.append(train_auroc)
            val_aucs.append(val_auroc)
            test_aucs.append(test_auroc)

            total_times.append(time.time() - start_time)
            epoch_times.append(time_per_epoch)
        train_accs_mean, train_accs_std = statistics.mean(train_accs), statistics.stdev(train_accs)
        val_accs_mean,   val_accs_std   = statistics.mean(val_accs),   statistics.stdev(val_accs)
        test_accs_mean,  test_accs_std  = statistics.mean(test_accs),  statistics.stdev(test_accs)

        train_aucs_mean, train_aucs_std = statistics.mean(train_aucs), statistics.stdev(train_aucs)
        val_aucs_mean,   val_aucs_std   = statistics.mean(val_aucs),   statistics.stdev(val_aucs)
        test_aucs_mean,  test_aucs_std  = statistics.mean(test_aucs),  statistics.stdev(test_aucs)

        epoch_time_mean = statistics.mean(epoch_times)
        total_time_mean = statistics.mean(total_times)

    csv_file_path = full_path +'/' + prog_args.dataset + f'|' + prog_args.model + '.csv'
    file_exists = os.path.isfile(csv_file_path)
    print(f'file exists: {file_exists}')

    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            headers = ['use_mixed', 'use_mlp', 'score_multiply', 'layer_norm', 'ratio', 'lr', 
                    'head', 'hidden', 'batch', 'pool_by', 'pool', 'fusion', 'skip_aggr', 
                    'train_acc', 'train_AUROC', 'val_acc', 'val_AUROC', 'test_acc', 'test_AUROC', 
                    'total_time_mean', 'epoch_time_mean']
            writer.writerow(headers)
        
        row_data = [
            prog_args.use_mixed, prog_args.mlp_score, prog_args.mul_score, prog_args.norm,
            prog_args.pool_ratio, prog_args.lr, prog_args.num_heads, prog_args.hidden_dim,
            prog_args.batch_size, prog_args.pool_by, prog_args.pool, prog_args.fusion, prog_args.skip_aggr, 
            f'{train_accs_mean:.2f}±{train_accs_std:.2f}', f'{train_aucs_mean:.2f}±{train_aucs_std:.2f}',
            f'{val_accs_mean:.2f}±{val_accs_std:.2f}', f'{val_aucs_mean:.2f}±{val_aucs_std:.2f}',
            f'{test_accs_mean:.2f}±{test_accs_std:.2f}', f'{test_aucs_mean:.2f}±{test_aucs_std:.2f}',
            f'{total_time_mean:.2f}', f'{epoch_time_mean:.2f}'
        ]

        writer.writerow(row_data)

if __name__ == "__main__":
    main()
