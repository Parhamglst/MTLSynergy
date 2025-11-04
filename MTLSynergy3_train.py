#!/usr/bin/env python3

import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import profiler
from utils.tools import CategoricalCrossEntropyLoss, EarlyStopping, GradNormController
from torch.nn import MSELoss
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
    RobertaModel,
    RobertaTokenizer,
)
import argparse

from static.constant import (
    CellAE_SaveBase,
    CellAE_OutputDim,
    MTLSynergy_InputDim,
    MTLSynergy_SaveBase,
    MTLSynergy2_Result,
)
from Models import MTLSynergy3, CellLineAE
from Dataset import mainDataset
import argparse
from transformers import AutoModel
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


CCLE_DRUGCOMB_FILTERED = "./data/CCLE_DrugComb_filtered.csv"
ONEIL_ALMANAC_CCLE_FILTERED_NORMALIZED = "./data/oneil_almanac_ccle_filtered_normalized.csv"
ONEIL_ALMANAC_EMBEDDINGS = "./data/oneil_almanac_embeddings.pt"
NUM_TASKS = 2

hyper_parameters_candidate = [
    {"learning_rate": 0.0001, "hidden_neurons": [4096, 2048, 1024, 1024]},
    {"learning_rate": 0.0001, "hidden_neurons": [2048, 1024, 512, 512]},
]

patience = 3
epochs = 25
batch_size = 2


def cleanup_ddp():
    dist.destroy_process_group()

def mean_pooling(last_hidden_state, attention_mask):
    # Expand mask to (batch_size, seq_len, 1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    # Sum of embeddings for non-masked tokens
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    # Count of non-masked tokens
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    # Mean pooling
    return sum_embeddings / sum_mask

# ema_mean_yhat = torch.zeros(1, device=device)
# ema_mean_y = torch.zeros(1, device=device)
# ema_var_yhat = torch.ones(1, device=device)
# ema_var_y = torch.ones(1, device=device)
# momentum = 0.99

# def corr_loss(y_hat, y, eps=1e-6):
#     ema_mean_yhat = momentum * ema_mean_yhat + (1 - momentum) * y_hat.mean()
#     ema_mean_y = momentum * ema_mean_y + (1 - momentum) * y.mean()
#     ema_var_yhat = momentum * ema_var_yhat + (1 - momentum) * y_hat.var(
#         unbiased=False
#     )
#     ema_var_y = momentum * ema_var_y + (1 - momentum) * y.var(unbiased=False)

#     y_hat_c = y_hat - ema_mean_yhat
#     y_c = y - ema_mean_y
#     num = (y_hat_c * y_c).mean()
#     denom = (ema_var_yhat * ema_var_y).sqrt() + eps
#     rho = num / denom
#     return (1 - rho)[0]


def train(
    model,
    drug_embeddings,
    chemBERTa_model,
    cellLineEncoder,
    train_loader,
    optimizer,
    mse,
    log_vars,
    device
):
    model.train()
    chemBERTa_model.train()
    cellLineEncoder.train()
    total_loss_epoch = np.zeros(NUM_TASKS)
    for batch in train_loader:
        (d_row, conc_row, d_col, conc_col, c_exp), (viability) = batch

        c_exp = c_exp.to(device).float()
        viability = viability.to(device).float()

        input_ids_row_list, attn_mask_row_list, token_type_ids_row_list = [], [], []
        input_ids_col_list, attn_mask_col_list, token_type_ids_col_list = [], [], []

        for drug_row, drug_col in zip(d_row, d_col):
            # Process drug_row
            row_embedding = drug_embeddings[drug_row]
            input_ids_row_list.append(row_embedding["input_ids"].squeeze(0))
            attn_mask_row_list.append(row_embedding["attention_mask"].squeeze(0))
            token_type_ids_row_list.append(torch.zeros_like(row_embedding["input_ids"].squeeze(0)))

            # Process drug_col, with fallback to drug_row if NaN
            col_embedding = (
                row_embedding if drug_col == "nan" else drug_embeddings[drug_col]
            )
            input_ids_col_list.append(col_embedding["input_ids"].squeeze(0))
            attn_mask_col_list.append(col_embedding["attention_mask"].squeeze(0))
            token_type_ids_col_list.append(torch.zeros_like(col_embedding["input_ids"].squeeze(0)))

        input_ids_row = torch.stack(input_ids_row_list).to(device)
        attn_mask_row = torch.stack(attn_mask_row_list).to(device)
        input_ids_col = torch.stack(input_ids_col_list).to(device)
        attn_mask_col = torch.stack(attn_mask_col_list).to(device)
        token_type_ids_row = torch.stack(token_type_ids_row_list).to(device)
        token_type_ids_col = torch.stack(token_type_ids_col_list).to(device)
        conc_row = conc_row.to(device).float().unsqueeze(1)
        conc_col = conc_col.to(device).float().unsqueeze(1)

        
        optimizer.zero_grad()

        # Feed into ChemBERTa
        d_row_embeddings = chemBERTa_model(
            input_ids=input_ids_row, attention_mask=attn_mask_row, token_type_ids=token_type_ids_row
        ).last_hidden_state[:, 0, :]
        d_col_embeddings = chemBERTa_model(
            input_ids=input_ids_col, attention_mask=attn_mask_col, token_type_ids=token_type_ids_col
        ).last_hidden_state[:, 0, :]
        c_embeddings = cellLineEncoder(c_exp)

        pred_mono, pred_combo = model(d_row_embeddings, conc_row, d_col_embeddings, conc_col, c_embeddings)

        losses = [
            mse(pred_mono, viability),
            mse(pred_combo, viability),
        ]

        # Weighted multi-task loss
        losses_tensor = torch.stack(losses)
        precision = torch.exp(-log_vars)
        weighted_losses = precision * losses_tensor + 0.5 * log_vars
        supervised_loss = torch.sum(weighted_losses)

        # Boundary-consistency
        mono_mask = torch.tensor([dc == "nan" for dc in d_col], device=device)
        combo_mask = ~mono_mask

        consistency_losses = []
        if mono_mask.any():
            consistency_losses.append(mse(pred_combo[mono_mask], pred_mono[mono_mask]))
        if combo_mask.any():
            consistency_losses.append(mse(pred_mono[combo_mask], pred_combo[combo_mask]))

        BC_loss = torch.mean(torch.stack(consistency_losses)) if consistency_losses else torch.tensor(0.0, device=device)

        # Final loss
        total_loss = supervised_loss + 0.1 * BC_loss

        total_loss.backward()
        optimizer.step()

        total_loss_epoch += [l.item() for l in losses]
        #prof.step()
    if device == torch.device("cuda:0"):
        weights = torch.exp(-log_vars.detach())
        print(f"Per Task Precisions: {', '.join([f'{w:.5f}' for w in weights.cpu().numpy()])}")

    #prof.stop()

    return total_loss_epoch / len(train_loader)


def evaluate(
    model,
    drug_embeddings,
    chemBERTa_model,
    cellLineEncoder,
    val_loader,
    mse,
    log_vars,
    device
):
    model.eval()
    total_loss_epoch = np.zeros(NUM_TASKS)
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            (d_row, conc_row, d_col, conc_col, c_exp), (viability) = batch

            true_var = {
                "Viability": np.var(viability.detach().cpu().numpy()),
            }
            c_exp = c_exp.to(device).float()
            viability = viability.to(device).float()

            input_ids_row_list, attn_mask_row_list = [], []
            input_ids_col_list, attn_mask_col_list = [], []
            token_type_ids_row_list = []
            token_type_ids_col_list = []

            for drug_row, drug_col in zip(d_row, d_col):
                # Process drug_row
                row_embedding = drug_embeddings[drug_row]
                input_ids_row_list.append(row_embedding["input_ids"].squeeze(0))
                attn_mask_row_list.append(row_embedding["attention_mask"].squeeze(0))
                token_type_ids_row_list.append(torch.zeros_like(row_embedding["input_ids"].squeeze(0)))

                # Process drug_col, with fallback to drug_row if NaN
                col_embedding = (
                    row_embedding if drug_col == "nan" else drug_embeddings[drug_col]
                )
                input_ids_col_list.append(col_embedding["input_ids"].squeeze(0))
                attn_mask_col_list.append(col_embedding["attention_mask"].squeeze(0))
                token_type_ids_col_list.append(torch.zeros_like(col_embedding["input_ids"].squeeze(0)))

            input_ids_row = torch.stack(input_ids_row_list).to(device)
            attn_mask_row = torch.stack(attn_mask_row_list).to(device)
            input_ids_col = torch.stack(input_ids_col_list).to(device)
            attn_mask_col = torch.stack(attn_mask_col_list).to(device)
            token_type_ids_row = torch.stack(token_type_ids_row_list).to(device)
            token_type_ids_col = torch.stack(token_type_ids_col_list).to(device)
            conc_row = conc_row.to(device).float().unsqueeze(1)
            conc_col = conc_col.to(device).float().unsqueeze(1)



            # Feed into ChemBERTa
            d_row_embeddings = chemBERTa_model(
                input_ids=input_ids_row, attention_mask=attn_mask_row, token_type_ids=token_type_ids_row
            ).last_hidden_state[:, 0, :]
            d_col_embeddings = chemBERTa_model(
                input_ids=input_ids_col, attention_mask=attn_mask_col, token_type_ids=token_type_ids_col
            ).last_hidden_state[:, 0, :]

            c_embeddings = cellLineEncoder(c_exp)

            pred_mono, pred_combo = model(d_row_embeddings, conc_row, d_col_embeddings, conc_col, c_embeddings)

            losses = [
                mse(pred_mono, viability),
                mse(pred_combo, viability),
            ]

            # Weighted multi-task loss
            losses_tensor = torch.stack(losses)
            precision = torch.exp(-log_vars)
            weighted_losses = precision * losses_tensor + 0.5 * log_vars
            supervised_loss = torch.sum(weighted_losses)

            # Boundary-consistency
            mono_mask = torch.tensor([dc == "nan" for dc in d_col], device=device)
            combo_mask = ~mono_mask

            consistency_losses = []
            if mono_mask.any():
                consistency_losses.append(mse(pred_combo[mono_mask], pred_mono[mono_mask]))
            if combo_mask.any():
                consistency_losses.append(mse(pred_mono[combo_mask], pred_combo[combo_mask]))

            BC_loss = torch.mean(torch.stack(consistency_losses)) if consistency_losses else torch.tensor(0.0, device=device)

            # Final loss
            loss = supervised_loss + 0.1 * BC_loss
            total_loss_epoch += [l.item() for l in losses]
            total_loss += loss.item()
            pred_var = {
                "Mono": np.var(pred_mono.detach().cpu().numpy()),
                "Combo": np.var(pred_combo.detach().cpu().numpy()),
            }
            print(
                f"Validation - Mono Var: {true_var['Mono']:.4f}, Predicted Var: {pred_var['Mono']:.4f}, "
                f"Combo Var: {true_var['Combo']:.4f}, Predicted Var: {pred_var['Combo']:.4f}, "
            )

    return total_loss_epoch / len(val_loader), total_loss / len(val_loader)



def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    is_main = (rank == 0)
    if is_main:
        print("CUDA available:", torch.cuda.is_available())
        print("Device count:", torch.cuda.device_count())
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    def ddp_reduce_mean(x: float) -> float:
        t = torch.tensor([x], dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / world_size).item()

    parser = argparse.ArgumentParser(
    description="Train MTLSynergy3 model with nested cross-validation."
    )
    parser.add_argument(
        "--weight_dir", type=str, required=True, help="Directory to save model weights."
    )
    args = parser.parse_args()
    MTLSynergy_SaveBase = args.weight_dir + "/"

    if is_main:
        print(f"Running on rank {rank} with world size {world_size}")
    
    
    kf_outer = KFold(n_splits=3, shuffle=True, random_state=42)
    outer_results = []

    drugcomb_df = pd.read_csv(
        ONEIL_ALMANAC_CCLE_FILTERED_NORMALIZED,
        delimiter=",",
        dtype={
            "drug_row": str,
            "conc_row": float,
            "drug_col": str,
            "conc_col": float,
            "cell_line_name": str,
            "viability": float,
        },
    )

    cell_line_df = pd.read_csv(CCLE_DRUGCOMB_FILTERED, index_col=0)
    drug_embeddings = torch.load(ONEIL_ALMANAC_EMBEDDINGS, map_location=device)

    # Load CellLineAE
    cell_path = CellAE_SaveBase + str(CellAE_OutputDim) + ".pth"


    for outer_fold, (train_val_idx, test_idx) in enumerate(kf_outer.split(drugcomb_df)):
        if is_main:
            print(f"Outer Fold {outer_fold+1}")

        train_val_df = drugcomb_df.iloc[train_val_idx]
        test_df = drugcomb_df.iloc[test_idx]

        hp_i = 0
        result_per_hp = []
        outer_save_path = f"{MTLSynergy_SaveBase}fold_{outer_fold}.pth"
        for hyper_parameters in hyper_parameters_candidate:
            hidden_neurons = hyper_parameters["hidden_neurons"]
            if is_main:
                print(
                    "--- Hyper parameters " + str(hp_i) + ":" + str(hyper_parameters) + " ---"
                )

            # Set up 4-Fold Inner Cross Validation
            kf_inner = KFold(n_splits=3, shuffle=True, random_state=42)
            inner_results = []
            for inner_fold, (train_idx, val_idx) in enumerate(kf_inner.split(train_val_df)):
                inner_save_path = (
                    f"{MTLSynergy_SaveBase}inner_fold_{outer_fold}_{inner_fold}.pth"
                )
                if is_main:
                    print(f"Inner Fold {inner_fold+1}")

                train_subset = train_val_df.iloc[train_idx]
                val_subset = train_val_df.iloc[val_idx]

                train_dataset = mainDataset(train_subset, cell_line_df)
                val_dataset = mainDataset(val_subset, cell_line_df)

                train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=12
                )
                val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=12
                )

                # Initialization of model, optimizer, and loss functions
                chemberta_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", attn_implementation="eager").to(device)
                cellLineAE = CellLineAE(output_dim=CellAE_OutputDim).to(device)
                cellLineAE.load_state_dict(torch.load(cell_path, map_location=device))
                for param in cellLineAE.parameters():
                    param.requires_grad = True
                for param in chemberta_model.parameters():
                    param.requires_grad = True
                model = MTLSynergy3(hidden_neurons, input_dim=MTLSynergy_InputDim + 2).to(device)
                # model = torch.compile(model)
                # chemberta_model = torch.compile(chemberta_model)
                # cellLineAE = torch.compile(cellLineAE)
                
                # Wrap models in DDP
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
                chemberta_model = torch.nn.parallel.DistributedDataParallel(chemberta_model, device_ids=[local_rank], find_unused_parameters=True)
                cellLineAE = torch.nn.parallel.DistributedDataParallel(cellLineAE, device_ids=[local_rank], find_unused_parameters=True)

                model_es = EarlyStopping(patience=patience)
                log_vars = torch.nn.Parameter(torch.zeros(NUM_TASKS, device=device))
                optimizer = AdamW(
                    [
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if "drug_cell_line_layer" in n
                                or "combo_layer" in n
                                or "mono_layer" in n
                            ],
                            "lr": hyper_parameters["learning_rate"],
                        },
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if "combo_out" in n
                                or "mono_out" in n
                            ],
                            "lr": hyper_parameters["learning_rate"] * 0.3,
                        },
                        {
                            "params": cellLineAE.parameters(),
                            "lr": hyper_parameters["learning_rate"] * 0.1,
                        },
                        {
                            "params": log_vars,
                            "lr": hyper_parameters["learning_rate"],
                        },
                        {
                            "params": chemberta_model.parameters(),
                            "lr": hyper_parameters["learning_rate"] * 0.1,
                        },
                    ]
                )

                mse = MSELoss(reduction="mean").to(device)

                best_val_loss = float("inf")
                inner_val_loss = []
                for epoch in range(epochs):
                    train_sampler.set_epoch(epoch)
                    train_loss = train(
                        model,
                        drug_embeddings,
                        chemberta_model,
                        cellLineAE.module.encoder,
                        train_loader,
                        optimizer,
                        mse,
                        log_vars,
                        device
                    )
                    val_loss_list, val_loss = evaluate(
                        model,
                        drug_embeddings,
                        chemberta_model,
                        cellLineAE.module.encoder,
                        val_loader,
                        mse,
                        log_vars,
                        device
                    )
                    if is_main:
                        print(
                            f"\tEpoch {epoch+1}:\n\t\tTrain Loss: {train_loss}\n\t\tVal Loss: {val_loss_list}, Weighted Val Loss = {val_loss}",
                            flush=True,
                        )
                    if is_main:
                        model_es(val_loss, model, chemberta_model, cellLineAE, inner_save_path, local_rank)

                    # Broadcast early_stop flag to all ranks
                    stop_tensor = torch.tensor([1 if model_es.early_stop else 0], device=device)
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = stop_tensor.item() == 1

                    dist.barrier()
                    if should_stop or epoch == epochs - 1:
                        best_val_loss = model_es.best_loss
                        if is_main:
                            inner_results.append(best_val_loss)
                        break

            dist.barrier()
            if is_main:
                result_per_inner_fold_mean = np.array(inner_results).mean()
                result_per_hp.append(result_per_inner_fold_mean)
                hp_i += 1
            torch.cuda.empty_cache()
        if is_main:
            best_hp_i = np.array(result_per_hp).argmin()
            best_hp_i_tensor = torch.tensor([best_hp_i], dtype=torch.long, device=device)
        else:
            best_hp_i_tensor = torch.empty(1, dtype=torch.long, device=device)
        dist.broadcast(best_hp_i_tensor, src=0)
        best_hp_i = best_hp_i_tensor.item()
        best_hp = hyper_parameters_candidate[best_hp_i]
        if is_main:
            print("--------- Best parameters: " + str(best_hp) + " ---------")
        train_subset, val_subset = train_test_split(
            train_val_df, test_size=0.2, random_state=42
        )

        final_train_dataset = mainDataset(train_subset, cell_line_df)
        final_val_dataset = mainDataset(val_subset, cell_line_df)
        final_test_dataset = mainDataset(test_df, cell_line_df)

        final_train_sampler = DistributedSampler(final_train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        final_val_sampler = DistributedSampler(final_val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        final_test_sampler = DistributedSampler(final_test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        final_train_loader = DataLoader(
            final_train_dataset, batch_size=batch_size, sampler=final_train_sampler, pin_memory=True, num_workers=12
        )
        final_val_loader = DataLoader(
            final_val_dataset, batch_size=batch_size, sampler=final_val_sampler, pin_memory=True, num_workers=12
        )
        final_test_loader = DataLoader(
            final_test_dataset, batch_size=batch_size, sampler=final_test_sampler, pin_memory=True, num_workers=12
        )

        # Retrain model using the best hyperparameters
        chemberta_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", attn_implementation="eager").to(device)
        cellLineAE = CellLineAE(output_dim=CellAE_OutputDim).to(device)
        cellLineAE.load_state_dict(torch.load(cell_path, map_location=device))
        for param in cellLineAE.parameters():
            param.requires_grad = True
        for param in chemberta_model.parameters():
            param.requires_grad = True
        model = MTLSynergy3(best_hp["hidden_neurons"], input_dim=MTLSynergy_InputDim + 2).to(device)
        model_es = EarlyStopping(patience=patience)
        
        # model = torch.compile(model)
        # chemberta_model = torch.compile(chemberta_model)
        # cellLineAE = torch.compile(cellLineAE)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        chemberta_model = torch.nn.parallel.DistributedDataParallel(chemberta_model, device_ids=[local_rank], find_unused_parameters=True)
        cellLineAE = torch.nn.parallel.DistributedDataParallel(cellLineAE, device_ids=[local_rank], find_unused_parameters=True)
        log_vars = torch.nn.Parameter(torch.zeros(NUM_TASKS, device=device))

        optimizer = AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "drug_cell_line_layer" in n
                        or "combo_layer" in n
                        or "mono_layer" in n
                    ],
                    "lr": best_hp["learning_rate"],
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "combo_out" in n
                        or "mono_out" in n
                    ],
                    "lr": best_hp["learning_rate"] * 0.3,
                },
                {
                    "params": cellLineAE.parameters(),
                    "lr": best_hp["learning_rate"] * 0.1,
                },
                {
                    "params": log_vars,
                    "lr": best_hp["learning_rate"],
                },
                {
                    "params": chemberta_model.parameters(),
                    "lr": best_hp["learning_rate"] * 0.1,
                },
            ]
        )
        mse = MSELoss(reduction="mean").to(device)

        for epoch in range(epochs):
            final_train_sampler.set_epoch(epoch)
            train_loss = train(
                model,
                drug_embeddings,
                chemberta_model,
                cellLineAE.module.encoder,
                final_train_loader,
                optimizer,
                mse,
                log_vars,
                device
            )

            val_loss_list, val_loss = evaluate(
                model,
                drug_embeddings,
                chemberta_model,
                cellLineAE.module.encoder,
                final_val_loader,
                mse,
                log_vars,
                device
            )
            val_loss_mean = ddp_reduce_mean(val_loss)
            if is_main:
                print(
                    f"\tEpoch {epoch+1}:\n\t\tTrain Loss: {train_loss}\n\t\tVal Loss: {val_loss_list}, Weighted Val Loss = {val_loss}",
                    flush=True,
                )
            # Call early stopping only on rank 0
            if is_main:
                model_es(val_loss, model, chemberta_model, cellLineAE, outer_save_path, local_rank)

            # Broadcast early_stop status from rank 0 to all ranks
            stop_tensor = torch.tensor([1 if model_es.early_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

            dist.barrier()
            if should_stop:
                if is_main:
                    print("Early stopping triggered (synced across ranks).")
                break
        # Load the best model
        dist.barrier()
        state_dict = torch.load(outer_save_path, map_location=device)
        model = MTLSynergy3(best_hp["hidden_neurons"], input_dim=MTLSynergy_InputDim + 2).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model.load_state_dict(state_dict)
        # Evaluate on the test set
        test_loss_list, test_loss = evaluate(
            model,
            drug_embeddings,
            chemberta_model,
            cellLineAE.module.encoder,
            final_test_loader,
            mse,
            log_vars,
            device
        )
        test_loss_mean = ddp_reduce_mean(test_loss)

        if is_main:
            print(
                f"\tEpoch {epoch+1}:\n\t\tTest Loss: {test_loss_list}, Weighted Test Loss = {test_loss_mean}",
                flush=True,
            )
            outer_results.append(test_loss_mean)
        torch.cuda.empty_cache()

    # Compute final test performance
    if is_main:
        final_score = np.mean(outer_results)
        print(f"Final Nested CV Test Loss: {final_score}")
    cleanup_ddp()


if __name__ == "__main__":
    main()
    
