#!/usr/bin/env python3

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
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
from Models import MTLSynergy2, ChemBERTaEncoder, CellLineAE
from Dataset import mainDataset
import argparse

CCLE_DRUGCOMB_FILTERED = "./data/CCLE_DrugComb_filtered.csv"
DRUGCOMB_FILTERED_TOKENIZED = "./data/DrugComb_filtered_tokenized_chembl.csv"
DRUGCOMB_EMBEDDINGS = "./data/DrugComb_embeddings_chembl.pt"

hyper_parameters_candidate = [
    {"learning_rate": 0.0001, "hidden_neurons": [8192, 4096, 2048, 2048]},
    {"learning_rate": 0.0001, "hidden_neurons": [4096, 2048, 1024, 1024]},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_pooling(last_hidden_state, attention_mask):
    # Expand mask to (batch_size, seq_len, 1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    # Sum of embeddings for non-masked tokens
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    # Count of non-masked tokens
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    # Mean pooling
    return sum_embeddings / sum_mask


def train(
    model,
    drug_embeddings,
    chemBERTa_model,
    cellLineEncoder,
    train_loader,
    optimizer,
    mse,
    cce,
    gradNormController: GradNormController,
    task_weights: nn.Parameter,
):
    model.train()
    total_loss_epoch = np.zeros(8)
    for batch in train_loader:
        (d_row, d_col, c_exp), (
            synergy_loewe,
            ri_row,
            syn_label,
            d1_label,
            bliss,
            zip_score,
            hsa,
            ic50,
        ) = batch

        c_exp = c_exp.to(device).float()
        synergy_loewe, ri_row, bliss, zip_score, hsa, ic50 = (
            synergy_loewe.to(device).float(),
            ri_row.to(device).float(),
            bliss.to(device).float(),
            zip_score.to(device).float(),
            hsa.to(device).float(),
            ic50.to(device).float(),
        )
        syn_label, d1_label = syn_label.to(device).long(), d1_label.to(device).long()

        input_ids_row_list, attn_mask_row_list = [], []
        input_ids_col_list, attn_mask_col_list = [], []

        for drug_row, drug_col in zip(d_row, d_col):
            # Process drug_row
            row_embedding = drug_embeddings[drug_row]
            input_ids_row_list.append(row_embedding["input_ids"])
            attn_mask_row_list.append(row_embedding["attention_mask"])

            # Process drug_col, with fallback to drug_row if NaN
            col_embedding = (
                row_embedding if drug_col == "nan" else drug_embeddings[drug_col]
            )
            input_ids_col_list.append(col_embedding["input_ids"])
            attn_mask_col_list.append(col_embedding["attention_mask"])

        input_ids_row = torch.stack(input_ids_row_list).to(device)
        attn_mask_row = torch.stack(attn_mask_row_list).to(device)
        input_ids_col = torch.stack(input_ids_col_list).to(device)
        attn_mask_col = torch.stack(attn_mask_col_list).to(device)

        # Feed into ChemBERTa
        output_row = chemBERTa_model(
            input_ids=input_ids_row, attention_mask=attn_mask_row
        )
        d_row_embeddings = mean_pooling(output_row.last_hidden_state, attn_mask_row)
        output_col = chemBERTa_model(
            input_ids=input_ids_col, attention_mask=attn_mask_col
        )
        d_col_embeddings = mean_pooling(output_col.last_hidden_state, attn_mask_col)

        c_embeddings = cellLineEncoder(c_exp)

        optimizer.zero_grad()

        (
            pred_syn,
            pred_ri,
            pred_syn_class,
            pred_sen_class,
            pred_bliss,
            pred_zip,
            pred_hsa,
            pred_ic50,
        ) = model(d_row_embeddings, d_col_embeddings, c_embeddings)

        ema_mean_yhat = torch.zeros(1, device=device)
        ema_mean_y = torch.zeros(1, device=device)
        ema_var_yhat = torch.ones(1, device=device)
        ema_var_y = torch.ones(1, device=device)
        momentum = 0.99

        def corr_loss(y_hat, y, eps=1e-6):
            nonlocal ema_mean_yhat, ema_mean_y, ema_var_yhat, ema_var_y, momentum
            ema_mean_yhat = momentum * ema_mean_yhat + (1 - momentum) * y_hat.mean()
            ema_mean_y = momentum * ema_mean_y + (1 - momentum) * y.mean()
            ema_var_yhat = momentum * ema_var_yhat + (1 - momentum) * y_hat.var(
                unbiased=False
            )
            ema_var_y = momentum * ema_var_y + (1 - momentum) * y.var(unbiased=False)

            y_hat_c = y_hat - ema_mean_yhat
            y_c = y - ema_mean_y
            num = (y_hat_c * y_c).mean()
            denom = (ema_var_yhat * ema_var_y).sqrt() + eps
            rho = num / denom
            return (1 - rho)[0]

        print(
            mse(pred_syn, synergy_loewe),
            mse(pred_ri, ri_row),
            cce(pred_syn_class, syn_label),
            cce(pred_sen_class, d1_label),
            mse(pred_bliss, bliss),
            mse(pred_zip, zip_score),
            mse(pred_hsa, hsa),
            mse(pred_ic50, ic50),
        )

        losses = [
            mse(pred_syn, synergy_loewe), # + 0.05 * corr_loss(pred_syn, synergy_loewe),
            mse(pred_ri, ri_row), # + 0.05 * corr_loss(pred_ri, ri_row),
            cce(pred_syn_class, syn_label),
            cce(pred_sen_class, d1_label),
            mse(pred_bliss, bliss), # + 0.05 * corr_loss(pred_bliss, bliss),
            mse(pred_zip, zip_score), # + 0.05 * corr_loss(pred_zip, zip_score),
            mse(pred_hsa, hsa), # + 0.05 * corr_loss(pred_hsa, hsa),
            mse(pred_ic50, ic50), # + 0.05 * corr_loss(pred_ic50, ic50),
        ]

        weights_soft = torch.softmax(task_weights, dim=0)
        scaled_losses = [weights_soft[i] * losses[i] for i in range(len(losses))]
        loss = sum(scaled_losses)
        if isinstance(model, torch.nn.DataParallel):
            shared_params = list(model.module.drug_cell_line_layer.parameters())
        else:
            shared_params = list(model.drug_cell_line_layer.parameters())
        gradnorm_loss = gradNormController.compute_gradnorm_loss(shared_params, losses)
        total_loss = loss + 0.1 * gradnorm_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print("Per Task Weights:", weights_soft.detach().cpu().numpy())

        total_loss_epoch += [l.item() for l in losses]

    return total_loss_epoch / len(train_loader)


def evaluate(
    model,
    drug_embeddings,
    chemBERTa_model,
    cellLineEncoder,
    val_loader,
    mse,
    cce,
    task_weights=None,
):
    model.eval()
    total_loss_epoch = np.zeros(8)
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            (d_row, d_col, c_exp), (
                synergy_loewe,
                ri_row,
                syn_label,
                d1_label,
                bliss,
                zip_score,
                hsa,
                ic50,
            ) = batch

            true_var = {
                "Synergy": np.var(synergy_loewe.detach().cpu().numpy()),
                "RI": np.var(ri_row.detach().cpu().numpy()),
                "IC50": np.var(ic50.detach().cpu().numpy()),
            }
            c_exp = c_exp.to(device).float()
            synergy_loewe, ri_row, bliss, zip_score, hsa, ic50 = (
                synergy_loewe.to(device).float(),
                ri_row.to(device).float(),
                bliss.to(device).float(),
                zip_score.to(device).float(),
                hsa.to(device).float(),
                ic50.to(device).float(),
            )
            syn_label, d1_label = (
                syn_label.to(device).long(),
                d1_label.to(device).long(),
            )

            input_ids_row_list, attn_mask_row_list = [], []
            input_ids_col_list, attn_mask_col_list = [], []

            for drug_row, drug_col in zip(d_row, d_col):
                # Process drug_row
                row_embedding = drug_embeddings[drug_row]
                input_ids_row_list.append(row_embedding["input_ids"])
                attn_mask_row_list.append(row_embedding["attention_mask"])

                # Process drug_col, with fallback to drug_row if NaN
                col_embedding = (
                    row_embedding if drug_col == "nan" else drug_embeddings[drug_col]
                )
                input_ids_col_list.append(col_embedding["input_ids"])
                attn_mask_col_list.append(col_embedding["attention_mask"])

            input_ids_row = torch.stack(input_ids_row_list).to(device)
            attn_mask_row = torch.stack(attn_mask_row_list).to(device)
            input_ids_col = torch.stack(input_ids_col_list).to(device)
            attn_mask_col = torch.stack(attn_mask_col_list).to(device)

            # Feed into ChemBERTa
            output_row = chemBERTa_model(
                input_ids=input_ids_row, attention_mask=attn_mask_row
            )
            d_row_embeddings = mean_pooling(output_row.last_hidden_state, attn_mask_row)
            output_col = chemBERTa_model(
                input_ids=input_ids_col, attention_mask=attn_mask_col
            )
            d_col_embeddings = mean_pooling(output_col.last_hidden_state, attn_mask_col)

            c_embeddings = cellLineEncoder(c_exp)

            (
                pred_syn,
                pred_ri,
                pred_syn_class,
                pred_sen_class,
                pred_bliss,
                pred_zip,
                pred_hsa,
                pred_ic50,
            ) = model(d_row_embeddings, d_col_embeddings, c_embeddings)

            losses = [
                mse(pred_syn, synergy_loewe),
                mse(pred_ri, ri_row),
                cce(pred_syn_class, syn_label),
                cce(pred_sen_class, d1_label),
                mse(pred_bliss, bliss),
                mse(pred_zip, zip_score),
                mse(pred_hsa, hsa),
                mse(pred_ic50, ic50),
            ]

            if task_weights is not None:
                weights_soft = torch.softmax(task_weights, dim=0)
                scaled_losses = [
                    weights_soft[i] * losses[i] for i in range(len(losses))
                ]
                loss = sum(scaled_losses)
            else:
                loss = sum(losses)
            total_loss_epoch += [l.item() for l in losses]
            total_loss += loss.item()
            pred_var = {
                "Synergy": np.var(pred_syn.detach().cpu().numpy()),
                "RI": np.var(pred_ri.detach().cpu().numpy()),
                "IC50": np.var(pred_ic50.detach().cpu().numpy()),
            }
            print(
                f"Validation - Synergy Var: {true_var['Synergy']:.4f}, Predicted Var: {pred_var['Synergy']:.4f}, "
                f"RI Var: {true_var['RI']:.4f}, Predicted Var: {pred_var['RI']:.4f}, "
                f"IC50 Var: {true_var['IC50']:.4f}, Predicted Var: {pred_var['IC50']:.4f}"
            )

    return total_loss_epoch / len(val_loader), total_loss / len(val_loader)


patience = 3
epochs = 25
batch_size = 32

parser = argparse.ArgumentParser(
    description="Train MTLSynergy2 model with nested cross-validation."
)
parser.add_argument(
    "--weight_dir", type=str, required=True, help="Directory to save model weights."
)
args = parser.parse_args()
MTLSynergy_SaveBase = args.weight_dir + "/"

kf_outer = KFold(n_splits=3, shuffle=True, random_state=42)
outer_results = []

drugcomb_df = pd.read_csv(
    DRUGCOMB_FILTERED_TOKENIZED,
    delimiter=",",
    dtype={
        "drug_row": str,
        "drug_col": str,
        "cell_line_name": str,
        "synergy_loewe": float,
        "ri_row": float,
        "ri_col": float,
        "ic50_row": float,
        "synergy_zip": float,
        "synergy_bliss": float,
        "synergy_hsa": float,
    },
)

cell_line_df = pd.read_csv(CCLE_DRUGCOMB_FILTERED, index_col=0)
drug_embeddings = torch.load(DRUGCOMB_EMBEDDINGS)
drug_embeddings = {
    key: value.to(device) if isinstance(value, torch.Tensor) else value
    for key, value in drug_embeddings.items()
}

# Load CellLineAE
cell_path = CellAE_SaveBase + str(CellAE_OutputDim) + ".pth"


for outer_fold, (train_val_idx, test_idx) in enumerate(kf_outer.split(drugcomb_df)):
    print(f"Outer Fold {outer_fold+1}")

    train_val_df = drugcomb_df.iloc[train_val_idx]
    test_df = drugcomb_df.iloc[test_idx]

    hp_i = 0
    result_per_hp = []
    outer_save_path = f"{MTLSynergy_SaveBase}fold_{outer_fold}.pth"
    for hyper_parameters in hyper_parameters_candidate:
        hidden_neurons = hyper_parameters["hidden_neurons"]
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
            print(f"Inner Fold {inner_fold+1}")

            train_subset = train_val_df.iloc[train_idx]
            val_subset = train_val_df.iloc[val_idx]

            train_dataset = mainDataset(train_subset, cell_line_df)
            val_dataset = mainDataset(val_subset, cell_line_df)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
            )

            # Initialization of model, optimizer, and loss functions
            chemberta_model = ChemBERTaEncoder().to(device)
            cellLineAE = CellLineAE(output_dim=CellAE_OutputDim).to(device)
            cellLineAE.load_state_dict(torch.load(cell_path))
            for param in cellLineAE.parameters():
                param.requires_grad = False
            for param in chemberta_model.parameters():
                param.requires_grad = False
            model = MTLSynergy2(hidden_neurons, input_dim=MTLSynergy_InputDim)
            task_weights = nn.Parameter(torch.ones(8).to(device))
            gradNormController = GradNormController(task_weights)
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                model = torch.nn.DataParallel(model)
            model.to(device)
            model_es = EarlyStopping(patience=patience)
            optimizer = AdamW(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if "Drug_cell_line_layer" in n
                            or "synergy_layer" in n
                            or "sensitivity_layer" in n
                        ],
                        "lr": hyper_parameters["learning_rate"],
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if "synergy_out_1" in n
                            or "sensitivity_out_1" in n
                            or "ic50" in n
                            or "bliss_out" in n
                            or "zip_out" in n
                            or "hsa_out" in n
                        ],
                        "lr": hyper_parameters["learning_rate"] * 0.3,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if "synergy_out_2" in n or "sensitivity_out_2" in n
                        ],
                        "lr": hyper_parameters["learning_rate"] * 0.1,
                    },
                    {
                        "params": [task_weights],
                        "lr": hyper_parameters["learning_rate"] * 0.1,
                    },
                    #{
                    #    "params": cellLineAE.parameters(),
                    #    "lr": hyper_parameters["learning_rate"] * 0.1,
                    #},
                    #{
                    #    "params": chemberta_model.parameters(),
                    #    "lr": hyper_parameters["learning_rate"] * 0.1,
                    #},
                ]
            )

            mse = MSELoss(reduction="mean").to(device)
            cce = CategoricalCrossEntropyLoss().to(device)

            best_val_loss = float("inf")
            inner_val_loss = []
            for epoch in range(epochs):
                train_loss = train(
                    model,
                    drug_embeddings,
                    chemberta_model,
                    cellLineAE.encoder,
                    train_loader,
                    optimizer,
                    mse,
                    cce,
                    gradNormController,
                    task_weights,
                )
                val_loss_list, val_loss = evaluate(
                    model,
                    drug_embeddings,
                    chemberta_model,
                    cellLineAE.encoder,
                    val_loader,
                    mse,
                    cce,
                    task_weights,
                )
                print(
                    f"\tEpoch {epoch+1}:\n\t\tTrain Loss: {train_loss}\n\t\tVal Loss: {val_loss_list}, Weighted Val Loss = {val_loss}",
                    flush=True,
                )
                model_es(val_loss, model, inner_save_path)
                if model_es.early_stop or epoch == epochs - 1:
                    best_val_loss = model_es.best_loss
                    inner_results.append(best_val_loss)
        result_per_inner_fold_mean = np.array(inner_results).mean()
        result_per_hp.append(result_per_inner_fold_mean)
        hp_i += 1
    best_hp_i = np.array(result_per_hp).argmin()
    best_hp = hyper_parameters_candidate[best_hp_i]
    print("--------- Best parameters: " + str(best_hp) + " ---------")
    train_subset, val_subset = train_test_split(
        train_val_df, test_size=0.2, random_state=42
    )

    final_train_dataset = mainDataset(train_subset, cell_line_df)
    final_val_dataset = mainDataset(val_subset, cell_line_df)
    final_test_dataset = mainDataset(test_df, cell_line_df)

    final_train_loader = DataLoader(
        final_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    final_val_loader = DataLoader(
        final_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    final_test_loader = DataLoader(
        final_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # Retrain model using the best hyperparameters
    chemberta_model = ChemBERTaEncoder()
    chemberta_model.to(device)
    cellLineAE = CellLineAE(output_dim=CellAE_OutputDim).to(device)
    cellLineAE.load_state_dict(torch.load(cell_path))
    for param in cellLineAE.parameters():
        param.requires_grad = True
    for param in chemberta_model.parameters():
        param.requires_grad = True
    model = MTLSynergy2(best_hp["hidden_neurons"], input_dim=MTLSynergy_InputDim)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    model_es = EarlyStopping(patience=patience)

    task_weights = nn.Parameter(torch.ones(8, device=device), requires_grad=True)
    gradNormController = GradNormController(task_weights)

    optimizer = AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "chemBERTaModel" not in n and "ic50" not in n
                ],
                "lr": hyper_parameters["learning_rate"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "chemBERTaModel" not in n and "ic50" in n
                ],
                "lr": hyper_parameters["learning_rate"] / 3,
            },
            {"params": [task_weights], "lr": hyper_parameters["learning_rate"] / 10},
        ]
    )
    mse = MSELoss(reduction="mean").to(device)
    cce = CategoricalCrossEntropyLoss().to(device)

    for epoch in range(epochs):
        train_loss = train(
            model,
            drug_embeddings,
            chemberta_model,
            cellLineAE.encoder,
            final_train_loader,
            optimizer,
            mse,
            cce,
            gradNormController,
            task_weights,
        )

        val_loss_list, val_loss = evaluate(
            model,
            drug_embeddings,
            chemberta_model,
            cellLineAE.encoder,
            final_val_loader,
            mse,
            cce,
            task_weights,
        )
        print(
            f"\tEpoch {epoch+1}:\n\t\tTrain Loss: {train_loss}\n\t\tVal Loss: {val_loss_list}, Weighted Val Loss = {val_loss}",
            flush=True,
        )
        model_es(val_loss, model, outer_save_path)
        if model_es.early_stop:
            print("Early stopping triggered.")
            break
    # Load the best model
    state_dict = torch.load(outer_save_path)
    model = MTLSynergy2(best_hp["hidden_neurons"], input_dim=MTLSynergy_InputDim)
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    # Evaluate on the test set
    test_loss_list, test_loss = evaluate(
        model,
        drug_embeddings,
        chemberta_model,
        cellLineAE.encoder,
        final_test_loader,
        mse,
        cce,
        task_weights,
    )
    outer_results.append(test_loss)

    print(
        f"\tEpoch {epoch+1}:\n\t\tTest Loss: {test_loss_list}, Weighted Test Loss = {test_loss}",
        flush=True,
    )

# Compute final test performance
final_score = np.mean(outer_results)
print(f"Final Nested CV Test Loss: {final_score}")
