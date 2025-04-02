#!/usr/bin/env python3

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from utils.tools import CategoricalCrossEntropyLoss, EarlyStopping
from torch.nn import MSELoss
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer

from static.constant import CellAE_SaveBase, CellAE_OutputDim, MTLSynergy_InputDim, MTLSynergy_SaveBase, MTLSynergy2_Result
from Models import MTLSynergy2, ChemBERTaEncoder, CellLineAE
from Dataset import mainDataset
from Preprocess_Data import CCLE_DRUGCOMB_FILTERED, DRUGCOMB_FILTERED_TOKENIZED, DRUGCOMB_EMBEDDINGS

hyper_parameters_candidate = [
    {
        'learning_rate': 0.0001,
        'hidden_neurons': [8192, 4096, 4096, 2048]
    },
    {
        'learning_rate': 0.0001,
        'hidden_neurons': [4096, 2048, 2048, 1024]
    },
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, drug_embeddings, cellLineEncoder, train_loader, optimizer, mse, cce):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        (d_row, d_col, c_exp), (synergy_loewe, ri_row, syn_label, d1_label) = batch
        
        c_exp = c_exp.to(device).float()
        synergy_loewe, ri_row = synergy_loewe.to(device).float(), ri_row.to(device).float()
        syn_label, d1_label = syn_label.to(device).long(), d1_label.to(device).long()
    
        zero_embedding = torch.zeros_like(next(iter(drug_embeddings.values()))).to(device)
        d_row_embeddings = torch.stack([drug_embeddings[drug] for drug in d_row]).to(device)
        d_col_embeddings = torch.stack([drug_embeddings.get(drug, zero_embedding) for drug in d_col]).to(device)


        c_embeddings = cellLineEncoder(c_exp)
        
        optimizer.zero_grad()

        pred_syn, pred_ri, pred_syn_class, pred_sen_class = model(d_row_embeddings, d_col_embeddings, c_embeddings)

        loss1 = mse(pred_syn, synergy_loewe)  
        loss2 = mse(pred_ri, ri_row) 
        loss3 = cce(pred_syn_class, syn_label)
        loss4 = cce(pred_sen_class, d1_label)

        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, drug_embeddings, cellLineEncoder, val_loader, mse, cce):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            (d_row, d_col, c_exp), (synergy_loewe, ri_row, syn_label, d1_label) = batch
            
            c_exp = c_exp.to(device).float()
            synergy_loewe, ri_row = synergy_loewe.to(device).float(), ri_row.to(device).float()
            syn_label, d1_label = syn_label.to(device).long(), d1_label.to(device).long()

            d_row_embeddings = torch.stack([drug_embeddings[drug] for drug in d_row]).to(device)
            d_col_embeddings = torch.stack([
                drug_embeddings[drug] if drug != 'nan' else torch.zeros_like(next(iter(drug_embeddings.values())))
                for drug in d_col
            ]).to(device)
            
            c_embeddings = cellLineEncoder(c_exp)
            
            pred_syn, pred_ri, pred_syn_class, pred_sen_class = model(d_row_embeddings, d_col_embeddings, c_embeddings)
            
            loss1 = mse(pred_syn, synergy_loewe)
            loss2 = mse(pred_ri, ri_row)
            loss3 = cce(pred_syn_class, syn_label)
            loss4 = cce(pred_sen_class, d1_label)
            loss = loss1 + loss2 + loss3 + loss4
            total_loss += loss.item()
    return total_loss / len(val_loader)

patience = 100
epochs = 10

kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
outer_results = []

drugcomb_df = pd.read_csv(DRUGCOMB_FILTERED_TOKENIZED, delimiter=',', 
                          dtype={'drug_row': str, 'drug_col': str, 'cell_line_name': str, 'synergy_loewe': float, 'ri_row': float, 'ri_col': float})

cell_line_df = pd.read_csv(CCLE_DRUGCOMB_FILTERED, index_col=0)
drug_embeddings = torch.load(DRUGCOMB_EMBEDDINGS)
drug_embeddings = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in drug_embeddings.items()}

# Load CellLineAE
cell_path = CellAE_SaveBase + str(CellAE_OutputDim) + '.pth'
cellLineAE = CellLineAE(output_dim=CellAE_OutputDim).to(device)
print('---- start to load cellLineAE ----')
cellLineAE.load_state_dict(torch.load(cell_path))
cellLineAE.eval()
print('---- load cellLineAE finished ----')


for outer_fold, (train_val_idx, test_idx) in enumerate(kf_outer.split(drugcomb_df)):
    print(f"Outer Fold {outer_fold+1}")

    train_val_df = drugcomb_df.iloc[train_val_idx]
    test_df = drugcomb_df.iloc[test_idx]
    
    hp_i = 0
    result_per_hp = []
    outer_save_path = f"{MTLSynergy_SaveBase}fold_{outer_fold}.pth"
    for hyper_parameters in hyper_parameters_candidate:
        hidden_neurons = hyper_parameters['hidden_neurons']
        print("--- Hyper parameters " + str(hp_i) + ":" + str(hyper_parameters) + " ---")
        
        # Set up 4-Fold Inner Cross Validation
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_results = []
        for inner_fold, (train_idx, val_idx) in enumerate(kf_inner.split(train_val_df)):
            inner_save_path = f"{MTLSynergy_SaveBase}inner_fold_{outer_fold}_{inner_fold}.pth"
            print(f"Inner Fold {inner_fold+1}")

            train_subset = train_val_df.iloc[train_idx]
            val_subset = train_val_df.iloc[val_idx]

            train_dataset = mainDataset(train_subset, cell_line_df)
            val_dataset = mainDataset(val_subset, cell_line_df)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

            # Initialize Model
            chemberta_model = ChemBERTaEncoder(device=device)
            
            model = MTLSynergy2(hidden_neurons, chemberta_model).to(device)
            model_es = EarlyStopping(patience=patience)
            optimizer = AdamW([
                {'params': model.chemBERTaModel.parameters(), 'lr': 1e-5},  # Fine-tuning ChemBERTa
                {'params': [p for n, p in model.named_parameters() if "chemBERTaModel" not in n], 'lr': hyper_parameters['learning_rate']}
            ])

            mse = MSELoss(reduction='mean').to(device)
            cce = CategoricalCrossEntropyLoss().to(device)

            val_losses = []
            best_val_loss = float('inf')
            inner_val_loss = []
            for epoch in range(epochs):
                train_loss = train(model, drug_embeddings, cellLineAE.encoder, train_loader, optimizer, mse, cce)
                val_loss = evaluate(model, drug_embeddings, cellLineAE.encoder, val_loader, mse, cce)
                val_losses.append(val_loss)
                print(f"    Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                model_es(val_loss, model, inner_save_path)
                if model_es.early_stop or epoch == epochs - 1:
                    best_val_loss = model_es.best_loss
                    inner_results.append(best_val_loss)
                    with open(MTLSynergy2_Result, 'a') as file:
                        file.write("When in epoch " + str(epoch - patience + 1) + ":\n")
                        file.write("Validation loss:" + str(inner_val_loss[epoch - patience]) + "\n")
                        file.write("Best loss:" + str(model_es.best_loss) + "\n")
                    break
        result_per_inner_fold_mean = np.array(inner_results).mean()
        result_per_hp.append(result_per_inner_fold_mean)
        hp_i += 1
    best_hp_i = np.array(result_per_hp).argmin()
    best_hp = hyper_parameters_candidate[best_hp_i]
    print("--------- Best parameters: " + str(best_hp) + " ---------")
    # Train the model using all training/validation data
    train_subset, val_subset = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    final_train_dataset = mainDataset(train_subset, cell_line_df)
    final_val_dataset = mainDataset(val_subset, cell_line_df)
    final_test_dataset = mainDataset(test_df, cell_line_df)

    final_train_loader = DataLoader(final_train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    final_val_loader = DataLoader(final_val_dataset, batch_size=32, shuffle=False, pin_memory=True)
    final_test_loader = DataLoader(final_test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    # Retrain model using the best hyperparameters
    chemberta_model = ChemBERTaEncoder(device=device)
    model = MTLSynergy2(best_hp['hidden_neurons'], chemberta_model, input_dim=MTLSynergy_InputDim).to(device)
    model_es = EarlyStopping(patience=patience)
    optimizer = AdamW([
                {'params': model.chemBERTaModel.parameters(), 'lr': 1e-5},  # Fine-tuning ChemBERTa
                {'params': [p for n, p in model.named_parameters() if "chemBERTaModel" not in n], 'lr': best_hp['learning_rate']}
            ])
    mse = MSELoss(reduction='mean').to(device)
    cce = CategoricalCrossEntropyLoss().to(device)
    
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = train(model, drug_embeddings, cellLineAE.encoder, final_train_loader, optimizer, mse, cce)
        train_losses.append(train_loss)
        
        val_loss = evaluate(model, drug_embeddings, cellLineAE.encoder, final_val_loader, mse, cce)
        val_losses.append(val_loss)
        
        model_es(val_loss, model, outer_save_path)
        if model_es.early_stop:
            with open(MTLSynergy2_Result, 'a') as file:
                stop_epoch = max(0, epoch - patience + 1)
                file.write(f"When in epoch {stop_epoch}:\n")
                val_loss_idx = min(epoch, patience) - patience
                if val_loss_idx >= 0 and val_loss_idx < len(val_loss):
                    file.write(f"Validation loss: {val_losses[val_loss_idx]}\n")
                file.write(f"Best loss: {model_es.best_loss}\n")
            break
    # Evaluate on the test set
    test_loss = evaluate(model, drug_embeddings, cellLineAE.encoder, final_test_loader, mse, cce)
    outer_results.append(test_loss)

    print(f"Outer Fold {outer_fold+1} Test Loss: {test_loss:.4f}\n")

# Compute final test performance
final_score = np.mean(outer_results)
print(f"Final Nested CV Test Loss: {final_score:.4f}")


