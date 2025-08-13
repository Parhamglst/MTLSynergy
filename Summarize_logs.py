import re
import ast
import numpy as np
from Preprocess_Data import get_smiles, DRUGCOMB_FILTERED_TOKENIZED
from Preprocess_Data_Old import prepareCellLine
import pandas as pd
from Models import MTLSynergy2, CellLineAE, ChemBERTaEncoder
import torch
from transformers import RobertaTokenizer
from static.constant import Ver3_CellAE_OutputDim, Ver3_MTLSynergy_InputDim
import matplotlib.pyplot as plt

LABELS = ['Loewe', 'RI', 'SynC', 'SenC', 'Bliss', 'ZIP', 'HSA', 'IC50']
COLORS = ['blue', 'green', 'red', 'yellow', 'black', 'purple', 'brown', 'orange']
IC50_MEAN_STD = [18.328315699034636, 137.18380291202186]


def parse_individual_val_losses(file_content):
    data = {}
    cur_fold = 0
    for line in file_content.splitlines():
        if "Outer Fold" in line:
            # Extract fold number
            match = re.search(r"Outer Fold (\d+)", line)
            if match:
                fold_number = match.group(1)
                if fold_number not in data:
                    data[fold_number] = {label: [] for label in LABELS}
                    cur_fold = fold_number
                    
        elif "Val Loss:" in line or "Test Loss:" in line:
            # Extract validation losses
            match = re.search(r"Val Loss: (.+)]", line)
            if not match:
                match = re.search(r"Test Loss: (.+)]", line)
            if match:
                losses = match.group(1).strip()[1:].split(' ')
                losses = [float(loss.strip()) for loss in losses if loss.strip()]
                for label, loss in zip(LABELS, losses):
                    data[cur_fold][label].append(loss)
    return data
    

def generate_plots(data):
    """
    Generates plots for the parsed validation losses data.

    Args:
        data (dict): The dictionary containing hyperparameter configurations and their validation losses.
    """

    for fold_name, losses in data.items():
        plt.figure(figsize=(18, 12))
        for label, epoch_losses in losses.items():
            if label == 'IC50':
                mean_ic50 = IC50_MEAN_STD[0]
                std_ic50 = IC50_MEAN_STD[1]
                epoch_losses = [loss * std_ic50 + mean_ic50 for loss in epoch_losses]
            plt.plot(epoch_losses[:-1], marker='o', label=label, color=COLORS[LABELS.index(label)])
            plt.axhline(y=epoch_losses[-1], color=COLORS[LABELS.index(label)], linestyle=':', label=f'Test Loss: {epoch_losses[-1]:.2f}')
        plt.title(f"Validation Losses for fold {fold_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"fold_{fold_name}.png")
        # plt.show()

def pcc(model, cellLineAE, chembertaEncoder, drug_tokenizer, n=5000):
    def _sim(d1, d2, c, model, cellLineAE, chembertaEncoder, drug_tokenizer):
        s1 = get_smiles(d1, {})
        s2 = get_smiles(d2, {})
        try:
            c_f = prepareCellLine(c)
        except Exception as e:
            return [False] * 8
        print(c)
        cell_line_features = torch.tensor(c_f, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            d1_embeddings = drug_tokenizer.encode(s1, return_tensors="pt", padding="max_length", truncation=True).to(device)
            d2_embeddings = drug_tokenizer.encode(s2, return_tensors="pt", padding='max_length', truncation=True).to(device) if s2 else torch.zeros_like(d1_embeddings).to(device)
            d1_encoded = chembertaEncoder(d1_embeddings)
            d2_encoded = chembertaEncoder(d2_embeddings)
            
            cell_line_encoded = cellLineAE.encoder(cell_line_features)
            output = model(d1_encoded.to(device), d2_encoded.to(device), cell_line_encoded.unsqueeze(0).to(device))

        return output
    
    
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
    drugcomb_df = drugcomb_df[drugcomb_df['drug_col'].notna()]
    sample = drugcomb_df.sample(n=n)
    pred_and_true = {}
    for row in sample.itertuples():
        d1 = row.drug_row
        d2 = row.drug_col
        c = row.cell_line_name
        pred_loewe, pred_sen, _, _, pred_bliss, pred_zip, pred_hsa, pred_ic50 = _sim(d1, d2, c, model, cellLineAE, chembertaEncoder, drug_tokenizer)
        if not pred_loewe:
            continue
        
        pred_and_true['d1'] = pred_and_true.get('d1', []) + [d1]
        pred_and_true['d2'] = pred_and_true.get('d2', []) + [d2]
        pred_and_true['c'] = pred_and_true.get('c', []) + [c]
        pred_and_true['loewe'] = pred_and_true.get('loewe', []) + [row.synergy_loewe]
        pred_and_true['ri_row'] = pred_and_true.get('ri_row', []) + [row.ri_row]
        pred_and_true['ri_col'] = pred_and_true.get('ri_col', []) + [row.ri_col]
        pred_and_true['ic50_row'] = pred_and_true.get('ic50_row', []) + [row.ic50_row]
        pred_and_true['zip'] = pred_and_true.get('zip', []) + [row.synergy_zip]
        pred_and_true['bliss'] = pred_and_true.get('bliss', []) + [row.synergy_bliss]
        pred_and_true['hsa'] = pred_and_true.get('hsa', []) + [row.synergy_hsa]
        pred_and_true['pred_loewe'] = pred_and_true.get('pred_loewe', []) + [pred_loewe.item()]
        pred_and_true['pred_sen'] = pred_and_true.get('pred_sen', []) + [pred_sen.item()]
        pred_and_true['pred_bliss'] = pred_and_true.get('pred_bliss', []) + [pred_bliss.item()]
        pred_and_true['pred_zip'] = pred_and_true.get('pred_zip', []) + [pred_zip.item()]
        pred_and_true['pred_hsa'] = pred_and_true.get('pred_hsa', []) + [pred_hsa.item()]
        pred_and_true['pred_ic50'] = pred_and_true.get('pred_ic50', []) + [pred_ic50.item() * IC50_MEAN_STD[1] + IC50_MEAN_STD[0]]
    
    plt.figure(figsize=(8,8))
    plt.scatter(pred_and_true['loewe'], pred_and_true['pred_loewe'], label='IC50', color='blue')
    pcc = np.corrcoef(pred_and_true['loewe'], pred_and_true['pred_loewe'])[0, 1]
    plt.title(f'IC50 Prediction (PCC: {pcc:.2f})')
    plt.axline((0,0), slope=1, color='red', linestyle='--', label='PCC Line')
    plt.axis('equal')
    plt.xlabel('True IC50')
    plt.ylabel('Predicted IC50')
    plt.legend()
    plt.grid()
    plt.savefig('ic50_pcc.png')

        
        
    
    
# LOG_FILE = './slurm-62968403.txt'
# NUM_TASKS = 8
# with open(LOG_FILE, 'r') as file:
#     log_content = file.read()
#     data = parse_individual_val_losses(log_content)
#     generate_plots(data)

# PCC main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chemBERTaEncoder = ChemBERTaEncoder().to(device)
tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
cellLineAE = CellLineAE(output_dim=Ver3_CellAE_OutputDim).to(device)
mtlSynergy = MTLSynergy2([8192, 4096, 4096, 2048], input_dim=Ver3_MTLSynergy_InputDim).to(device)

cellLineAE.load_state_dict(torch.load("save/AutoEncoder/CellLineAE_" + str(Ver3_CellAE_OutputDim) + ".pth"))
cellLineAE.eval()
mtl_state_dict = torch.load("save/MTLSynergy/chp3/fold_4.pth")
mtlSynergy.load_state_dict(mtl_state_dict)
mtlSynergy.eval()

pcc(mtlSynergy, cellLineAE, chemBERTaEncoder, tokenizer, n=1000)