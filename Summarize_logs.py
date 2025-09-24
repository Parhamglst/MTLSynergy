import re
import ast
import numpy as np
from transformers import AutoModel
from Preprocess_Data import get_smiles, DRUGCOMB_FILTERED_NORMALIZED, DRUGCOMB_COMBINATION_ONLY
from Preprocess_Data_Old import prepareCellLine
import pandas as pd
from Models import MTLSynergy2, CellLineAE
import torch
from transformers import RobertaTokenizer
from static.constant import Ver3_CellAE_OutputDim, Ver3_MTLSynergy_InputDim
import matplotlib.pyplot as plt
from collections import OrderedDict

LABELS = ['Loewe', 'RI', 'SynC', 'SenC', 'Bliss', 'ZIP', 'HSA', 'IC50']
COLORS = ['blue', 'green', 'red', 'yellow', 'black', 'purple', 'brown', 'orange']
MEAN_STDS= [(-3.3382211854621424, 11.284730681806884), (12.575582105969719, 20.478139959859508), 
            (5.99584825904328,16.135937080882954), (0.6817214551198124, 25.476638054426935), 
            (0.6755261120365301, 41.47994442740779), (-1.056786546827462,8.851911202852678), 
            (18.328315699034636, 137.18380291202186)]

WEIGHTS = [[4096, 2048, 1024, 1024], [2048, 1024, 512, 512]]


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
            mean_std = MEAN_STDS[LABELS.index(label)]
            mean = mean_std[0]
            std = mean_std[1]
            epoch_losses = [loss * std + mean for loss in epoch_losses]
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
            d1_embeddings = drug_tokenizer(s1, return_tensors="pt", padding="max_length", truncation=True).to(device)
            d2_embeddings = drug_tokenizer(s2, return_tensors="pt", padding='max_length', truncation=True).to(device) if s2 else d1_embeddings
            token_type_ids1 = torch.zeros_like(d1_embeddings["input_ids"].squeeze(0))
            token_type_ids2 = torch.zeros_like(d2_embeddings["input_ids"].squeeze(0))
            d1_encoded = chembertaEncoder(input_ids=d1_embeddings['input_ids'], attention_mask=d1_embeddings['attention_mask'], token_type_ids=token_type_ids1).last_hidden_state[:,0,:]
            d2_encoded = chembertaEncoder(input_ids=d2_embeddings['input_ids'], attention_mask=d2_embeddings['attention_mask'], token_type_ids=token_type_ids2).last_hidden_state[:,0,:]

            cell_line_encoded = cellLineAE.encoder(cell_line_features)
            output = model(d1_encoded.to(device), d2_encoded.to(device), cell_line_encoded.unsqueeze(0).to(device))

        return output
    
    
    drugcomb_df = pd.read_csv(
        DRUGCOMB_COMBINATION_ONLY,
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
        pred_and_true['pred_ic50'] = pred_and_true.get('pred_ic50', []) + [pred_ic50.item()]
    
    plt.figure(figsize=(8,8))
    plt.scatter(pred_and_true['loewe'], pred_and_true['pred_loewe'], label='Loewe', color='blue')
    pcc = np.corrcoef(pred_and_true['loewe'], pred_and_true['pred_loewe'])[0, 1]
    plt.title(f'Loewe Prediction (PCC: {pcc:.2f})')
    plt.axline((0,0), slope=1, color='red', linestyle='--', label='PCC Line')
    plt.axis('equal')
    plt.xlabel('True Loewe')
    plt.ylabel('Predicted Loewe')
    plt.legend()
    plt.grid()
    plt.savefig('loewe_pcc.png')

        
def remove_validation_lines(input_file, output_file):
    """
    Reads an input file and writes its content to an output file,
    skipping any lines that start with 'Validation'.

    Args:
        input_file (str): The path to the file to read from.
        output_file (str): The path to the file to write to.
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Iterate over each line in the input file
            for line in infile:
                if 'Epoch' in line or 'Train' in line or 'Val Loss:' in line or 'Weighted Val Loss' in line:
                    # If the line contains 'Epoch', write it to the new file.
                    outfile.write(line)
        
        print(f"Successfully processed the file.")
        print(f"Lines not containing 'Epoch', 'Train', or 'Val Loss:' were removed.")
        print(f"The cleaned content is saved in '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")       


def clean_log_data(log_data):
    """
    Removes specific validation lines from a string of log data using regex.

    Args:
        log_data: A string containing the log data, which can be multi-lined.

    Returns:
        A string with the specified validation lines removed.
    """
    
    # This regex pattern is constructed to match the target lines exactly.
    # - `^` anchors the match to the beginning of a line.
    # - `\d+\.\d+` matches the floating-point numbers.
    # - `$` anchors the match to the end of a line.
    # This combination ensures that only lines containing this exact structure
    # and nothing more are targeted for removal.
    pattern_to_remove = re.compile(
        r"^Validation - Synergy Var: \d+\.\d+, "
        r"Predicted Var: \d+\.\d+, "
        r"RI Var: \d+\.\d+, "
        r"Predicted Var: \d+\.\d+, "
        r"IC50 Var: \d+\.\d+, "
        r"Predicted Var: \d+\.\d+$"
    )
    # Split the input string into a list of individual lines.
    lines = log_data.splitlines()

    # Create a new list, keeping only the lines that do NOT match the pattern.
    # A line is kept if `pattern_to_remove.search(line)` returns None.
    filtered_lines = [line for line in lines if not pattern_to_remove.search(line)]

    # Join the remaining lines back into a single string.
    cleaned_data = "\n".join(filtered_lines)
    
    return cleaned_data



def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            # Remove 'module.' from the beginning of the key
            new_key = key[7:] 
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


# # LOG_FILE = './slurm-62968403.txt'
# # NUM_TASKS = 8
# # with open(LOG_FILE, 'r') as file:
# #     log_content = file.read()
# #     data = parse_individual_val_losses(log_content)
# #     generate_plots(data)

# PCC main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chemberta_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", attn_implementation="eager").to(device)
tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
cellLineAE = CellLineAE(output_dim=Ver3_CellAE_OutputDim).to(device)
mtlSynergy = MTLSynergy2(WEIGHTS[1], input_dim=Ver3_MTLSynergy_InputDim).to(device)


mtl_state_dict = torch.load("./save/parhamg.1866847.0/fold_0.pth", map_location=device)
mtl_state_dict = remove_module_prefix(mtl_state_dict)
cellLineAE_state_dict = torch.load('./save/parhamg.1866847.0/fold_0_cellLineAE.pth', map_location=device)
cellLineAE_state_dict = remove_module_prefix(cellLineAE_state_dict)
chemberta_state_dict = torch.load('./save/parhamg.1866847.0/fold_0_chemberta.pth', map_location=device)
chemberta_state_dict = remove_module_prefix(chemberta_state_dict)

mtlSynergy.load_state_dict(mtl_state_dict)
mtlSynergy.eval()
cellLineAE.load_state_dict(cellLineAE_state_dict)
cellLineAE.eval()
chemberta_model.load_state_dict(chemberta_state_dict)
chemberta_model.eval()

pcc(mtlSynergy, cellLineAE, chemberta_model, tokenizer, n=1000)

# log_dir = './logs/slurm-65944783.out'

# with open(log_dir, 'r') as file:
#     with open('./logs/slurm-65944783_cleaned.out', 'w') as cleaned_file:
#         log_content = file.read()
#         cleaned_log = clean_log_data(log_content)
#         cleaned_file.write(cleaned_log)
