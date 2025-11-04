import re
import ast
from xml.parsers.expat import model
import numpy as np
from transformers import AutoModel
from Preprocess_Data import get_smiles_almanac, DRUGCOMB_COMBINATION_ONLY
from Preprocess_Data_Old import prepareCellLine
import pandas as pd
from Models import MTLSynergy3, CellLineAE
import torch
from transformers import RobertaTokenizer
from static.constant import Ver3_CellAE_OutputDim, Ver3_MTLSynergy_InputDim, CellAE_OutputDim, MTLSynergy_InputDim
import matplotlib.pyplot as plt
from collections import OrderedDict
from Preprocess_Data import ALMANAC_NAMES, ONEIL_ALMANAC_CCLE_FILTERED_NORMALIZED

LABELS = ['viability']
COLORS = ['blue']
MEAN_STDS= [(67.67636802830089, 44.525290054033725)]

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
    def _sim(d1, d1_con, d2, d2_con, c, model, cellLineAE, chembertaEncoder, drug_tokenizer, mapping):
        s1 = get_smiles_almanac(d1, {}, mapping)
        s2 = get_smiles_almanac(d2, {}, mapping)
        try:
            c_f = prepareCellLine(c)
        except Exception as e:
            return [False]
        print(c)
        cell_line_features = torch.tensor(c_f, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            d1_embeddings = drug_tokenizer(s1[1], return_tensors="pt", padding="max_length", truncation=True).to(device)
            d2_embeddings = drug_tokenizer(s2[1], return_tensors="pt", padding='max_length', truncation=True).to(device) if s2 else d1_embeddings
            token_type_ids1 = torch.zeros_like(d1_embeddings["input_ids"].squeeze(0))
            token_type_ids2 = torch.zeros_like(d2_embeddings["input_ids"].squeeze(0))
            d1_encoded = chembertaEncoder(input_ids=d1_embeddings['input_ids'], attention_mask=d1_embeddings['attention_mask'], token_type_ids=token_type_ids1).last_hidden_state[:,0,:]
            d2_encoded = chembertaEncoder(input_ids=d2_embeddings['input_ids'], attention_mask=d2_embeddings['attention_mask'], token_type_ids=token_type_ids2).last_hidden_state[:,0,:]

            cell_line_encoded = cellLineAE.encoder(cell_line_features)
            mono, combo = model(d1_encoded.to(device), d1_con.to(device), d2_encoded.to(device), d2_con.to(device), cell_line_encoded.unsqueeze(0).to(device))

        return mono, combo


    oneil_almanac_df = pd.read_csv(
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
    mapping = pd.read_csv(ALMANAC_NAMES, delimiter='\t', dtype={"NSC": str, "Name": str})
    mapping['NSC'] = mapping['NSC'].str.strip()
    mapping.set_index('NSC', inplace=True)
    mapping = mapping.groupby('NSC')['Name'].apply(list).to_dict()
    
    oneil_almanac_df = oneil_almanac_df[oneil_almanac_df['drug_col'].notna()]
    sample = oneil_almanac_df.sample(n=n)
    pred_and_true = {}
    for row in sample.itertuples():
        d1 = row.drug_row
        d1_con = torch.tensor([[row.conc_row]], dtype=torch.float32, device=device)
        d2 = row.drug_col
        d2_con = torch.tensor([[row.conc_col]], dtype=torch.float32, device=device)
        c = row.cell_line_name
        mono_pred, combo_pred = _sim(d1, d1_con, d2, d2_con, c, model, cellLineAE, chembertaEncoder, drug_tokenizer, mapping)
        if not combo_pred[0]:
            continue
        
        pred_and_true['d1'] = pred_and_true.get('d1', []) + [d1]
        pred_and_true['d2'] = pred_and_true.get('d2', []) + [d2]
        pred_and_true['c'] = pred_and_true.get('c', []) + [c]
        pred_and_true['pred_viability'] = pred_and_true.get('pred_viability', []) + [combo_pred.item()]
        pred_and_true['viability'] = pred_and_true.get('viability', []) + [row.viability]
    
    plt.figure(figsize=(8,8))
    plt.scatter(pred_and_true['viability'], pred_and_true['pred_viability'], label='viability', color='blue')
    pcc = np.corrcoef(pred_and_true['viability'], pred_and_true['pred_viability'])[0, 1]
    plt.title(f'Viability Prediction (PCC: {pcc:.2f})')
    plt.axline((0,0), slope=1, color='red', linestyle='--', label='PCC Line')
    plt.axis('equal')
    plt.xlabel('True Viability')
    plt.ylabel('Predicted Viability')
    plt.legend()
    plt.grid()
    plt.savefig('viability_pcc.png')

        
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
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chemberta_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", attn_implementation="eager").to(device)
    tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    cellLineAE = CellLineAE(output_dim=CellAE_OutputDim).to(device)
    mtlSynergy = MTLSynergy3(WEIGHTS[0], input_dim=MTLSynergy_InputDim+2).to(device)

    cellLineAE_state_dict = torch.load('./save/chp4/fold_2_cellLineAE.pth', map_location=device)
    cellLineAE_state_dict = remove_module_prefix(cellLineAE_state_dict)
    chemberta_state_dict = torch.load('./save/chp4/fold_2_chemberta.pth', map_location=device)
    chemberta_state_dict = remove_module_prefix(chemberta_state_dict)
    mtl_state_dict = torch.load("./save/chp4/fold_2.pth", map_location=device)
    mtl_state_dict = remove_module_prefix(mtl_state_dict)


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
