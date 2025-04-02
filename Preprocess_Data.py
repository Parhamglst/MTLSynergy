from pubchempy import get_compounds
import numpy as np
import pandas as pd
from Preprocess_Data_Old import remove_ensembl, CELL_LINES_GENES_FILTERED, name_to_depmap
import re
from transformers import RobertaTokenizer
import torch

CHEMBL_DATASET = './data/chembl_35_chemreps.txt'
CHEMBL_MAPPINGS = './data/chembl_uniprot_mapping.txt'
CCLE_MAPPING = './data/sample_info.csv'
DRUGCOMB = './data/summary_v_1_5.csv'
CCLE_DRUGCOMB_FILTERED = './data/CCLE_DrugComb_filtered.csv'
DRUGCOMB_SMILES = './data/DrugComb_SMILES.csv'
DRUGCOMB_SMILES_FILTERED = './data/DrugComb_SMILES_filtered.csv'
DRUGCOMB_FILTERED_TOKENIZED = './data/DrugComb_filtered_tokenized.csv'
DRUGCOMB_CELLLINE_FILTERED = './data/DrugComb_CellLine_filtered.csv'
DRUGCOMB_EMBEDDINGS = './data/DrugComb_embeddings.pt'
DRUGCOMB_FILTERED = './data/DrugComb_filtered.csv'

def get_smiles(drug_name_or_cas, drug_smiles):
    if drug_name_or_cas in drug_smiles:
        return drug_smiles[drug_name_or_cas]
    compounds = None
    try:
        compounds = get_compounds(drug_name_or_cas, 'name')
    except:
        try:
            compounds = get_compounds(drug_name_or_cas, 'cid')
        except:
            drug_smiles[drug_name_or_cas] = None
            return None
    drug_smiles[drug_name_or_cas] = compounds[0].isomeric_smiles if compounds else None
    return drug_smiles[drug_name_or_cas]

def construct_cell_line_features():
    cell_line_df = pd.read_csv(CELL_LINES_GENES_FILTERED)
    
    drugCombCellLineFeatures = pd.DataFrame(columns=cell_line_df.columns)
    drugCombCellLineFeatures.set_index('Unnamed: 0', inplace=True)
    
    mapping_df = pd.read_csv(CCLE_MAPPING)
    cell_line_df.set_index('Unnamed: 0', inplace=True)
    
    drugCombCellLines = pd.read_csv(DRUGCOMB, usecols=['cell_line_name'])
    drugCombCellLines = np.unique(list(drugCombCellLines['cell_line_name']))
    
    for cell_line in drugCombCellLines:
        clean_name = cell_line
        if '[' in cell_line:
            clean_name = cell_line.split('[')[0]
        clean_name = re.sub(r"[-. /;]", "", clean_name.strip())
        
        try:
            depmap = name_to_depmap(clean_name, mapping_df)
            ccle_row = cell_line_df.loc[[depmap]]
            ccle_row.index = [cell_line]
            drugCombCellLineFeatures = pd.concat([drugCombCellLineFeatures, ccle_row])
        except:
            print(f"Could not find {cell_line}")
    
    new_column_names = [remove_ensembl(col) for col in cell_line_df.columns]
    drugCombCellLineFeatures.columns = new_column_names
    drugCombCellLineFeatures.to_csv(CCLE_DRUGCOMB_FILTERED)
    return

def drugcomb_to_smiles():
    drug_smiles = {}
    i = 0
    write_header = True
    for chunk in pd.read_csv(DRUGCOMB, chunksize=10000):
        chunk.insert(2, 'drug_row_smiles', None)
        chunk.insert(3, 'drug_col_smiles', None)
        to_drop = []
        for idx, row in chunk.iterrows():
            drug_row = get_smiles(row['drug_row'], drug_smiles)
            if drug_row is None and not pd.isnull(chunk.at[idx, 'drug_row']):
                to_drop.append(idx)
                continue
            drug_col = get_smiles(row['drug_col'], drug_smiles)
            if drug_col is None and not pd.isnull(chunk.at[idx, 'drug_col']):
                to_drop.append(idx)
                continue
            chunk.at[idx, 'drug_row_smiles'] = drug_row
            chunk.at[idx, 'drug_col_smiles'] = drug_col
        chunk.drop(to_drop, inplace=True)
        chunk.to_csv(DRUGCOMB_SMILES, mode='a', index=False, header=write_header)
        write_header = False
        i += 1
        print(f"Processed chunk {i}")

def common_cell_lines():
    drugcomb_df = pd.read_csv(DRUGCOMB)
    cell_line_df = pd.read_csv(CCLE_DRUGCOMB_FILTERED, index_col=0)
    
    common_cell_lines = set(drugcomb_df['cell_line_name']).intersection(set(cell_line_df.index))
    
    drugcomb_df_filtered = drugcomb_df[drugcomb_df['cell_line_name'].isin(common_cell_lines)]
    drugcomb_df_filtered.to_csv(DRUGCOMB_CELLLINE_FILTERED, index=False)

def tokenize_smiles():
    drugcomb_df = pd.read_csv(DRUGCOMB_SMILES_FILTERED)
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    drugcomb_df['drug_col'] = drugcomb_df['drug_col'].apply(lambda x: tokenizer.encode(x, return_tensors="pt", padding='max_length', truncation=True) if not pd.isnull(x) else None)
    drugcomb_df['drug_row'] = drugcomb_df['drug_row'].apply(lambda x: tokenizer.encode(x, return_tensors="pt", padding='max_length', truncation=True))
    
    drugcomb_df.to_csv(DRUGCOMB_FILTERED_TOKENIZED, index=False)
    return

def drugcomb_filtered():
    # drugcomb_df = pd.read_csv(DRUGCOMB_CELLLINE_FILTERED, delimiter=',', dtype={'drug_row': str, 'drug_col': str, 'cell_line_name': str, 'synergy_loewe': float, 'ri_row': float, 'ri_col': float})
    # drug_smiles = {}
    # to_drop = []
    # for idx, row in drugcomb_df.iterrows():
    #     drug_row = get_smiles(row['drug_row'], drug_smiles)
    #     if drug_row is None and pd.notna(row['drug_row']):
    #         to_drop.append(idx)
    #         continue
    #     drug_col = get_smiles(row['drug_col'], drug_smiles)
    #     if drug_col is None and pd.notna(row['drug_col']):
    #         to_drop.append(idx)
    #         continue
    # drugcomb_df.drop(to_drop, inplace=True)
    # drugcomb_df.to_csv(DRUGCOMB_FILTERED, index=False)
    embeddings = torch.load(DRUGCOMB_EMBEDDINGS)
    
    remove_dash_n(DRUGCOMB_CELLLINE_FILTERED)
    
    drugcomb_df = pd.read_csv(DRUGCOMB_CELLLINE_FILTERED, delimiter=',', dtype={'drug_row': str, 'drug_col': str, 'cell_line_name': str, 'synergy_loewe': float, 'ri_row': float, 'ri_col': float})
    
    # Filter drug_row by the keys that exist in embeddings
    drugcomb_df = drugcomb_df[drugcomb_df['drug_row'].isin(embeddings.keys())]
    
    drugcomb_df = drugcomb_df.reset_index(drop=True)
    
    mask = drugcomb_df['drug_col'].notna() & ~drugcomb_df['drug_col'].isin(embeddings.keys())
    drugcomb_df = drugcomb_df[~mask]

    
    drugcomb_df.to_csv(DRUGCOMB_FILTERED, index=False)

def remove_dash_n(path):
    drugcomb_df = pd.read_csv(path, delimiter=',', dtype=str)
    drugcomb_df.replace({'\\N': None}, inplace=True)  # Replace \N with None
    drugcomb_df = drugcomb_df.astype({'synergy_loewe': float, 'ri_row': float, 'ri_col': float})
    drugcomb_df[['synergy_loewe']] = drugcomb_df[['synergy_loewe']].fillna(0)
    drugcomb_df.to_csv(path, index=False)

def _drugcomb_tokenized():
    tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    drugcomb_df = pd.read_csv(DRUGCOMB_CELLLINE_FILTERED)
    
    drug_smiles = {}

    # Apply get_smiles function to fetch all SMILES strings at once
    drugcomb_df['drug_row_smiles'] = drugcomb_df['drug_row'].map(lambda d: get_smiles(d, drug_smiles) if pd.notnull(d) else None)
    drugcomb_df['drug_col_smiles'] = drugcomb_df['drug_col'].map(lambda d: get_smiles(d, drug_smiles) if pd.notnull(d) else None)

    # Identify rows with missing SMILES and drop them
    to_drop = drugcomb_df[drugcomb_df['drug_row_smiles'].isna() | (drugcomb_df['drug_col_smiles'].isna() & drugcomb_df['drug_row_smiles'].isna())].index
    drugcomb_df.drop(index=to_drop, inplace=True)
    
    # Get unique drugs for tokenization
    unique_drugs = pd.concat([drugcomb_df['drug_row'], drugcomb_df['drug_col']]).dropna().unique().tolist()
    
    # Tokenize all unique drugs in batch
    tokenized_output = tokenizer(unique_drugs, return_tensors="pt", padding="max_length", truncation=True)

    # Extract tokenized input_ids (since tokenizer output is a dictionary)
    embeddings = {drug: tokenized_output['input_ids'][i] for i, drug in enumerate(unique_drugs)}

    # Save embeddings
    torch.save(embeddings, DRUGCOMB_EMBEDDINGS)


if __name__ == '__main__':
    drugcomb_filtered()