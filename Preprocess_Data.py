from pubchempy import get_properties, get_cids, get_compounds
import numpy as np
import pandas as pd
from Preprocess_Data_Old import remove_ensembl, CELL_LINES_GENES_FILTERED, name_to_depmap
import re
from transformers import RobertaTokenizer
import torch
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cmapPy.pandasGEXpress.parse import parse
from time import sleep
# from chembl_webresource_client.new_client import new_client

CHEMBL_DATASET = './data/chembl_35_chemreps.txt'
CHEMBL_MAPPINGS = './data/chembl_uniprot_mapping.txt'
CCLE_MAPPING = './data/sample_info.csv'
DRUGCOMB = './data/summary_v_1_5.csv'
CCLE_DRUGCOMB_FILTERED = './data/CCLE_DrugComb_filtered.csv'
CELL_LINES_GENES_FILTERED_NORMALIZED = './data/CCLE_genes_filtered_normalized.csv'
DRUGCOMB_SMILES = './data/DrugComb_SMILES.csv'
DRUGCOMB_SMILES_FILTERED = './data/DrugComb_SMILES_filtered.csv'
DRUGCOMB_FILTERED_TOKENIZED = './data/DrugComb_filtered_tokenized.csv'
DRUGCOMB_FILTERED_NORMALIZED = './data/DrugComb_filtered_normalized.csv'
DRUGCOMB_FILTERED_TOKENIZED_CHEMBL = './data/DrugComb_filtered_tokenized_chembl.csv'
DRUGCOMB_CELLLINE_FILTERED = './data/DrugComb_CellLine_filtered.csv'
DRUGCOMB_EMBEDDINGS = './data/DrugComb_embeddings.pt'
DRUGCOMB_EMBEDDINGS_CHEMBL = './data/DrugComb_embeddings_chembl.pt'
DRUGCOMB_FILTERED = './data/DrugComb_filtered.csv'
DRUGCOMB_COMBINATION_ONLY = './data/DrugComb_combination_only.csv'
DRUGCOMB_DUPLICATED_REVERSED = './data/DrugComb_duplicated_reversed.csv'
DOSES = './data/doses_CssSyn2020_1.csv'
CONC_IC50 = './data/conc_ic50.csv'
LINCS_RAW = '/home/pareus/nvme0n1p1/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx'

def get_smiles(drug_name_or_cas, drug_smiles):
    if drug_name_or_cas in drug_smiles:
        return drug_smiles[drug_name_or_cas]

    try:
        props = get_properties(
            ['CanonicalSMILES'],
            drug_name_or_cas,
            'name'
        )
        if props:
            drug_smiles[drug_name_or_cas] = props[0].get('ConnectivitySMILES')
            return drug_smiles[drug_name_or_cas]
    except:
        try:
            props = get_properties(
                ['CanonicalSMILES'],
                drug_name_or_cas,
                'cid'
            )
            if props:
                drug_smiles[drug_name_or_cas] = props[0].get('ConnectivitySMILES')
                return drug_smiles[drug_name_or_cas]
        except:
            pass

    drug_smiles[drug_name_or_cas] = None
    return None


# def get_smiles_chembl(drug_name_or_cas, drug_smiles):
#     if drug_name_or_cas in drug_smiles:
#         return drug_smiles[drug_name_or_cas]
#     molecule = new_client.molecule
#     try:
#         result = molecule.search(drug_name_or_cas)
#         if result:
#             drug_smiles[drug_name_or_cas] = result[0]["molecule_structures"]['canonical_smiles']
#             return drug_smiles[drug_name_or_cas]
#     except Exception as e:
#         print(f"Error fetching SMILES for NAME {drug_name_or_cas}: {e}")


def construct_cell_line_features():
    cell_line_df = pd.read_csv(CELL_LINES_GENES_FILTERED_NORMALIZED)
    
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

def drugcomb_tokenized():
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
    drug_name_to_smiles = pd.concat([
        drugcomb_df[['drug_row', 'drug_row_smiles']].rename(columns={'drug_row': 'drug', 'drug_row_smiles': 'smiles'}),
        drugcomb_df[['drug_col', 'drug_col_smiles']].rename(columns={'drug_col': 'drug', 'drug_col_smiles': 'smiles'})
    ]).dropna().drop_duplicates().set_index('drug')['smiles'].to_dict()

    # Tokenize all unique drugs in batch
    tokenized_output = tokenizer(list(drug_name_to_smiles.values()), return_tensors="pt", padding="max_length", truncation=True)

    # Extract tokenized input_ids (since tokenizer output is a dictionary)
    embeddings = {
        drug: {
            'input_ids': tokenized_output['input_ids'][i],
            'attention_mask': tokenized_output['attention_mask'][i]
        }
        for i, (drug, smiles) in enumerate(drug_name_to_smiles.items())
    }
    assert None not in embeddings.values(), "Some drugs were not tokenized correctly."

    # Save embeddings
    torch.save(embeddings, DRUGCOMB_EMBEDDINGS)

def drugcomb_tokenized_filter():
    embeddings = torch.load(DRUGCOMB_EMBEDDINGS)
    drugcomb_df = pd.read_csv(
        DRUGCOMB_FILTERED,
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
    valid_drugs = set(embeddings.keys())
    drugcomb_df = drugcomb_df[
        (drugcomb_df["drug_row"].isin(valid_drugs)) & 
        ((drugcomb_df["drug_col"].isna()) | (drugcomb_df["drug_col"].isin(valid_drugs)))
    ]
    drugcomb_df.to_csv(DRUGCOMB_FILTERED_TOKENIZED, index=False)
    

def drugcomb_chembl_filter():
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
    all_drugs = set(drugcomb_df['drug_row'].unique()).union(set(drugcomb_df['drug_col'].unique()))
    chembl_drugs = torch.load(DRUGCOMB_EMBEDDINGS_CHEMBL)
    diff = all_drugs.difference(set(chembl_drugs.keys()))
    diff.discard('nan')
    drugcomb_df = drugcomb_df[~drugcomb_df['drug_row'].isin(diff)]
    drugcomb_df = drugcomb_df[~drugcomb_df['drug_col'].isin(diff)]
    drugcomb_df.to_csv(DRUGCOMB_FILTERED_TOKENIZED_CHEMBL, index=False)

    
def drugcomb_conc_ic50_filter():
    df_conc = pd.read_csv(DOSES, delimiter='|')
    df_ic50 = pd.read_csv(DRUGCOMB_FILTERED_TOKENIZED)
    
    def keep_conc(df):
        df['conc_r'] = pd.to_numeric(df['conc_r'], errors='coerce')
        df['conc_c'] = pd.to_numeric(df['conc_c'], errors='coerce')

        df1 = df[(df['conc_r'] > 0) & (df['conc_c'] == 0)]
        df2 = df[(df['conc_c'] > 0) & (df['conc_r'] == 0)]

        
        df1 = df1[['drug_row', 'cell_line_name', 'conc_r', 'inhibition']]
        df2 = df2[['drug_col', 'cell_line_name', 'conc_c', 'inhibition']]
        df1.columns = ['drug_name', 'cell_line', 'concentration', 'inhibition']
        df2.columns = ['drug_name', 'cell_line', 'concentration', 'inhibition']
        out = pd.concat([df1, df2], ignore_index=True)
        # Group by and average inhibition values
        out = out.groupby(['drug_name', 'cell_line', 'concentration'], as_index=False).agg({'inhibition': 'mean'})

        return out
    
    def keep_ic50(df):
        df1 = df[df['ic50_row'] > 0]
        df2 = df[df['ic50_col'] > 0]
        
        df1 = df1[['drug_row', 'cell_line_name', 'ic50_row']]
        df2 = df2[['drug_col', 'cell_line_name', 'ic50_col']]
        
        df1.columns = ['drug_name', 'cell_line', 'ic50']
        df2.columns = ['drug_name', 'cell_line', 'ic50']
        
        out = pd.concat([df1, df2], ignore_index=True)
        out.drop_duplicates(subset=['drug_name', 'cell_line'], inplace=True)
        
        return out
    df_conc = keep_conc(df_conc)
    df_ic50 = keep_ic50(df_ic50)

    final_df = pd.merge(df_conc, df_ic50, on=['drug_name', 'cell_line'])

    final_df = final_df[['drug_name', 'cell_line', 'concentration', 'inhibition', 'ic50']]
    assert final_df.shape[0] == final_df[['drug_name', 'cell_line', 'concentration']].drop_duplicates().shape[0]
    
    final_df.to_csv(CONC_IC50, index=False)
    return

def fit_dose_response():
    def four_param_logistic(x, bottom, top, logIC50, hill_slope):
        return bottom + (top - bottom) / (1 + 10**((logIC50 - np.log10(x)) * hill_slope))
    
    df = pd.read_csv(CONC_IC50)
    df['concentration'] = pd.to_numeric(df['concentration'], errors='coerce')
    df['inhibition'] = pd.to_numeric(df['inhibition'], errors='coerce')
    df['ic50'] = pd.to_numeric(df['ic50'], errors='coerce')
    
    grouped = df.groupby(['drug_name', 'cell_line'])

    results = []
    n = 0
    m = 0
    for (drug, cell), group in grouped:
        group = group.sort_values(by='concentration')
        conc = group['concentration'].values
        inhib = group['inhibition'].values
        ic50 = group['ic50'].iloc[0]
        if len(conc) < 4 or len(inhib) < 4:
            print(f"Not enough data for {drug} | {cell}")
            continue
        try:
            concentrations = conc[conc > 0]
            initial_guess = [0, 100, np.log10(1), 1]
            popt, _ = curve_fit(four_param_logistic, concentrations, inhib, p0=initial_guess)
            bottom, top, logIC50, hill_slope = popt
            results.append({
                'drug_name': drug,
                'cell_line': cell,
                'min': bottom,
                'max': top,
                'ic50': 10**logIC50,
                'hill_slope': hill_slope
            })
        except RuntimeError:
            print(f"Fit failed for {drug}-{cell}")
            m += 1

        n += 1
    
    print(f"Total fits: {n}, Failed fits: {m}")
    results = pd.DataFrame(results)
    df_merged = df.merge(results, on=['drug_name', 'cell_line'], how='left')
    df_merged.to_csv('./data/dose-response.csv', index=False)
    
    # Random drug-cell line to plot
    random_drug_cell = results[np.random.randint(0, len(results))]
    drug_name = random_drug_cell['drug_name']
    cell_line = random_drug_cell['cell_line']

    # Filter and clean
    subset = df[(df['drug_name'] == drug_name) & (df['cell_line'] == cell_line)]
    conc = subset['concentration'].values
    inhib = subset['inhibition'].values
    mask = conc > 0
    conc = conc[mask]
    inhib = inhib[mask]

    # Extract fit parameters
    A = random_drug_cell['min']         # bottom
    B = random_drug_cell['max']         # top
    logIC50 = np.log10(random_drug_cell['ic50'])
    hill_slope = random_drug_cell['hill_slope']

    # Generate fit curve
    x = np.logspace(np.log10(min(conc)), np.log10(max(conc)), 100)
    y = four_param_logistic(x, A, B, logIC50, hill_slope)

    # Plot
    plt.semilogx(conc, inhib, 'o', label='Data')
    plt.semilogx(x, y, label='Fitted Curve')
    plt.axvline(x=ic50, color='r', linestyle='--', label='IC50')
    plt.xlabel('Concentration')
    plt.ylabel('Inhibition')
    plt.title(f'Drug: {drug_name}, Cell Line: {cell_line}')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

def prepareCellLine():
    cell_line_df = pd.read_csv(CELL_LINES_GENES_FILTERED, index_col=0)
    cell_line_df = cell_line_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    cell_line_df.to_csv(CELL_LINES_GENES_FILTERED_NORMALIZED)
    return

def normalize_drugcomb():
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
    with open("./data/paramss.txt", "a") as f:
        for col in ["synergy_loewe", "ri_row", "ri_col", "synergy_zip", "synergy_bliss", "synergy_hsa"]:
            mean, std = drugcomb_df[col].mean(), drugcomb_df[col].std()
            f.write(f"{col} mean: {mean}\n")
            f.write(f"{col} std: {std}\n")
            drugcomb_df[col] = (drugcomb_df[col] - mean) / std
        drugcomb_df["ic50_row_log"] = np.log1p(drugcomb_df["ic50_row"])
        mean, std = drugcomb_df["ic50_row_log"].mean(), drugcomb_df["ic50_row_log"].std()
        f.write(f"ic50_row_log mean: {mean}\n")
        f.write(f"ic50_row_log std: {std}\n")
        drugcomb_df["ic50_row"] = (drugcomb_df["ic50_row_log"] - mean) / std
        drugcomb_df.drop(columns=["ic50_row_log"], inplace=True)
    drugcomb_df.to_csv(DRUGCOMB_FILTERED_NORMALIZED, index=False)

def parse_LINCS():
    gctx = parse(LINCS_RAW)
    expr = gctx.data_df
    print(f"Shape of expression data: {expr.shape}")
    
    meta = gctx.row_metadata_df
    print(f"Metadata: {meta}")

def common_LINCS_drugs():
    drugcomb_df = pd.read_csv(DRUGCOMB_FILTERED_TOKENIZED)
    pert_info = pd.read_csv('./data/GSE92742_Broad_LINCS_pert_info.txt', sep='\t')
    cell_info = pd.read_csv('./data/GSE92742_Broad_LINCS_cell_info.txt', sep='\t', dtype=str)
    
    druuuuuuuuuuugs = drugcomb_df['drug_row'].unique()
    for i in range(len(druuuuuuuuuuugs)):
        flag = True
        while flag:
            try:
                druuuuuuuuuuugs[i] = get_cids(druuuuuuuuuuugs[i])
                sleep(0.2)
                flag = False
            except Exception as e:
                print(f"Error fetching CID for {druuuuuuuuuuugs[i]}: {e}", flush=True)
                sleep(1)
    pert_druuuuuuuuuuuuuugs = pert_info[pert_info['pubchem_id'] == druuuuuuuuuuugs]
    print("je")

def common_genes():
    ccle_df = pd.read_csv(CCLE_DRUGCOMB_FILTERED)
    lincs_df = pd.read_csv('./data/cellinfo_beta.txt', sep='\t')
    common = set(ccle_df.iloc[:,0]).intersection(set(lincs_df.iloc[:,0]))
    print(f"Common genes: {len(common)}")

def print_stats():
    df = pd.read_csv(
        DRUGCOMB_FILTERED_NORMALIZED,
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
    
    cols_to_analyze = ["synergy_loewe", "ri_row", "ic50_row", "synergy_zip", "synergy_bliss", "synergy_hsa"]
    
    for col in cols_to_analyze:
        if col in df.columns:
            print(f"Statistics for column: {col}")
            print(f"  Mean: {df[col].mean()}")
            print(f"  Max:  {df[col].max()}")
            print(f"  Min:  {df[col].min()}")
            print(f"  Std:  {df[col].std()}")
            print(f"  0.5 percentile: {df[col].quantile(0.005)}")
            print(f"  99.5 percentile:    {df[col].quantile(0.995)}")
            print("-" * 30)
        else:
            print(f"Column {col} not found in the dataframe.")
            
            
def print_top_synergy_values():
    df = pd.read_csv(
        DRUGCOMB_FILTERED_NORMALIZED,
        delimiter=",",
        usecols=["synergy_loewe"],
        dtype={"synergy_loewe": float},
    )
    
    print("Top 5 most repeated values in 'synergy_loewe':")
    top_5 = df['synergy_loewe'].value_counts().head(5)
    print(top_5)

def combination_only():
    df = pd.read_csv(
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
    df = df[df['drug_col'].notna()]
    with open("./data/params_comb_only.txt", "a") as f:
        for col in ["synergy_loewe", "ri_row", "ri_col", "synergy_zip", "synergy_bliss", "synergy_hsa"]:
            mean, std = df[col].mean(), df[col].std()
            f.write(f"{col} mean: {mean}\n")
            f.write(f"{col} std: {std}\n")
            df[col] = (df[col] - mean) / std
        df["ic50_row_log"] = np.log1p(df["ic50_row"])
        mean, std = df["ic50_row_log"].mean(), df["ic50_row_log"].std()
        f.write(f"ic50_row_log mean: {mean}\n")
        f.write(f"ic50_row_log std: {std}\n")
        df["ic50_row"] = (df["ic50_row_log"] - mean) / std
        df.drop(columns=["ic50_row_log"], inplace=True)
    df.to_csv(DRUGCOMB_COMBINATION_ONLY, index=False)

def duplicated_reversed_rows():
    df = pd.read_csv(
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
    df = df[df['drug_col'].notna()]
    df_copy = df.copy()
    df_copy[['drug_row', 'drug_col']] = df_copy[['drug_col', 'drug_row']].values
    df_copy[['ri_row', 'ri_col']] = df_copy[['ri_col', 'ri_row']].values
    df_copy[['ic50_row', 'ic50_col']] = df_copy[['ic50_col', 'ic50_row']].values
    df = pd.concat([df, df_copy], axis=0)
    with open("./data/params_comb_only_duplicated.txt", "a") as f:
        for col in ["synergy_loewe", "ri_row", "ri_col", "synergy_zip", "synergy_bliss", "synergy_hsa"]:
            mean, std = df[col].mean(), df[col].std()
            f.write(f"{col} mean: {mean}\n")
            f.write(f"{col} std: {std}\n")
            df[col] = (df[col] - mean) / std
        df["ic50_row_log"] = np.log1p(df["ic50_row"])
        mean, std = df["ic50_row_log"].mean(), df["ic50_row_log"].std()
        f.write(f"ic50_row_log mean: {mean}\n")
        f.write(f"ic50_row_log std: {std}\n")
        df["ic50_row"] = (df["ic50_row_log"] - mean) / std
        df.drop(columns=["ic50_row_log"], inplace=True)
    df.to_csv(DRUGCOMB_DUPLICATED_REVERSED, index=False)
    
def viability_only():
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
    print(drugcomb_df['study_name'].unique())
    # with open("./data/params_viability.txt", "a") as f:
    #     for col in ["synergy_loewe", "ri_row", "ri_col", "synergy_zip", "synergy_bliss", "synergy_hsa"]:
    #         mean, std = drugcomb_df[col].mean(), drugcomb_df[col].std()
    #         f.write(f"{col} mean: {mean}\n")
    #         f.write(f"{col} std: {std}\n")
    #         drugcomb_df[col] = (drugcomb_df[col] - mean) / std
    #     drugcomb_df["ic50_row_log"] = np.log1p(drugcomb_df["ic50_row"])
    #     mean, std = drugcomb_df["ic50_row_log"].mean(), drugcomb_df["ic50_row_log"].std()
    #     f.write(f"ic50_row_log mean: {mean}\n")
    #     f.write(f"ic50_row_log std: {std}\n")
    #     drugcomb_df["ic50_row"] = (drugcomb_df["ic50_row_log"] - mean) / std
    #     drugcomb_df.drop(columns=["ic50_row_log"], inplace=True)
    # drugcomb_df.to_csv(DRUGCOMB_FILTERED_NORMALIZED, index=False)

if __name__ == '__main__':
    viability_only()