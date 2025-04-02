import pandas as pd
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator, Descriptors
import numpy as np
import re
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from pubchempy import get_compounds

PAPERS_CELL_LINE_ORIGINAL = './data/12859_2023_5524_MOESM2_ESM.xlsx'
PAPERS_DRUG_ORIGINAL = './data/12859_2023_5524_MOESM1_ESM.xlsx'
PAPERS_CELL_LINE_FEATURES = './data/cell_line_features.csv'
CELL_LINES_GENES_FILTERED = './data/CCLE_genes_filtered.csv'
CELL_LINES_FILTERED = './out/ccle_filtered.csv'
DRUGS_DESC_MEAN_STD = './out/drugs_desc_mean_std.csv'
DRUGS_MORGAN_MEAN_STD = './out/drugs_morgan_mean_std.csv'
CELL_LINES_MEAN_STD = './out/cell_lines_mean_std.csv'
CCLE_CELL_LINES_FILTERED_NORMALIZED = './out/ccle_cell_lines_filtered_normalized.csv'
CHEMBL_DATASET = './data/chembl_35_chemreps.txt'
CHEMBL_MAPPINGS = './data/chembl_uniprot_mapping.txt'


def reconstructDrugDataset(input_file):
    # Read the Excel file
    nci = pd.read_excel(input_file, skiprows=4)

    # Convert SMILES to RDKit molecules
    molecules = nci['smiles'].apply(MolFromSmiles)
    
    descriptors = find_common_descriptor_columns(molecules)
    morgan = morganFingerprints(molecules)
    
    pd.concat([descriptors, morgan], axis=1).to_csv('./out/reconstructed.csv', index=False)

    return

def prepareDrug(smiles):
    # Convert SMILES to RDKit molecules
    molecule = MolFromSmiles(smiles)
    
    descriptors = []
    for desc in Descriptors.descList:
        descriptors.append(desc[1](molecule))
    # Normalize the data
    desc_mean_std_df = pd.read_csv(DRUGS_DESC_MEAN_STD)
    for i in range(len(descriptors)):
        descriptors[i] = (descriptors[i] - desc_mean_std_df['mean'][i]) / desc_mean_std_df['std'][i]
    common_descriptors = list(pd.read_csv('./out/common_columns.csv')['Common Columns'])
    
    drug_features = []
    for cmn in common_descriptors:
        drug_features.append(descriptors[cmn])
    assert len(drug_features) == 189
    
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    morgan_fingerprint = morgan_generator.GetFingerprint(molecule)
    morgan_mean_std_df = pd.read_csv(DRUGS_MORGAN_MEAN_STD)
    for i in range(len(morgan_fingerprint)):
        drug_features.append((morgan_fingerprint[i] - morgan_mean_std_df['mean'][i]) / morgan_mean_std_df['std'][i])

    return drug_features

def prepareCellLine(cellLine):
    cellLine = re.sub(r"[-. ]", "", cellLine)
    sample_info = pd.read_csv('./data/sample_info.csv')
    depmap_id = name_to_depmap(cellLine, sample_info)
    cellLineDf = pd.read_csv(CELL_LINES_GENES_FILTERED)
    cellLineDf.set_index('Unnamed: 0', inplace=True)
    cellLineFeatures = cellLineDf.loc[depmap_id]
    meanStds = pd.read_csv(CELL_LINES_MEAN_STD)
    for i in range(len(cellLineFeatures)):
        cellLineFeatures[i] = (cellLineFeatures[i] - meanStds['mean'][i]) / meanStds['std'][i]
    return np.array(cellLineFeatures)

def normalizeCellLineData(cellLineDf, meanStdDf):
    # Normalize the data
    for col in range(len(cellLineDf.columns)):
        mean = meanStdDf.iloc[col]['mean']
        std = meanStdDf.iloc[col]['std']
        cellLineDf.iloc[:, col] = (cellLineDf.iloc[:, col] - mean) / std
    cellLineDf.to_csv(CCLE_CELL_LINES_FILTERED_NORMALIZED, index=False)
    return cellLineDf

def findCommonCellLines(ccle_filtered, sample_info, cell_lines):
    ccle_filtered = pd.read_csv(ccle_filtered)
    sample_info = pd.read_csv(sample_info)
    cell_lines = pd.read_csv(cell_lines)
    
    ccle_filtered.set_index('Unnamed: 0', inplace=True)
    filtered = pd.DataFrame()
    for idx, row in cell_lines.iterrows():
        clean_name = re.sub(r"[-. ]", "", row['name'])
        try:
            depmap_id = name_to_depmap(clean_name, sample_info)
        except:
            for name in row['synonyms'].split(';'):
                try:
                    clean_name = re.sub(r"[-. ]", "", name.strip())
                    depmap_id = name_to_depmap(clean_name, sample_info)
                    break
                except:
                    continue
        try:
            filtered = pd.concat([filtered, ccle_filtered.loc[[depmap_id]]])
        except:
            filtered.loc[len(filtered)] = [0] * filtered.shape[1]
            print(f'Cell line {depmap_id} not found in CCLE')
    filtered.to_csv(CELL_LINES_FILTERED, index=True)
    return

def fit_mean_std(ccle_df, mtl_df):
    def objective(x, a, b):
        return (x - a) / b
    cellLinesMeanStd = pd.DataFrame(columns=['mean', 'std'])
    for col in range(len(mtl_df.columns)):
        X = np.float128(ccle_df.iloc[:, col])
        Y = np.float128(mtl_df.iloc[:, col])
        popt, _ = curve_fit(objective, X, Y)
        mean = popt[0]
        std = popt[1]
        cellLinesMeanStd.loc[col] = [mean, std]
    cellLinesMeanStd.to_csv(CELL_LINES_MEAN_STD, index=False)
    return

def name_to_depmap(name, sample_info_df):
    try:
        return sample_info_df[sample_info_df['stripped_cell_line_name'].str.lower() == name.lower()].iloc[0]['DepMap_ID']
    except:
        raise ValueError('Cell line not found in sample info file')


def morganFingerprints(molecules):
    # Convert Morgan fingerprints to bit vectors
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    morgan_fingerprints = molecules.apply(lambda mol: np.array(morgan_generator.GetFingerprint(mol), dtype=int))

    # Convert to DataFrame
    fingerprint_df = pd.DataFrame(morgan_fingerprints.tolist())  # Expand arrays into columns
    
    mean_std_df = pd.DataFrame(columns=['mean', 'std'])
    # Normalize the data
    i = 1
    for col in fingerprint_df.columns:
        mean = fingerprint_df[col].mean()
        std = fingerprint_df[col].std()
        mean_std_df.loc[i] = [mean, std]
        fingerprint_df[col] = (fingerprint_df[col] - mean) / std
        i+=1
    # Save to CSV
    fingerprint_df.to_csv('./out/morgan.csv', index=False)
    mean_std_df.to_csv(DRUGS_MORGAN_MEAN_STD, index=False)
    
    return fingerprint_df

def molDescriptors(molecules):
    descriptors = []
    for mol in molecules:
        descriptors.append([desc[1](mol) for desc in Descriptors.descList])
    
    descriptors_df = pd.DataFrame(descriptors)
    
    commonColumns = np.array(pd.read_csv('./out/common_columns.csv')['Common Columns'])
    
    descriptors_df = descriptors_df[commonColumns]
    
    mean_std_df = pd.DataFrame(columns=['mean', 'std'])
    # Normalize the data
    i = 1
    for col in descriptors_df.columns:
        mean = descriptors_df[col].mean()
        std = descriptors_df[col].std()
        mean_std_df.loc[i] = [mean, std]
        descriptors_df[col] = (descriptors_df[col] - mean) / std
        i += 1
    mean_std_df.to_csv(DRUGS_DESC_MEAN_STD, index=False)
    descriptors_df.to_csv('./out/descriptors.csv', index=False)
    
    return descriptors_df

def zscore(df: pd.DataFrame):
    return (df - df.mean()) / df.std()
    
def find_common_descriptor_columns(molecules):
    descriptors = []
    for mol in molecules:
        descriptors.append([desc[1](mol) for desc in Descriptors.descList])
    
    descriptors_df = pd.DataFrame(descriptors)
    descriptors_df.columns = [i for i in range(descriptors_df.shape[1])]
    
    mean_std_df = pd.DataFrame(columns=['mean', 'std'])
    i = 1
    for col in descriptors_df.columns:
        mean = descriptors_df[col].mean()
        std = descriptors_df[col].std()
        mean_std_df.loc[i] = [mean, std]
        i+=1
    mean_std_df.to_csv(DRUGS_DESC_MEAN_STD, index=False)
    
    # Normalize the data
    descriptors_df = descriptors_df.apply(zscore)
    for col in descriptors_df.columns:
        descriptors_df[col].fillna(descriptors_df[col].mean(), inplace=True)
    descriptors_df.dropna(axis=1, inplace=True)
    new_cols = []
    for i in range(len(descriptors_df.columns)):
        new_cols.append(descriptors_df.iloc[:, i].name)
    
    descriptors_df.to_csv('./out/descriptors.csv', index=False)
    
    df = pd.read_csv('./data/drug_features.csv', usecols=range(189)).select_dtypes(include=[np.number])
    similarity_matrix = cdist(descriptors_df.T, df.T, metric='euclidean')
    row, col = linear_sum_assignment(similarity_matrix)
    alignment = sorted(zip(row, col), key=lambda x: x[1])
    common_columns = pd.DataFrame([descriptors_df.columns[row] for row, _ in alignment], columns=['Common Columns'])
    common_columns.to_csv('./out/common_columns.csv', index=False)
    
    aligned_matrix = [descriptors_df.iloc[:, row] for row, _ in alignment]
    aligned_df = pd.DataFrame(aligned_matrix).T
    aligned_df.columns = range(aligned_df.shape[1])
    aligned_df.to_csv('./out/aligned_drugs.csv', index=False)
    return aligned_df

def reconstructCellLineDataset(input_file):
    commonColumns = list(pd.read_csv('./out/common_genes.csv')['Common Genes'])
    cellLineData = pd.read_csv(input_file)
    cmn = [cellLineData.columns[0]]
    for col in commonColumns:
        for c in cellLineData.columns:
            if c.startswith(col):
                cmn.append(c)
                break
    cellLineData = cellLineData[cmn]
    cellLineData.to_csv(CELL_LINES_GENES_FILTERED, index=False)
    return

def find_common_cell_line_columns(input_file):
    papersGenes = next(pd.read_csv(PAPERS_CELL_LINE_FEATURES, chunksize=1)).columns
    newGenes = list(next(pd.read_csv(input_file, chunksize=1)).columns[1:])
    
    for i in range(len(newGenes)):
        newGenes[i] = remove_ensembl(newGenes[i]).strip()
    
    commonGenes = []
    for gene in papersGenes:
        if gene in newGenes:
            commonGenes.append(gene)

    commonGenesDf = pd.DataFrame(commonGenes, columns=['Common Genes'])    
    commonGenesDf.to_csv('./out/common_genes.csv', index=False)
    return

def remove_ensembl(name, regex=r"(.*)\("):
    m = re.match(regex, name)
    if m:
        return m.group(1)
    return name

# find_common_descriptor_columns('./data/12859_2023_5524_MOESM1_ESM.xlsx')
# reconstructDrugDataset(PAPERS_DRUG_ORIGINAL)
# find_common_cell_line_columns('./data/CCLE_expression_full.csv')
# papersGenes = next(pd.read_csv(PAPERS_CELL_LINE_SUMMARIZED, chunksize=1)).columns[1:]
# cmn = list(pd.read_csv('./out/common_genes.csv')['Common Genes'])
# for gene in papersGenes:
#     if gene not in cmn:
#         print(gene)
# reconstructCellLineDataset('./data/CCLE_expression_full.csv')
# findCommonCellLines(CELL_LINES_GENES_FILTERED, './data/sample_info.csv', './data/cell_lines.csv')
# ccle_filtered = pd.read_csv(CELL_LINES_FILTERED, skipfooter=2, engine='python').set_index('Unnamed: 0')
# cell_line_features = pd.read_csv(PAPERS_CELL_LINE_FEATURES, skipfooter=2, engine='python')
# fit_mean_std(ccle_filtered, cell_line_features)
# ccleFiltered = pd.read_csv(CELL_LINES_FILTERED, usecols=range(0, 5001)).set_index('Unnamed: 0')
# drugsMeanStd = pd.read_csv(CELL_LINES_MEAN_STD)
# normalizeCellLineData(ccleFiltered, drugsMeanStd)