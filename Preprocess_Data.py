import pandas as pd
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator, Descriptors
import numpy as np


def reconstructDataset(input_file):
    # Read the Excel file
    nci = pd.read_excel(input_file, skiprows=4)

    # Convert SMILES to RDKit molecules
    molecules = nci['smiles'].apply(MolFromSmiles)
    
    morgan = morganFingerprints(molecules)
    descriptors = molDescriptors(molecules)
    
    pd.concat([descriptors, morgan], axis=1).to_csv('./out/reconstructed.csv', index=False)

    return

def morganFingerprints(molecules):
    # Convert Morgan fingerprints to bit vectors (lists of 0s and 1s)
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    morgan_fingerprints = molecules.apply(lambda mol: np.array(morgan_generator.GetFingerprint(mol), dtype=int))

    # Convert to DataFrame
    fingerprint_df = pd.DataFrame(morgan_fingerprints.tolist())  # Expand arrays into columns
    
    # Normalize the data
    for col in fingerprint_df.columns:
        fingerprint_df[col] = (fingerprint_df[col] - fingerprint_df[col].mean()) / fingerprint_df[col].std()
    
    # Save to CSV
    fingerprint_df.to_csv('./out/morgan.csv', index=False)
    
    return fingerprint_df

def molDescriptors(molecules):
    descriptors = []
    for mol in molecules:
        descriptors.append([desc[1](mol) for desc in Descriptors.descList])
    
    descriptors_df = pd.DataFrame(descriptors)
    
    # Normalize the data
    for col in descriptors_df.columns:
        descriptors_df[col] = (descriptors_df[col] - descriptors_df[col].mean()) / descriptors_df[col].std()
        
    commonColumns = np.array(pd.read_csv('./out/common_columns.csv')['Common Columns'])
    
    descriptors_df = descriptors_df[commonColumns]
    
    descriptors_df.to_csv('./out/descriptors.csv', index=False)
    
    return descriptors_df


def find_common_descriptor_columns(input_file):
    # Read the Excel file
    nci = pd.read_excel(input_file, skiprows=4)

    # Convert SMILES to RDKit molecules
    molecules = nci['smiles'].apply(MolFromSmiles)
    
    descriptors = []
    for mol in molecules:
        descriptors.append([desc[1](mol) for desc in Descriptors.descList])
    
    descriptors_df = pd.DataFrame(descriptors)
    
    # Normalize the data
    for col in descriptors_df.columns:
        descriptors_df[col] = (descriptors_df[col] - descriptors_df[col].mean()) / descriptors_df[col].std()
    
    df = pd.read_csv('./data/drug_features.csv', usecols=range(189)).select_dtypes(include=[np.number])
    commonColumns = set()
    for descCol in descriptors_df.columns:
        for dfCol in df.columns:
            if np.corrcoef(descriptors_df[descCol], df[dfCol])[0, 1] > 0.99:
                commonColumns.add(descCol)
    
    # Write commonColumns to a CSV file
    commonColumnsDf = pd.DataFrame(list(commonColumns), columns=['Common Columns'])
    commonColumnsDf.to_csv('./out/common_columns.csv', index=False)
    
    return

# find_common_descriptor_columns('./12859_2023_5524_MOESM1_ESM.xlsx')
reconstructDataset('./12859_2023_5524_MOESM1_ESM.xlsx')