import torch
import numpy as np
from Models import MTLSynergy, DrugAE, CellLineAE
from static.constant import DrugAE_SaveBase, CellAE_SaveBase, MTLSynergy_SaveBase, MTLSynergy_Result, DrugAE_OutputDim, CellAE_OutputDim
from Preprocess_Data import prepareDrug, prepareCellLine
import pandas as pd
from scipy.spatial.distance import euclidean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_simulation(drugAE, cellLineAE, model, drug1, drug2, cell_line_features):
    # Convert the input data to tensors
    cell_line_features = torch.tensor(cell_line_features, dtype=torch.float32, device=device)
    drug1 = torch.tensor(drug1, dtype=torch.float32, device=device)
    drug2 = torch.tensor(drug2, dtype=torch.float32, device=device)

    # Run the simulation
    with torch.no_grad():
        drug1_encoded = drugAE.encoder(drug1)
        drug2_encoded = drugAE.encoder(drug2)
        cell_line_encoded = cellLineAE.encoder(cell_line_features)
        output = model(drug1_encoded.unsqueeze(0), drug2_encoded.unsqueeze(0), cell_line_encoded.unsqueeze(0))

    return output

drugAE = DrugAE().to(device)
cellLineAE = CellLineAE().to(device)
mtlSynergy = MTLSynergy([8192, 4096, 4096, 2048], DrugAE_OutputDim + CellAE_OutputDim).to(device)

drugAE.load_state_dict(torch.load(DrugAE_SaveBase + str(DrugAE_OutputDim) + '.pth'))
drugAE.eval()
cellLineAE.load_state_dict(torch.load(CellAE_SaveBase + str(CellAE_OutputDim) + '.pth'))
cellLineAE.eval()
mtlSynergy.load_state_dict(torch.load(MTLSynergy_SaveBase + 'fold_4.pth'))
mtlSynergy.eval()

drug1_smiles = 'OP(O)(O)=O.OP(O)(O)=O.OP(O)(O)=O.OP(O)(O)=O.COc1ccc2nc3cc(Cl)ccc3c(Nc4cc(CN5CCCC5)c(O)c(CN6CCCC6)c4)c2n1'
drug2_smiles = 'C1CC[C@H]([C@@H](C1)N)N.C(=O)(C(=O)[O-])[O-].[Pt+2]'
# drug1_smiles = 'C1=C(C(=O)NC(=O)N1)F'
# drug2_smiles = 'CC1(CCCN1)C2=NC3=C(C=CC=C3N2)C(=O)N'

drug1_features = prepareDrug(drug1_smiles)
drug2_features = prepareDrug(drug2_smiles)

s = pd.read_csv('./data/drug_features.csv')
d1 = s.iloc[0, :]
d2 = s.iloc[1, :]
d1c = euclidean(d1, drug1_features)
d2c = euclidean(d2, drug2_features)

cell_line_features = prepareCellLine('FaDu')

syn_out, drug1_sen_out, syn_out_class, d1_sen_out_class = run_simulation(drugAE, cellLineAE, mtlSynergy, drug1_features, drug2_features, cell_line_features)
syn_out = syn_out.cpu().numpy()
drug1_sen_out = drug1_sen_out.cpu().numpy()
syn_out_class = syn_out_class.cpu().numpy()
d1_sen_out_class = d1_sen_out_class.cpu().numpy()

print(syn_out, drug1_sen_out, syn_out_class, d1_sen_out_class)

