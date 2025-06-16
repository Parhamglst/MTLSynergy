import torch
import numpy as np
from Models_old import MTLSynergy, MTLSynergy2, ChemBERTaEncoder, CellLineAE, DrugAE
from transformers import RobertaTokenizer
from static.constant import MTLSynergy_InputDim
from Preprocess_Data_Old import prepareCellLine, prepareDrug
from Preprocess_Data import get_smiles
import argparse
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_simulation(drug1, cell_line, cell_line_AE_path, drug_AE_path, model_path = './save/MTLSynergy/fold_4.pth', drug2=None):
    def _run(drugAE, cellLineAE, model, drug1, drug2, cell_line_features):
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
    DrugAE_OutputDim = int(drug_AE_path[-7:-4])
    CellAE_OutputDim = int(cell_line_AE_path[-7:-4])
    drugAE = DrugAE(output_dim=DrugAE_OutputDim).to(device)
    cellLineAE = CellLineAE(output_dim=CellAE_OutputDim).to(device)
    mtlSynergy = MTLSynergy([8192, 4096, 4096, 2048], DrugAE_OutputDim + CellAE_OutputDim).to(device)

    drugAE.load_state_dict(torch.load(drug_AE_path))
    drugAE.eval()
    cellLineAE.load_state_dict(torch.load(cell_line_AE_path))
    cellLineAE.eval()
    mtlSynergy.load_state_dict(torch.load(model_path))
    mtlSynergy.eval()

    drug1_smiles = get_smiles(drug1, {})
    drug2_smiles = get_smiles(drug2, {})
    
    if drug1_smiles is None:
        raise ValueError(f"Drug1 SMILES not found for {drug1}")
    if drug2_smiles is None and drug2 is not None:
        raise ValueError(f"Drug2 SMILES not found for {drug2}")

    drug1_features = prepareDrug(drug1_smiles)
    drug2_features = prepareDrug(drug2_smiles)

    cell_line_features = prepareCellLine(cell_line)
    
    syn_out, drug1_sen_out, syn_out_class, d1_sen_out_class = _run(drugAE, cellLineAE, mtlSynergy, drug1_features, drug2_features, cell_line_features)
    drug2_sen_out = None
    if drug2 is not None:
        syn2_out, drug2_sen_out, syn2_out_class, d2_sen_out_class = _run(drugAE, cellLineAE, mtlSynergy, drug2_features, drug1_features, cell_line_features)
        syn_out = (syn_out + syn2_out) / 2
        
    syn_out = syn_out.cpu().numpy()
    drug1_sen_out = drug1_sen_out.cpu().numpy()
    if drug2_sen_out is not None:
        drug2_sen_out = drug2_sen_out.cpu().numpy()
        display_output(syn_out, drug1_sen_out, drug2_sen_out)
        return
    syn_out_class = syn_out_class.cpu().numpy()
    d1_sen_out_class = d1_sen_out_class.cpu().numpy()
    display_output(syn_out, drug1_sen_out)

    

def run_simulation2(drug1, cell_line, cell_line_AE_path, model_path='./save/MTLSynergy/chp2/fold_0.pth', drug2=None):
    def _sim(tokenizer, chemBERTaEncoder, cellLineAE, model, drug1_smiles, drug2_smiles, cell_line_features):
        # Convert the input data to tensors
        cell_line_features = torch.tensor(cell_line_features, dtype=torch.float32, device=device)

        # Run the simulation
        with torch.no_grad():
            d1_embeddings = tokenizer.encode(drug1_smiles, return_tensors="pt", padding="max_length", truncation=True).to(device)
            d2_embeddings = tokenizer.encode(drug2_smiles, return_tensors="pt", padding='max_length', truncation=True).to(device) if drug2_smiles else torch.zeros_like(d1_embeddings).to(device)
            d1_encoded = chemBERTaEncoder(d1_embeddings)
            d2_encoded = chemBERTaEncoder(d2_embeddings)
            
            cell_line_encoded = cellLineAE.encoder(cell_line_features)
            output = model(d1_encoded.to(device), d2_encoded.to(device), cell_line_encoded.unsqueeze(0).to(device))

        return output
    
    chemBERTaEncoder = ChemBERTaEncoder().to(device)
    tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    cellLineAE = CellLineAE().to(device)
    mtlSynergy = MTLSynergy2([8192, 4096, 4096, 2048], input_dim=MTLSynergy_InputDim).to(device)
    mtlSynergy = torch.nn.DataParallel(mtlSynergy)

    cellLineAE.load_state_dict(torch.load(cell_line_AE_path))
    cellLineAE.eval()
    mtlSynergy.load_state_dict(torch.load(model_path))
    mtlSynergy.eval()

    drug1_smiles = get_smiles(drug1, {})
    drug2_smiles = get_smiles(drug2, {})

    if drug1_smiles is None:
        raise ValueError(f"Drug1 SMILES not found for {drug1}")
    if drug2_smiles is None and drug2 is not None:
        raise ValueError(f"Drug2 SMILES not found for {drug2}")

    cell_line_features = prepareCellLine(cell_line)

    syn_out, drug1_sen_out, syn_out_class, d1_sen_out_class = _sim(tokenizer, chemBERTaEncoder, cellLineAE, mtlSynergy, drug1_smiles, drug2_smiles, cell_line_features)
    drug2_sen_out = None
    if drug2 is not None:
        syn2_out, drug2_sen_out, syn2_out_class, d2_sen_out_class = _sim(tokenizer, chemBERTaEncoder, cellLineAE, mtlSynergy, drug2_smiles, drug1_smiles, cell_line_features)
        syn_out = (syn_out + syn2_out) / 2
    syn_out = syn_out.cpu().numpy()
    drug1_sen_out = drug1_sen_out.cpu().numpy()
    if drug2 is not None:
        drug2_sen_out = drug2_sen_out.cpu().numpy()
        display_output(syn_out, drug1_sen_out, drug2_sen_out)
        return
    syn_out_class = syn_out_class.cpu().numpy()
    d1_sen_out_class = d1_sen_out_class.cpu().numpy()
    display_output(syn_out, drug1_sen_out)

def display_output(syn_out, drug1_sen_out, drug2_sen_out=None):
    print("Synergy Loewe:", syn_out)
    print("Drug1 Sensitivity:", drug1_sen_out)
    if drug2_sen_out is not None:
        print("Drug2 Sensitivity:", drug2_sen_out)

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_version = config.get('model_version', 2)
    model_path = config.get('model_path', 'save/MTLSynergy/chp2/fold_0.pth')
    drugAE_path = config.get('drugAE_path', 'save/AutoEncoder/DrugAE_128.pth')
    cellAE_path = config.get('cellAE_path', 'save/AutoEncoder/CellLineAE_256.pth')
    
    # Argument parser
    argparser = argparse.ArgumentParser(description="Run simulation for drug synergy prediction.")
    argparser.add_argument("--drug1", '-d1', type=str, required=True, help="Name of the first drug. If it includes spaces, use underscore.")
    argparser.add_argument("--cell_line", '-c', type=str, required=True, help="Name of the cell line.")
    argparser.add_argument("--drug2", '-d2', type=str, help="Name of the second drug (optional). If it includes spaces, use underscore.")
    argparser.add_argument("--model_version", '-v', type=int, default=model_version, help="Model version to use (1 or 2).")
    args = argparser.parse_args()
    
    # Arguments
    drug1 = args.drug1
    drug1 = drug1.replace('_', ' ')
    cell_line = args.cell_line
    drug2 = args.drug2
    if drug2 is not None:
        drug2 = drug2.replace('_', ' ')
    model_version = args.model_version
    
    if model_version == 1:
        run_simulation(drug1, cell_line, cellAE_path, drugAE_path, drug2=drug2)
    elif model_version == 2:
        run_simulation2(drug1, cell_line, cellAE_path, model_path=model_path, drug2=drug2)