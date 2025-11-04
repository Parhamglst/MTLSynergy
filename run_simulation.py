import torch
import numpy as np
from Models_old import MTLSynergy, MTLSynergy2, ChemBERTaEncoder, CellLineAE, DrugAE
import Models
from transformers import RobertaTokenizer, AutoModel
from Summarize_logs import remove_module_prefix
from Summarize_viability_logs import MEAN_STDS
from static.constant import MTLSynergy_InputDim, Ver3_MTLSynergy_InputDim, Ver3_CellAE_OutputDim, CellAE_OutputDim, DrugAE_OutputDim
from Preprocess_Data_Old import prepareCellLine, prepareDrug
from Preprocess_Data import get_smiles, get_smiles_almanac, ALMANAC_NAMES
import argparse
import yaml
import pandas as pd

WEIGHTS = [[4096, 2048, 1024, 1024], [2048, 1024, 512, 512]]
IC50_ROW_MEAN =  18.328315699034636
IC50_ROW_STD = 137.18380291202186

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_simulation(drug1, cell_line, cell_line_AE_path, drug_AE_path, model_path = './save/MTLSynergy/fold_4.pth', drug2=None, smiles1=False, smiles2=False):
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

    if not smiles1:
        drug1_smiles = get_smiles(drug1, {})
    else:
        drug1_smiles = drug1
    if not smiles2 and drug2 is not None:
        drug2_smiles = get_smiles(drug2, {})
    else:
        drug2_smiles = drug2
    
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

    

def run_simulation2(drug1, cell_line, cell_line_AE_path, model_path='./save/MTLSynergy/chp2/fold_0.pth', drug2=None, smiles1=False, smiles2=False):
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
    cellLineAE = CellLineAE(output_dim=CellAE_OutputDim).to(device)
    mtlSynergy = MTLSynergy2([8192, 4096, 4096, 2048], input_dim=MTLSynergy_InputDim).to(device)
    mtlSynergy = torch.nn.DataParallel(mtlSynergy)

    cellLineAE.load_state_dict(torch.load(cell_line_AE_path))
    cellLineAE.eval()
    mtlSynergy.load_state_dict(torch.load(model_path))
    mtlSynergy.eval()

    if not smiles1:
        drug1_smiles = get_smiles(drug1, {})
    else:
        drug1_smiles = drug1
    if not smiles2 and drug2 is not None:
        drug2_smiles = get_smiles(drug2, {})
    else:
        drug2_smiles = drug2

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

def run_simulation3(drug1, cell_line, cell_line_AE_path, model_path='./save/MTLSynergy/chp3/fold_3.pth', drug2=None, smiles1=False, smiles2=False):
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
    cellLineAE = CellLineAE(output_dim=Ver3_CellAE_OutputDim).to(device)
    mtlSynergy = Models.MTLSynergy2([8192, 4096, 4096, 2048], input_dim=Ver3_MTLSynergy_InputDim).to(device)
    # mtlSynergy = torch.nn.DataParallel(mtlSynergy)

    cellLineAE.load_state_dict(torch.load(cell_line_AE_path))
    cellLineAE.eval()
    mtl_state_dict = torch.load(model_path)
    mtlSynergy.load_state_dict(mtl_state_dict)
    mtlSynergy.eval()

    if not smiles1:
        drug1_smiles = get_smiles(drug1, {})
    else:
        drug1_smiles = drug1
    if not smiles2 and drug2 is not None:
        drug2_smiles = get_smiles(drug2, {})
    else:
        drug2_smiles = drug2

    if drug1_smiles is None:
        raise ValueError(f"Drug1 SMILES not found for {drug1}")
    if drug2_smiles is None and drug2 is not None:
        raise ValueError(f"Drug2 SMILES not found for {drug2}")

    cell_line_features = prepareCellLine(cell_line)

    syn_out, drug1_sen_out, syn_out_class, d1_sen_out_class, bliss, zip, hsa, ic50 = _sim(tokenizer, chemBERTaEncoder, cellLineAE, mtlSynergy, drug1_smiles, drug2_smiles, cell_line_features)
    drug2_sen_out = None
    if drug2 is not None:
        syn2_out, drug2_sen_out, syn2_out_class, d2_sen_out_class, bliss2, zip2, hsa2, ic50_2 = _sim(tokenizer, chemBERTaEncoder, cellLineAE, mtlSynergy, drug2_smiles, drug1_smiles, cell_line_features)
        syn_out = (syn_out + syn2_out) / 2
        bliss = (bliss + bliss2) / 2
        zip = (zip + zip2) / 2
        hsa = (hsa + hsa2) / 2
    syn_out = syn_out.cpu().numpy()
    drug1_sen_out = drug1_sen_out.cpu().numpy()
    ic50 = ic50.cpu().numpy()
    if drug2 is not None:
        drug2_sen_out = drug2_sen_out.cpu().numpy()
        ic50_2 = ic50_2.cpu().numpy()
        bliss = bliss.cpu().numpy()
        zip = zip.cpu().numpy()
        hsa = hsa.cpu().numpy()
        display_output_v3(drug1_sen_out= drug1_sen_out, drug1_ic50_out=ic50, drug2_sen_out=drug2_sen_out, drug2_ic50_out=ic50_2, syn_out=syn_out, bliss=bliss, zip=zip, hsa=hsa)
        return
    syn_out_class = syn_out_class.cpu().numpy()
    d1_sen_out_class = d1_sen_out_class.cpu().numpy()
    display_output_v3(drug1_sen_out= drug1_sen_out, drug1_ic50_out=ic50)

def run_simulation_viability(drug1, conc1, cell_line, chemberta_path, cell_line_AE_path, model_path='./save/chp4/fold_2.pth', drug2=None, conc2=None, smiles1=False, smiles2=False):
    def _sim(tokenizer, chemBERTaEncoder, cellLineAE, model, drug1_smiles, drug1_conc, drug2_smiles, drug2_conc, cell_line_features):
        # Convert the input data to tensors
        cell_line_features = torch.tensor(cell_line_features, dtype=torch.float32, device=device)

        # Run the simulation
        with torch.no_grad():
            d1_embeddings = tokenizer(drug1_smiles[1], return_tensors="pt", padding="max_length", truncation=True).to(device)
            d2_embeddings = tokenizer(drug2_smiles[1], return_tensors="pt", padding='max_length', truncation=True).to(device) if drug2_smiles else d1_embeddings
            token_type_ids1 = torch.zeros_like(d1_embeddings["input_ids"].squeeze(0))
            token_type_ids2 = torch.zeros_like(d2_embeddings["input_ids"].squeeze(0))
            d1_encoded = chemBERTaEncoder(input_ids=d1_embeddings['input_ids'], attention_mask=d1_embeddings['attention_mask'], token_type_ids=token_type_ids1).last_hidden_state[:,0,:]
            d2_encoded = chemBERTaEncoder(input_ids=d2_embeddings['input_ids'], attention_mask=d2_embeddings['attention_mask'], token_type_ids=token_type_ids2).last_hidden_state[:,0,:]
            
            cell_line_encoded = cellLineAE.encoder(cell_line_features)
            d1_con = torch.tensor([[drug1_conc]], dtype=torch.float32, device=device)
            d2_con = torch.tensor([[drug2_conc]], dtype=torch.float32, device=device) if drug2_conc is not None else torch.tensor([[0.0]], dtype=torch.float32, device=device)
            output = model(d1_encoded.to(device), d1_con.to(device), d2_encoded.to(device), d2_con.to(device), cell_line_encoded.unsqueeze(0).to(device))

        return output
    
    chemBERTaEncoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", attn_implementation="eager").to(device)
    tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    cellLineAE = CellLineAE(output_dim=CellAE_OutputDim).to(device)
    mtlSynergy = Models.MTLSynergy3(WEIGHTS[0], input_dim=MTLSynergy_InputDim+2).to(device)

    cellLineAE_state_dict = torch.load(cell_line_AE_path, map_location=device)
    cellLineAE_state_dict = remove_module_prefix(cellLineAE_state_dict)
    chemberta_state_dict = torch.load(chemberta_path, map_location=device)
    chemberta_state_dict = remove_module_prefix(chemberta_state_dict)
    mtl_state_dict = torch.load(model_path, map_location=device)
    mtl_state_dict = remove_module_prefix(mtl_state_dict)


    chemBERTaEncoder.load_state_dict(chemberta_state_dict)
    chemBERTaEncoder.eval()
    cellLineAE.load_state_dict(cellLineAE_state_dict)
    cellLineAE.eval()
    mtlSynergy.load_state_dict(mtl_state_dict)
    mtlSynergy.eval()
    
    mapping = pd.read_csv(ALMANAC_NAMES, delimiter='\t', dtype={"NSC": str, "Name": str})
    mapping['NSC'] = mapping['NSC'].str.strip()
    mapping.set_index('NSC', inplace=True)
    mapping = mapping.groupby('NSC')['Name'].apply(list).to_dict()

    if not smiles1:
        drug1_smiles = get_smiles_almanac(drug1, {}, mapping)
    else:
        drug1_smiles = drug1
    if not smiles2 and drug2 is not None:
        drug2_smiles = get_smiles_almanac(drug2, {}, mapping)
    else:
        drug2_smiles = drug2

    if drug1_smiles is None:
        raise ValueError(f"Drug1 SMILES not found for {drug1}")
    if drug2_smiles is None and drug2 is not None:
        raise ValueError(f"Drug2 SMILES not found for {drug2}")

    cell_line_features = prepareCellLine(cell_line)

    unnormalize = lambda x, mean, std: x * std + mean
    mono_pred, combo_pred = _sim(tokenizer, chemBERTaEncoder, cellLineAE, mtlSynergy, drug1_smiles, conc1, drug2_smiles, conc2, cell_line_features)
    
    mono_pred = unnormalize(mono_pred, MEAN_STDS[0][0], MEAN_STDS[0][1])
    combo_pred = unnormalize(combo_pred, MEAN_STDS[0][0], MEAN_STDS[0][1])

    viability_pred = mono_pred
    if drug2 is not None:
        mono_pred2, combo_pred2 = _sim(tokenizer, chemBERTaEncoder, cellLineAE, mtlSynergy, drug2_smiles, conc2, drug1_smiles, conc1, cell_line_features)
        mono_pred2 = unnormalize(mono_pred2, MEAN_STDS[0][0], MEAN_STDS[0][1])
        combo_pred2 = unnormalize(combo_pred2, MEAN_STDS[0][0], MEAN_STDS[0][1])
        viability_pred = (combo_pred + combo_pred2) / 2
    viability_pred = viability_pred.cpu().numpy()
    if drug2 is not None:
        print(f"Combination Viability of {drug1} at {conc1:.4f} + {drug2} at {conc2:.4f} on {cell_line}:", viability_pred)
        return
    if drug2 is None:
        print(f"Monotherapy Viability of {drug1} at {conc1:.4f} on {cell_line}:", viability_pred)
        return

def display_output(syn_out, drug1_sen_out, drug2_sen_out=None):
    print("Synergy Loewe:", syn_out)
    print("Drug1 Sensitivity:", drug1_sen_out)
    if drug2_sen_out is not None:
        print("Drug2 Sensitivity:", drug2_sen_out)

def display_output_v3(drug1_sen_out, drug1_ic50_out, drug2_sen_out=None, drug2_ic50_out=None, syn_out=None, bliss=None, zip=None, hsa=None):
    if syn_out is not None:
        print("Synergy Loewe:", syn_out)
    print("Drug1 Sensitivity:", drug1_sen_out)
    print("Drug1 IC50:", drug1_ic50_out * IC50_ROW_STD + IC50_ROW_MEAN)
    if drug2_sen_out is not None:
        print("Drug2 Sensitivity:", drug2_sen_out)
        print("Drug2 IC50:", drug2_ic50_out * IC50_ROW_STD + IC50_ROW_MEAN)
    if bliss is not None:
        print("Bliss Independence:", bliss)
    if zip is not None:
        print("Zero Interaction Potency (ZIP):", zip)
    if hsa is not None:
        print("Highest Single Agent (HSA):", hsa)


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_version = config.get('model_version', 4)
    model_path = config.get('model_path', 'save/chp4/fold_2.pth')
    drugAE_path = config.get('drugAE_path', 'save/AutoEncoder/DrugAE_128.pth')
    cellAE_path = config.get('cellAE_v4_path', 'save/AutoEncoder/CellLineAE_256.pth')
    chemberta_v4_path = config.get('chemberta_v4_path', 'save/chp4/fold_2_chemberta.pth')
    cellAE_v4_path = config.get('cellAE_v4_path', 'save/chp4/fold_2_cellLineAE.pth')

    # Argument parser
    argparser = argparse.ArgumentParser(description="Run simulation for drug synergy prediction.")
    group1 = argparser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--drug1", '-d1', type=str, help="Name of the first drug. If it includes spaces, use underscore.")
    group1.add_argument("--smiles1", '-s1', type=str, help="SMILES string of the drug.")
    group2 = argparser.add_mutually_exclusive_group()
    group2.add_argument("--smiles2", '-s2', type=str, help="SMILES string of the second drug (optional).")
    group2.add_argument("--drug2", '-d2', type=str, help="Name of the second drug (optional). If it includes spaces, use underscore.")
    argparser.add_argument("--conc1", '-c1', type=float, required=True if model_version == 4 else False, help="Concentration of the first drug (only for model version 4).")
    argparser.add_argument("--conc2", '-c2', type=float, required=False, help="Concentration of the second drug (only for model version 4).")
    argparser.add_argument("--cell_line", '-c', type=str, required=True, help="Name of the cell line.")
    argparser.add_argument("--model_version", '-v', type=int, default=model_version, help="Model version to use (1, 2, 3, or 4).")
    args = argparser.parse_args()
    
    # Arguments
    drug1 = args.drug1 if args.drug1 else args.smiles1
    if drug1 is None:
        raise ValueError("Please provide either --drug1 or --smiles1.")
    drug1 = drug1.replace('_', ' ')
    drug1_conc = args.conc1
    cell_line = args.cell_line
    drug2 = args.drug2 if args.drug2 else args.smiles2
    if drug2 is not None:
        drug2 = drug2.replace('_', ' ')
        drug2_conc = args.conc2
    model_version = args.model_version
    
    if model_version == 1:
        drugAE_path = drugAE_path + str(DrugAE_OutputDim) + '.pth'
        cellAE_path = cellAE_path + str(CellAE_OutputDim) + '.pth'
        run_simulation(drug1, cell_line, cellAE_path, drugAE_path, drug2=drug2, smiles1 = True if args.smiles1 else False, smiles2 = True if args.smiles2 else False)
    elif model_version == 2:
        cellAE_path = cellAE_path + str(CellAE_OutputDim) + '.pth'
        run_simulation2(drug1, cell_line, cellAE_path, model_path=model_path, drug2=drug2, smiles1 = True if args.smiles1 else False, smiles2 = True if args.smiles2 else False)
    elif model_version == 3:
        cellAE_path = cellAE_path + str(Ver3_CellAE_OutputDim) + '.pth'
        run_simulation3(drug1, cell_line, cellAE_path, model_path=model_path, drug2=drug2, smiles1 = True if args.smiles1 else False, smiles2 = True if args.smiles2 else False)
    elif model_version == 4:
        cellAE_path = cellAE_path
        run_simulation_viability(drug1, drug1_conc, cell_line, chemberta_v4_path, cellAE_path, model_path=model_path, drug2=drug2, conc2=drug2_conc, smiles1 = True if args.smiles1 else False, smiles2 = True if args.smiles2 else False)