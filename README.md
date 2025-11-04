# MTLSynergy

## Initialization

Create a conda environment using python version 3.13.2 and install the packages inside the requirements.txt file using the setup.sh file.

### Model Weights

Make sure the model weights are stored in the save folder similar to this.

```Markdown
save/
├── MTLSynergy/
│   ├── chp2/
│       ├── weights1.pth
│       ├── weights2.pth
│   ├── model_v1_weights.pth
│   ├── model_v2_weights.pth
│   ├── ...
├── AutoEncoder/
│   ├── CellLineAE_weights.pth
│   ├── DrugAE_weights.pth
│   ├── ...
├── chp4/
│   ├── fold_2.pth
│   ├── fold_2_cellLineAE.pth
│   ├── ...
```

## Usage

Make sure the config file is curated to your needs. Then, navigate to the project's folder in a terminal (CMD on Windows) and execute this command to perform a simulation:

```bash
python3 run_simulation.py 
```

**Flags:**

- **`--drug1`, `-d1`**  
  First drug. If there are spaces in the drug name, replace them with underscores.

- **`--conc1`, `-c1`**  
  Concentration of the first drug in molar (v4 of the model only).
  
- **`--smiles1`, `-s1`**  
  SMILES string of the first drug if drug1 was not found by name. Not needed if the output was generated with drug1.

- **`--drug2`, `-d2`** *(Optional)*  
  Second drug. If there are spaces in the drug name, replace them with underscores.

- **`--conc1`, `-c1`**  
  Concentration of the second drug in molar (v4 of the model only).

- **`--smiles2`, `-s2`** *(Optional)*  
  SMILES string of the second drug if drug2 was not found by name. Not needed if the output was generated with drug2.

- **`--cell_line`, `-c`**  
  Cell line that the drug combination is being tested on.

- **`--model_version`, `-v`**
  `'1'` is the paper's original model, `'2'` is with ChemBERTa and expanded dataset, `'3'` is with ChemBERTa, expanded dataset, and more outputs (Bliss, HSA, ZIP, and IC50), and `'4'` is with concentrations as input and viability as the output.

## Credits

This project is a fork of [MTLSynergy](https://github.com/TOJSSE-iData/MTLSynergy), originally developed by [nianwuuo](https://github.com/nianwuluo).
