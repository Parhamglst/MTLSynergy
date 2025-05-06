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
```

## Usage

Make sure the config file is curated to your needs. Then, navigate to the project's folder in a terminal (CMD on Windows) and execute this command to perform a simulation:

```bash
python3 run_simulation.py 
```

**Flags:**

- **`--drug1`, `-d1`**  
  First drug. If there are spaces in the drug name, replace them with underscores.

- **`--drug2`, `-d2`** *(Optional)*  
  Second drug. If there are spaces in the drug name, replace them with underscores.

- **`--cell_line`, `-c`**  
  Cell line that the drug combination is being tested on.

- **`--model_version`, `-v`**
  `'1'` is the paper's original model, `'2'` is with ChemBERTa and expanded dataset.

## Credits

This project is a fork of [MTLSynergy](https://github.com/TOJSSE-iData/MTLSynergy), originally developed by [nianwuuo](https://github.com/nianwuluo).
