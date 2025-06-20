import re
import ast
import numpy as np

LABELS = ['Loewe', 'RI', 'SynC', 'SenC', 'Bliss', 'ZIP', 'HSA', 'IC50']
COLORS = ['blue', 'green', 'red', 'yellow', 'black', 'purple', 'brown', 'orange']
IC50_MEAN_STD = [18.328315699034636, 137.18380291202186]

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
    import matplotlib.pyplot as plt

    for fold_name, losses in data.items():
        plt.figure(figsize=(18, 12))
        for label, epoch_losses in losses.items():
            if label == 'IC50':
                mean_ic50 = IC50_MEAN_STD[0]
                std_ic50 = IC50_MEAN_STD[1]
                epoch_losses = [loss * std_ic50 + mean_ic50 for loss in epoch_losses]
            plt.plot(epoch_losses[:-1], marker='o', label=label, color=COLORS[LABELS.index(label)])
            plt.axhline(y=epoch_losses[-1], color=COLORS[LABELS.index(label)], linestyle=':', label=f'Test Loss: {epoch_losses[-1]:.2f}')
        plt.title(f"Validation Losses for fold {fold_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"fold_{fold_name}.png")
        # plt.show()

# --- CONFIG ---
LOG_FILE = './slurm-62968403.txt'
NUM_TASKS = 8
with open(LOG_FILE, 'r') as file:
    log_content = file.read()
    data = parse_individual_val_losses(log_content)
    generate_plots(data)