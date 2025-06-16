import re
import ast
import numpy as np

LABELS = ['Loewe', 'RI', 'SynC', 'SenC', 'Bliss', 'ZIP', 'HSA', 'IC50']

def parse_individual_val_losses(file_content):
    """
    Parses individual validation losses from the given text content into a dictionary.

    Args:
        file_content (str): The string content of the log file.

    Returns:
        dict: A dictionary where keys are hyperparameter configurations (as strings)
              and values are dictionaries. These inner dictionaries have keys
              like "Inner Fold X" and values are lists of lists, where each
              inner list contains the 8 individual validation losses (floats)
              for an epoch in that fold.
    """
    results = {}
    current_hyper_params_key = None
    current_fold_data = None

    # Split by hyperparameter sections first
    hyper_param_sections = re.split(r"(--- Hyper parameters \d+:.*? ---)", file_content)

    for section_idx, section_text in enumerate(hyper_param_sections):
        if not section_text.strip():
            continue

        hyper_match = re.match(r"--- Hyper parameters \d+:(.*?) ---", section_text)
        if hyper_match:
            current_hyper_params_str = hyper_match.group(1).strip()
            try:
                current_hyper_params_key = str(ast.literal_eval(current_hyper_params_str))
            except (ValueError, SyntaxError):
                current_hyper_params_key = current_hyper_params_str
            results[current_hyper_params_key] = {}
            # The actual content for this hyper_param_section is what follows its declaration
            # or what's between this and the next hyper_param_section declaration.
            # We'll process the content associated with this current_hyper_params_key
            # by looking at the *next* part of the split if it's not a hyper_param_header.
            # This logic is a bit tricky because split keeps delimiters.
            # A simpler way is to process the content *after* the hyper_match within the *same* section_text.
            content_for_current_hyper_param = section_text[hyper_match.end():]
        elif current_hyper_params_key: # If we are already inside a hyper_param block
            content_for_current_hyper_param = section_text
        else: # Skip content before the first hyper_param block
            continue
            
        if current_hyper_params_key and content_for_current_hyper_param.strip():
            current_fold_data_for_hyper_param = results[current_hyper_params_key]
            
            # Split by inner folds within the current hyperparameter section's content
            inner_fold_sections = re.split(r"(Inner Fold \d+)", content_for_current_hyper_param)
            
            active_epoch_losses_list = []
            current_fold_name = None

            for part in inner_fold_sections:
                if not part.strip():
                    continue
                
                fold_match = re.match(r"Inner Fold (\d+)", part)
                if fold_match:
                    # If there was a previous fold being processed, save its data
                    if current_fold_name and active_epoch_losses_list:
                        current_fold_data_for_hyper_param[current_fold_name] = active_epoch_losses_list
                    
                    fold_number = fold_match.group(1)
                    current_fold_name = f"Inner Fold {fold_number}"
                    active_epoch_losses_list = [] # Reset for the new fold
                else:
                    # This part contains epoch data for the current_fold_name
                    if current_fold_name: # Ensure we are logically inside a fold
                        # Find all "Val Loss: [...]" lines
                        # Using re.DOTALL is not ideal here if multiple Val Loss lines could appear without an Epoch delimiter
                        # Assuming one Val Loss line per Epoch block as per file structure
                        
                        # Split the part by "Epoch" to process each epoch individually
                        epoch_blocks = re.split(r"(Epoch \d+:)", part)
                        for epoch_block_idx, epoch_data_segment in enumerate(epoch_blocks):
                            if epoch_data_segment.startswith("Epoch") or not epoch_data_segment.strip():
                                continue # Skip the "Epoch X:" delimiter itself or empty parts

                            # Corrected regex to find "Val Loss: [...]"
                            loss_match = re.search(r"Val Loss:\s*\[(.*?)\]", epoch_data_segment)
                            if loss_match:
                                losses_str = loss_match.group(1).strip()
                                individual_losses_str = losses_str.split() # Split by space
                                try:
                                    epoch_losses = [float(loss) for loss in individual_losses_str]
                                    active_epoch_losses_list.append(epoch_losses)
                                except ValueError as e:
                                    print(f"Warning: Could not convert one of the losses in '{individual_losses_str}' to float for {current_hyper_params_key}, {current_fold_name}. Error: {e}")
            
            # Save the last fold's data for the current hyperparameter set
            if current_fold_name and active_epoch_losses_list:
                 current_fold_data_for_hyper_param[current_fold_name] = active_epoch_losses_list

    # Clean up any hyperparameter entries that didn't end up with fold data
    results = {k: v for k, v in results.items() if v}
    for hp_key, folds in list(results.items()): # Iterate over a copy for modification
        results[hp_key] = {fk: fv for fk, fv in folds.items() if fv} # Clean empty folds
        if not results[hp_key]: # If all folds were empty for this hp_key
            del results[hp_key]
            
    return results

def generate_plots(data):
    """
    Generates plots for the parsed validation losses data.

    Args:
        data (dict): The dictionary containing hyperparameter configurations and their validation losses.
    """
    import matplotlib.pyplot as plt

    for hyper_params, folds in data.items():
        for fold_name, losses in folds.items():
            plt.figure(figsize=(18, 12))
            losses = np.transpose(losses)
            for epoch_losses, label in zip(losses, LABELS):
                if label == 'IC50':
                    plt.plot(epoch_losses, marker='o', label=label)
            plt.title(f"Validation Losses for {hyper_params} - {fold_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.legend()
            plt.grid()
            plt.savefig(f"IC50_{hyper_params}_{fold_name}.png")
            # plt.show()

# --- CONFIG ---
LOG_FILE = './slurm-62440182.txt'
NUM_TASKS = 8
with open(LOG_FILE, 'r') as file:
    log_content = file.read()
    data = parse_individual_val_losses(log_content)
    generate_plots(data)