#!/usr/bin/env python3

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.data import DataLoader
from torch.optim import Adam
from Dataset import DrugDataset, CellLineDataset
from Models import DrugAE, CellLineAE
from torch.nn import MSELoss
import torch
import time
import pandas as pd
from utils.tools import EarlyStopping, set_seed
from static.constant import DrugAE_OutputDim_Optional, CellAE_OutputDim_Optional, DrugAE_SaveBase, CellAE_SaveBase, \
    DrugAE_Result, CellLineAE_Result
from Preprocess_Data import CELL_LINES_GENES_FILTERED_NORMALIZED
from torch.utils.data import random_split

device = torch.device('cuda')


def fit(model, train_dataloader, train_num, optimizer, criterion):
    print('---Training---')
    model.train()
    train_running_loss = 0.0
    for i, (x, y) in enumerate(train_dataloader):
        data, target = x.float().to(device), y.float().to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += (loss.item() * x.shape[0])
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / train_num
    return train_loss


def validate(model, validation_dataloader, validation_num, criterion):
    model.eval()
    validation_running_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(validation_dataloader):
            data, target = x.float().to(device), y.float().to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            validation_running_loss += (loss.item() * x.shape[0])
        validation_loss = validation_running_loss / validation_num
        return validation_loss


import torch
import time
import os
import logging
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

# --- Configuration ---
# It's good practice to group all your settings in one place.
# This makes them easier to find and modify.

# Data and Paths
CELL_AE_SAVE_BASE = "./models/CellAE_"
CELL_AE_RESULT_LOG = "./results/CellLineAE_Result.log"

# Hyperparameters
HYPERPARAMS = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'epochs': 1500,
    'patience': 100,
}

# Model Architecture Options
CELL_AE_OUTPUT_DIMS = [1024, 128, 256, 512] # Use a more descriptive name

# --- Helper Functions ---

def setup_logger(log_file):
    """Sets up a logger to write to both a file and the console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def train_model_for_dim(output_dim, device, logger):
    """
    Encapsulates the entire training and validation process for a single output dimension.
    This makes the main loop cleaner and the code more modular.
    """
    logger.info(f"---- Starting training for output dimension: {output_dim} ----")
    set_seed(1) # Assuming set_seed is defined elsewhere

    # --- Data Loading ---
    # Moved data loading inside the function to keep the scope clean,
    # though it could be passed as an argument if it's slow to load.
    cell_line_features_data = pd.read_csv(CELL_LINES_GENES_FILTERED_NORMALIZED, index_col=0)
    cell_line_num = cell_line_features_data.shape[0]
    logger.info(f"Number of cell lines: {cell_line_num}")
    
    cell_line_dataset = CellLineDataset(cell_line_features_data) # Assuming CellLineDataset is defined

    train_size = int(0.8 * len(cell_line_dataset))
    val_size = len(cell_line_dataset) - train_size
    train_dataset, val_dataset = random_split(cell_line_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=HYPERPARAMS['batch_size'])


    # --- Model Initialization ---
    model = CellLineAE(output_dim=output_dim).to(device) # Assuming CellLineAE is defined
    optimizer = Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])
    loss_fn = MSELoss(reduction='mean').to(device)
    
    # --- Training Setup ---
    early_stopper = EarlyStopping(patience=HYPERPARAMS['patience']) # Assuming EarlyStopping is defined
    model_save_path = f"{CELL_AE_SAVE_BASE}{output_dim}.pth"

    # Load existing model if it exists
    if os.path.exists(model_save_path):
        logger.info(f"Loading pre-existing model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
        # It's good practice to re-validate to establish a baseline
        initial_loss = validate(model, validation_loader, cell_line_num, loss_fn) # Assuming validate is defined
        early_stopper.best_loss = initial_loss
        logger.info(f"Initial validation loss from loaded model: {initial_loss:.4f}")

    # --- Training Loop ---
    logger.info("Starting training loop...")
    best_val_loss = float("inf")

    for epoch in range(HYPERPARAMS['epochs']):
        train_loss = fit(model, train_loader, cell_line_num, optimizer, loss_fn)
        validation_loss = validate(model, validation_loader, cell_line_num, loss_fn)

        logger.info(f"Epoch {epoch + 1}/{HYPERPARAMS['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {validation_loss:.4f}")

        # Save model if validation improves
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved new best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")


    logger.info(f"Training finished for output_dim={output_dim}. Best validation loss: {best_val_loss:.4f}")
    return best_val_loss


# --- Main Execution ---

def main():
    """Main function to run the experiment."""
    # Ensure result and model directories exist
    os.makedirs(os.path.dirname(CELL_AE_SAVE_BASE), exist_ok=True)
    os.makedirs(os.path.dirname(CELL_AE_RESULT_LOG), exist_ok=True)

    logger = setup_logger(CELL_AE_RESULT_LOG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    for dim in CELL_AE_OUTPUT_DIMS:
        train_model_for_dim(dim, device, logger)
        logger.info("-" * 50) # Separator for clarity in logs

if __name__ == "__main__":
    # This is a standard Python convention to make the script runnable
    main()
