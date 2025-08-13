import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import pandas as pd
import torch.distributed as dist


class EarlyStopping():
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, chemberta_model, cellLineEncoder, save_path, rank):
        if rank == 0:
            if self.best_loss == None:
                self.best_loss = val_loss
                torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), save_path)
                torch.save(chemberta_model.module.state_dict() if hasattr(chemberta_model, "module") else chemberta_model.state_dict(), save_path.replace('.pt', '_chemberta.pt'))
                torch.save(cellLineEncoder.module.state_dict() if hasattr(cellLineEncoder, "module") else cellLineEncoder.state_dict(), save_path.replace('.pt', '_cellLineAE.pt'))
                print(f"INFO: Early stopping initialized with best loss {self.best_loss}")
            elif self.best_loss - val_loss > self.min_delta:
                self.best_loss = val_loss
                # reset counter if validation loss improves
                self.counter = 0
                # save weights
                torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), save_path)
                torch.save(chemberta_model.module.state_dict() if hasattr(chemberta_model, "module") else chemberta_model.state_dict(), save_path.replace('.pt', '_chemberta.pt'))
                torch.save(cellLineEncoder.module.state_dict() if hasattr(cellLineEncoder, "module") else cellLineEncoder.state_dict(), save_path.replace('.pt', '_cellLineAE.pt'))
                print(f"INFO: Early stopping best loss updated to {self.best_loss}")
            elif self.best_loss - val_loss < self.min_delta:
                self.counter += 1
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
                if self.counter >= self.patience:
                    print('INFO: Early stopping')
                    self.early_stop = True
        if dist.is_initialized():
            dist.barrier()


def set_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed(seed)  # set random seed for GPU


def double_data(data):
    double_summary = pd.DataFrame()
    double_summary['drug_row_idx'] = data['drug_col_idx']
    double_summary['drug_col_idx'] = data['drug_row_idx']
    double_summary['cell_line_idx'] = data['cell_line_idx']
    double_summary['ri_row'] = data['ri_col']
    double_summary['ri_col'] = data['ri_row']
    double_summary['synergy_loewe'] = data['synergy_loewe']
    double_summary['syn_fold'] = data['syn_fold']
    double_summary['sen_fold_1'] = data['sen_fold_2']
    double_summary['sen_fold_2'] = data['sen_fold_1']
    result = pd.concat([data, double_summary], axis=0)
    return result


def init_weights(modules, exclude_prefix='chemBERTaModel'):
    for n, m in modules.items():
        if exclude_prefix in n:
            continue  # Skip ChemBERTa layers
        if isinstance(m, torch.nn.Sequential):
            for layer in m:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, 1.0)
                    torch.nn.init.constant_(layer.bias, 0.0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)


def filter1(data, fold_test, flag=0):
    x, y = data
    d1_features, d2_features, c_features, sen_fold = x
    y1, y2, y3, y4 = y
    if flag == 0:
        remain = sen_fold != fold_test
    else:
        remain = sen_fold == fold_test
    return (d1_features[remain], d2_features[remain], c_features[remain]), (
        y1[remain], y2[remain], y3[remain], y4[remain])


def filter2(data, fold_test, flag=0):
    x, y = data
    d1_features, d2_features, c_features, sen_fold = x
    y1, y2 = y
    if flag == 0:
        remain = np.where(sen_fold != fold_test)
    else:
        remain = np.where(sen_fold == fold_test)
    return (d1_features[remain], d2_features[remain], c_features[remain]), (y1[remain], y2[remain])


def score_classification(x, threshold):
    return 1 if x > threshold else 0


def calculate(result, name, fold_num, save_path):
    tol_result = {}
    keys = result[0].keys()
    for key in keys:
        tol_result[key] = []
    for i in range(fold_num):
        for key in keys:
            tol_result[key].append(result[i][key])
    print(str(name) + " result :")
    with open(save_path, 'a') as file:
        file.write(str(name) + " result :\n")
    for key in keys:
        print(str(key) + ": " + str([np.mean(tol_result[key]), np.std(tol_result[key])]))
        with open(save_path, 'a') as file:
            file.write(str(key) + ": " + str([np.mean(tol_result[key]), np.std(tol_result[key])]) + "\n")



class CategoricalCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true, y_pred.shape[1])
        y_pred = torch.clamp(y_pred, 1e-9, 1.0)
        tol_loss = -torch.sum(y_true * torch.log(y_pred), dim=1)
        loss = torch.mean(tol_loss, dim=0)
        return loss
    

class GradNormController:
    def __init__(self, alpha=1.5):
        self.alpha = alpha
        self.initial_losses = None
        self.epsilon = 1e-8

    def compute_gradnorm_loss(self, shared_params, task_losses, task_weights):
        # Get gradient norms for each task
        norms = []
        target_device = task_losses[0].device if task_losses else task_weights[0].device # Fallback if task_losses is empty
        task_weights_on_device = [w.to(target_device) for w in task_weights]
        for i, loss in enumerate(task_losses):
            grad = torch.autograd.grad(
                task_weights_on_device[i] * loss, shared_params,
                retain_graph=True, create_graph=True, allow_unused=True
            )
            flat_grad = torch.cat([g.flatten() for g in grad if g is not None])
            norm = torch.linalg.norm(flat_grad)
            norms.append(norm)

        norms = torch.stack(norms)
        norms_mean = norms.mean()

        if self.initial_losses is None:
            self.initial_losses = torch.stack([l.detach() for l in task_losses])

        current_losses = torch.stack([l.detach() for l in task_losses])
        loss_ratios = current_losses / (self.initial_losses + self.epsilon)
        loss_ratios = torch.clamp(loss_ratios, min=0.0, max=100.0)
        mean_loss_ratio = loss_ratios.mean() + self.epsilon
        inverse_train_rates = loss_ratios / mean_loss_ratio

        target_norms = norms_mean * (inverse_train_rates ** self.alpha)
        
        grad_norm_loss = F.l1_loss(norms, target_norms.detach())

        return grad_norm_loss
