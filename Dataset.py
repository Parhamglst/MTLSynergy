from torch.utils.data import Dataset
import numpy as np
from utils.tools import double_data, score_classification


class DrugDataset(Dataset):
    def __init__(self, drug_features):
        self.drug_features = drug_features

    def __len__(self):
        return self.drug_features.shape[0]

    def __getitem__(self, idx):
        drug_item = np.array(self.drug_features.iloc[idx])
        return drug_item, drug_item


class CellLineDataset(Dataset):
    def __init__(self, cell_line_features):
        self.cell_line_features = cell_line_features

    def __len__(self):
        return self.cell_line_features.shape[0]

    def __getitem__(self, idx):
        cell_line_item = np.array(self.cell_line_features.iloc[idx])
        return cell_line_item, cell_line_item


class MTLSynergyDataset(Dataset):
    def __init__(self, drugs, cell_lines, summary, syn_threshold=30, ri_threshold=50):
        self.drugs = drugs
        self.cell_lines = cell_lines
        self.summary = double_data(summary)
        self.syn_threshold = syn_threshold
        self.ri_threshold = ri_threshold

    def __len__(self):
        return self.summary.shape[0]

    def __getitem__(self, idx):
        data = self.summary.iloc[idx]
        (
            d1_idx,
            d2_idx,
            c_idx,
            d1_ri,
            d2_ri,
            syn,
            syn_fold,
            sen_fold_1,
            sen_fold_2,
            _,
        ) = data
        d1 = np.array(self.drugs.iloc[int(d1_idx)])
        d2 = np.array(self.drugs.iloc[int(d2_idx)])
        c_exp = np.array(self.cell_lines.iloc[int(c_idx)])
        syn_label = np.array(score_classification(syn, self.syn_threshold))
        d1_label = np.array(score_classification(d1_ri, self.ri_threshold))
        return (d1, d2, c_exp, np.array(sen_fold_1)), (
            np.array(syn),
            np.array(d1_ri),
            syn_label,
            d1_label,
        )


class MTLSynergy_LeaveCellOutDataset(Dataset):
    def __init__(self, drugs, cell_lines, summary, syn_threshold=30, ri_threshold=50):
        self.drugs = drugs
        self.cell_lines = cell_lines
        self.summary = double_data(summary)
        self.syn_threshold = syn_threshold
        self.ri_threshold = ri_threshold

    def __len__(self):
        return self.summary.shape[0]

    def __getitem__(self, idx):
        data = self.summary.iloc[idx]
        (
            d1_idx,
            d2_idx,
            c_idx,
            d1_ri,
            d2_ri,
            syn,
            syn_fold,
            sen_fold_1,
            sen_fold_2,
            fold,
        ) = data
        d1 = np.array(self.drugs.iloc[int(d1_idx)])
        d2 = np.array(self.drugs.iloc[int(d2_idx)])
        c_exp = np.array(self.cell_lines.iloc[int(c_idx)])
        syn_label = np.array(score_classification(syn, self.syn_threshold))
        d1_label = np.array(score_classification(d1_ri, self.ri_threshold))
        return (d1, d2, c_exp), (np.array(syn), np.array(d1_ri), syn_label, d1_label)


class mainDataset(Dataset):
    def __init__(self, summary, cell_lines, syn_threshold=30, ri_threshold=50):
        self.summary = summary
        self.cell_lines = cell_lines
        self.syn_threshold = syn_threshold
        self.ri_threshold = ri_threshold

    def __len__(self):
        return self.summary.shape[0]

    def __getitem__(self, idx):
        data = self.summary.iloc[idx]
        d_row, d_col, cell_line, synergy_loewe, ri_row, bliss, zip_score, hsa, ic50 = (
            data["drug_row"],
            data["drug_col"],
            data["cell_line_name"],
            data["synergy_loewe"],
            data["ri_row"],
            data["synergy_bliss"],
            data["synergy_zip"],
            data["synergy_hsa"],
            data["ic50_row"],
        )
        d_col = str(d_col)
        c_exp = np.array(self.cell_lines.loc[cell_line])
        syn_label = np.array(
            score_classification(np.float64(synergy_loewe), self.syn_threshold)
        )
        d1_label = np.array(score_classification(np.float64(ri_row), self.ri_threshold))
        return (d_row, d_col, c_exp), (
            np.array(np.float64(synergy_loewe), dtype=np.float64),
            np.array(ri_row, dtype=np.float64),
            syn_label,
            d1_label,
            np.array(np.float64(bliss), dtype=np.float64),
            np.array(np.float64(zip_score), dtype=np.float64),
            np.array(np.float64(hsa), dtype=np.float64),
            np.array(np.float64(ic50), dtype=np.float64),
        )
