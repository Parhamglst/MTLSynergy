import torch
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d, Dropout, Softmax
from static.constant import DrugAE_InputDim, DrugAE_OutputDim, CELLAE_InputDim, CellAE_OutputDim, MTLSynergy_InputDim
from utils.tools import init_weights
from transformers import AutoTokenizer, AutoModel


class DrugAE(Module):
    def __init__(self, input_dim=DrugAE_InputDim, output_dim=DrugAE_OutputDim):
        super(DrugAE, self).__init__()
        if output_dim == 32 or output_dim == 64:
            hidden_dim = 256
        elif output_dim == 128 or output_dim == 256:
            hidden_dim = 512
        else:
            hidden_dim = 1024
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_dim, output_dim),
        )
        self.decoder = Sequential(
            Linear(output_dim, hidden_dim),
            ReLU(True),
            Linear(hidden_dim, input_dim),
        )
        init_weights(self._modules)

    def forward(self, input):
        x = self.encoder(input)
        y = self.decoder(x)
        return y


class CellLineAE(Module):
    def __init__(self, input_dim=CELLAE_InputDim, output_dim=CellAE_OutputDim):
        super(CellLineAE, self).__init__()
        if output_dim == 128 or output_dim == 256:
            hidden_dim = 512
        elif output_dim == 512:
            hidden_dim = 1024
        else:
            hidden_dim = 4096
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_dim, output_dim),
        )
        self.decoder = Sequential(
            Linear(output_dim, hidden_dim),
            ReLU(True),
            Linear(hidden_dim, input_dim)
        )
        init_weights(self._modules)

    def forward(self, input):
        x = self.encoder(input)
        y = self.decoder(x)
        return y


class MTLSynergy(Module):
    def __init__(self, hidden_neurons, input_dim=MTLSynergy_InputDim):
        super(MTLSynergy, self).__init__() 
        self.drug_cell_line_layer = Sequential(
            Linear(input_dim, hidden_neurons[0]),
            BatchNorm1d(hidden_neurons[0]),
            ReLU(True),
            Linear(hidden_neurons[0], hidden_neurons[1]),
            ReLU(True)
        )
        self.synergy_layer = Sequential(
            Linear(2 * hidden_neurons[1], hidden_neurons[2]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[2], 128),
            ReLU(True)
        )
        self.sensitivity_layer = Sequential(
            Linear(hidden_neurons[1], hidden_neurons[3]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[3], 64),
            ReLU(True)
        )
        self.synergy_out_1 = Linear(128, 1)
        self.synergy_out_2 = Sequential(Linear(128, 2), Softmax(dim=1))
        self.sensitivity_out_1 = Linear(64, 1)
        self.sensitivity_out_2 = Sequential(Linear(64, 2), Softmax(dim=1))
        init_weights(self._modules)

    def forward(self, d1, d2, c_exp):
        d1_c = self.drug_cell_line_layer(torch.cat((d1, c_exp), 1))
        d2_c = self.drug_cell_line_layer(torch.cat((d2, c_exp), 1))
        d1_sen = self.sensitivity_layer(d1_c)
        syn = self.synergy_layer(torch.cat((d1_c, d2_c), 1))
        syn_out_1 = self.synergy_out_1(syn)
        syn_out_2 = self.synergy_out_2(syn)
        d1_sen_out_1 = self.sensitivity_out_1(d1_sen)
        d1_sen_out_2 = self.sensitivity_out_2(d1_sen)
        return syn_out_1.squeeze(-1), d1_sen_out_1.squeeze(-1), syn_out_2, d1_sen_out_2


class MTLSynergy2(Module):
    def __init__(self, hidden_neurons, input_dim=MTLSynergy_InputDim):
        super(MTLSynergy2, self).__init__() 
        self.drug_cell_line_layer = Sequential(
            Linear(input_dim, hidden_neurons[0]),
            BatchNorm1d(hidden_neurons[0]),
            ReLU(True),
            Linear(hidden_neurons[0], hidden_neurons[1]),
            ReLU(True)
        )
        self.synergy_layer = Sequential(
            Linear(hidden_neurons[1], hidden_neurons[2]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[2], 128),
            ReLU(True)
        )
        self.sensitivity_layer = Sequential(
            Linear(hidden_neurons[1], hidden_neurons[3]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[3], 128),
            ReLU(True)
        )
        
        # Drug synergy output layers
        self.synergy_out_1 = Linear(128, 1)
        self.synergy_out_2 = Sequential(Linear(128, 2), Softmax(dim=1))
        
        self.bliss_out = Linear(128, 1)
        
        self.zip_out = Linear(128, 1)
        
        self.hsa_out = Linear(128, 1)
        
        # Drug sensitivity output layers
        self.sensitivity_out_1 = Linear(128, 1)
        self.sensitivity_out_2 = Sequential(Linear(128, 2), Softmax(dim=1))

        self.ic50 = Linear(128, 1)

        init_weights(self._modules)

    def forward(self, d1_embeddings, d2_embeddings, c_exp):
        shared_output = self.drug_cell_line_layer(torch.cat((d1_embeddings, d2_embeddings, c_exp), 1))

        d1_sen = self.sensitivity_layer(shared_output)
        syn = self.synergy_layer(shared_output)

        syn_out_1 = self.synergy_out_1(syn)
        syn_out_2 = self.synergy_out_2(syn)
        bliss_out = self.bliss_out(syn)
        zip_out = self.zip_out(syn)
        hsa_out = self.hsa_out(syn)
        
        d1_sen_out_1 = self.sensitivity_out_1(d1_sen)
        d1_sen_out_2 = self.sensitivity_out_2(d1_sen)
        ic50_out = self.ic50(d1_sen)
        
        return syn_out_1.squeeze(-1), d1_sen_out_1.squeeze(-1), syn_out_2, d1_sen_out_2, bliss_out.squeeze(-1), zip_out.squeeze(-1), hsa_out.squeeze(-1), ic50_out.squeeze(-1)
    

class MTLSynergy3(Module):
    def __init__(self, hidden_neurons, input_dim=MTLSynergy_InputDim + 2):
        super(MTLSynergy3, self).__init__()
        self.drug_cell_line_layer = Sequential(
            Linear(input_dim, hidden_neurons[0]),
            BatchNorm1d(hidden_neurons[0]),
            ReLU(True),
            Linear(hidden_neurons[0], hidden_neurons[1]),
            ReLU(True)
        )
        self.combo_layer = Sequential(
            Linear(hidden_neurons[1], hidden_neurons[2]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[2], 128),
            ReLU(True)
        )
        self.mono_layer = Sequential(
            Linear(hidden_neurons[1], hidden_neurons[3]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[3], 128),
            ReLU(True)
        )
        
        # Drug combination viability output layer
        self.combo_out = Linear(128, 1)
        
        # Single agent viability output layer
        self.mono_out = Linear(128, 1)

        init_weights(self._modules)

    def forward(self, d1_embeddings, d1_conc, d2_embeddings, d2_conc, c_exp):
        shared_layer = self.drug_cell_line_layer(torch.cat((d1_embeddings, d1_conc, d2_embeddings, d2_conc, c_exp), 1))

        d1_sen = self.mono_layer(shared_layer)
        syn = self.combo_layer(shared_layer)

        combo_out = self.combo_out(syn)

        mono_out = self.mono_out(d1_sen)

        return  mono_out.squeeze(-1), combo_out.squeeze(-1)
    
def chemprop_chemberta():
    pass