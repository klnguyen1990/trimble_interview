import torch

class Args(object):
    def __init__(self):
        self.device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = 64#16 
        self.input_dim = 3*128*128 
        self.output_dim = 2
        self.epochs = 120
        self.lr = 1e-4#1e-4
        self.dropout_prob = 0.2
        self.seed = 42
     

args = Args()

torch.manual_seed(args.seed)


