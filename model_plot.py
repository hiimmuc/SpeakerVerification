import argparse
import importlib
from torchsummary import summary

parser = argparse.ArgumentParser(description="Model plot")
if __name__ == '__main__':
    parser.add_argument('--model', type=str, default='ECAPA_TDNN')
    parser.add_argument('--nOut', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    

    args = parser.parse_args()
    
    model = importlib.import_module('models.' + args.model).__getattribute__('MainModel')(nOut=args.nOut, 
                                                                                          n_mels=80,
                                                                                          augment=False, 
                                                                                          augment_options={}).to('cpu')
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))
    model.to('cpu')
    summary(model, (80, 102), batch_size=args.batch_size, device='cpu')
    