import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from model2 import MergeNN
from config import get_configs
from dloader2 import HDRDataset


args = get_configs()

input_transformer = transforms.Compose(
    [transforms.Resize(args.insize),
     transforms.ToTensor()])

output_transformer = transforms.Compose(
    [transforms.ToTensor()])

train_dataset = HDRDataset(args.train_ds_path, "train", args.train_xsl_path, args.outsize, args.insize, input_transformer,
                           output_transformer, flip_prob=0.5)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

model = MergeNN()
saved_dict = torch.load( f'model_1.pt')
model.load_state_dict(saved_dict['model_state_dict'])
model.to(device)

mu =torch.LongTensor([5000]).to(device)

for epoch in range(1, args.n_epochs + 1):

    train_losses = []
    train_bar = tqdm(train_loader)
    for data in train_bar:

        H2R_x = torch.cat([data["H2R"], data["GH2R"]], dim=1).to(device)
        L2R_x = torch.cat([data["L2R"], data["GL2R"]], dim=1).to(device)
        tif1_x = torch.cat([data["TIF1"], data["GTIF1"]], dim=1).to(device)
        tif2_x = torch.cat([data["TIF2"], data["GTIF2"]], dim=1).to(device)
        tif3_x = torch.cat([data["TIF3"], data["GTIF3"]], dim=1).to(device)

        y_generated = model(H2R_x, L2R_x, tif1_x, tif2_x, tif3_x)

        log_y_generated = y_generated #torch.div(torch.log(1 + mu * y_generated) , torch.log(1 + mu))
        log_ground_truth = data['GT'].to(device) #torch.div(torch.log(1 + mu * data['GT'].to(device)), torch.log(1 + mu))


        plt.subplot(121)
        plt.imshow(log_ground_truth[0].permute(1,2,0).data.cpu().numpy())
        plt.subplot(122)
        plt.imshow(log_y_generated[0].permute(1,2,0).data.cpu().numpy())
        plt.show()
        exit()