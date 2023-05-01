import numpy as np
import matplotlib.pyplot as plt
from dloader2 import HDRDataset
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch
from torch import nn, optim

from model2 import MergeNN
from config import get_configs
from utils import get_transformers


args = get_configs()
input_transformer, output_transformer = get_transformers(args)

train_dataset = HDRDataset(args.train_ds_path, "train", args.train_xsl_path, args.outsize, args.insize, input_transformer,
                           output_transformer, flip_prob=0.5)

# test_dataset = HDRDataset(test_dataset_path, "test", test_excel_output_path, outsize, input_transformer,
#                           output_transformer, flip_prob=0.5)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
# val_loader   = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

model = MergeNN()
model.to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

mu =torch.LongTensor([5000]).to(device)

print("Number of parameters in model is ", sum(p.numel() for p in model.parameters() if p.requires_grad))

logged_loss = 1
for epoch in range(1, args.n_epochs + 1):

    train_losses = []
    train_bar = tqdm(train_loader)
    for data in train_bar:

        H2R_x = torch.cat([data["H2R"], data["GH2R"]], dim=1).to(device)
        L2R_x = torch.cat([data["L2R"], data["GL2R"]], dim=1).to(device)
        tif1_x = torch.cat([data["TIF1"], data["GTIF1"]], dim=1).to(device)
        tif2_x = torch.cat([data["TIF2"], data["GTIF2"]], dim=1).to(device)
        tif3_x = torch.cat([data["TIF3"], data["GTIF3"]], dim=1).to(device)

        # ground_truth=ground_truth.type(torch.FloatTensor).to("cuda")
        # high_to_ref_x = torch.concat([high_to_ref_x, gamma_high_to_ref_x], dim=1).type(torch.FloatTensor)
        y_generated = model(H2R_x, L2R_x, tif1_x, tif2_x, tif3_x)

        # print(y_generated.shape, data['GT'].shape)
        # exit()

        # //TODO check
        # log_y_generated = torch.div(torch.log(1 + mu * (y_generated + 1) / 2.), torch.log(1 + mu) * 2. - 1)
        # log_ground_truth = torch.div(torch.log(1 + mu * (data['GT'].to(device) + 1) / 2.), torch.log(1 + mu) * 2. - 1)

        log_y_generated = torch.div(torch.log(1 + mu * y_generated) , torch.log(1 + mu))
        log_ground_truth = torch.div(torch.log(1 + mu * data['GT'].to(device)), torch.log(1 + mu))

        loss = criterion(y_generated, data['GT'].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_bar.set_description(f'Train Epoch: {epoch} Loss: {np.mean(train_losses):.6f}')

    if np.mean(train_losses) < logged_loss:
        logged_loss = np.mean(train_losses)
        print('Detected')
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch},
                   f'model_1.pt')

    # plt.subplot(121)
    # plt.imshow(data['GT'][0].permute(1,2,0).data.numpy())
    # plt.subplot(122)
    # plt.imshow(y_generated[0].permute(1,2,0).data.cpu().numpy())
    # plt.show()
    # exit()