import torch

from torch.utils.data import DataLoader,TensorDataset
from model import *
from torch import optim
from tqdm import tqdm
from dataloader import *
learning_rate = 0.001
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
num_epochs=30

def collate(batch):
    speech_feats=[]
    text_feats=[]
    label=[]
    for sample in batch:
        speech_feats.append(sample['audio'])
        text_feats.append(sample['text'])
        label.append(sample['label'])

    return speech_feats,text_feats,label

dataset=generator()




dataloader_train = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0,collate_fn=collate)

model = build_model(
        model_size=128,
        max_length=300, num_heads=4,
        num_blocks=3, dropout=0.1,mode='audio',mask=True

    ).to(device)


# print(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for _ in range(1):
    model.train()
    loss_acc=[]
    for i,sampled in enumerate(tqdm(dataloader_train)):
        # text=torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled[0]]))

        text_feats=torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled[1]]))
        audio_feats=torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled[0]]))
        labels=torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled[2]]))
        text_feats=text_feats.long()
        audio_feats=audio_feats.float()
        labels=labels.long()

        output=model(text_feats,audio_feats)

        total_loss = criterion(output, labels.to(device))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(total_loss.item())


