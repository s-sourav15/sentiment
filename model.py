import torch
import torch.nn as nn
import torch.functional as f
from layers import *
from multiHeadAttention import *
from position import *


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


class Transformer(nn.Module):
    def __init__(self, d_model, hidden_size, num_heads, num_blocks, activation=nn.ReLU(), dropout=0.1, mask=None):
        super().__init__()
        self.num_blocks = num_blocks
        self.transformer = [Encoder(d_model, hidden_size, num_heads, mask, dropout, activation) for _ in
                            range(num_blocks)]
        self.transformerS = nn.Sequential(
            *[Encoder(d_model, hidden_size, num_heads, mask, dropout, activation)
              for _ in range(num_blocks)]
        )

    def forward(self, k, q, v):
        for i in range(self.num_blocks):
            k = self.transformer[i](k, q, v)
        return k


class build_model(nn.Module):
    def __init__(self, max_length, embeddings=None, model_size=128, train_embedding=False, dropout=0.1, num_heads=4,
                 num_blocks=6, mask=False, mode="text"):
        super().__init__()
        self.max_length = max_length
        self.model_size = model_size
        self.train_embedding = train_embedding
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        # self.embeddings=embeddings
        self.conv = nn.Conv2d(30, 3, 1, 3, padding=0)
        self.mode = mode
        self.maxpooling = nn.MaxPool2d(2, 2)
        if embeddings is not None:
            self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=not train_embedding)
            self.emb_ff = nn.Linear(embeddings.size(1), self.model_size)
        else:
            self.embeddings = nn.Embedding(max_length, model_size)
            self.emb_ff = nn.Linear(model_size, model_size)
        self.masking = mask
        # self.mask=[]
        self.posit = Position(model_size, max_length)
        self.output = nn.Linear(self.model_size, 9)

        self.pos = nn.Linear(max_length, self.model_size)

        self.transformer = Transformer(self.model_size, self.model_size, self.num_heads, self.num_blocks,
                                       dropout=dropout)  # ,mask=self.mask)

    def forward(self, xText, xAudio):
        batch_size = xText.size()[0]
        print(xText.size())

        xText = xText.view(-1)
        xText = self.embeddings(xText)
        xText = self.emb_ff(xText)
        posText = self.posit(xText, 0)
        xText *= math.sqrt(self.model_size)
        posText = posText[-1, :, :]
        xText += posText
        xText = xText.view(batch_size, xText.size()[0] // batch_size, xText.size()[1])
        xAudio = nn.Linear(600, self.model_size)(xAudio)  # todo hardcoded
        posAudio = self.posit(xAudio, 1)
        xAudio *= math.sqrt(self.model_size)
        xAudio += posAudio

        # if self.masking == True:
        #     self.mask = buffered_future_mask(x)  ## todo fix the self.mask part declare it in the class and feed it here
        # print(self.mask)
        x = self.transformer(xText, xAudio, xAudio)
        x = x.mean(dim=1)
        return self.output(x)
