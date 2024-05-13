from models.PUSNet import pusnet
import torch

model = pusnet()

compare = torch.gt

a = torch.tensor([1])
b = torch.tensor([0])

cc = compare(a, b).float()

print(cc)