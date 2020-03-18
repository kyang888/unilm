import torch
torch.autograd.profiler.profile()
import tensorboardX
writer = tensorboardX.SummaryWriter()