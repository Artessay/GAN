import argparse
import torch

print(torch.cuda.device_count())

opt = argparse.Namespace()
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    opt.cuda = True
else:
    opt.cuda = False
print(opt)

