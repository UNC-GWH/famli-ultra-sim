import torch
from nets import cut



model_fn = "/mnt/GWH/Groups/FAMLI/Shared/C1_ML_Analysis/train_output/Cut/allvslast/allvsleltek/v0.1/epoch=6-val_loss=6.62.ckpt"

model = cut.CutG.load_from_checkpoint(model_fn).eval()
x = torch.rand(1, 1, 256, 256)
model.to_torchscript(file_path=model_fn.replace('.ckpt', '_v0.1.pt'), method="trace", example_inputs=x.cuda())