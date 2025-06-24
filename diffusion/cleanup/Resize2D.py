import torch.nn.functional as F

class Resize2D:
    def __init__(self, target_size, mode='nearest'):
        """
        Resize a 2D tensor [B, C, H, W] to the target size.

        Args:
            target_size (tuple): (H, W), use -1 to keep that dimension unchanged
        """
        assert len(target_size) == 2, "target_size must be a tuple of (H, W)"
        self.target_size = target_size
        self.mode = mode

    def __call__(self, x):
        """
        Resize the input tensor.

        Args:
            x (torch.Tensor): [B, C, H, W]

        Returns:
            torch.Tensor: resized tensor [B, C, H', W']
        """
        assert x.ndim == 4, "Input tensor must have shape [B, C, H, W]"
        B, C, H, W = x.shape

        out_H = self.target_size[0] if self.target_size[0] != -1 else H
        out_W = self.target_size[1] if self.target_size[1] != -1 else W

        resized = F.interpolate(x, size=(out_H, out_W), mode=self.mode)
        return resized