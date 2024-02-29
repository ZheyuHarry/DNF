from mindspore import nn


# Depth-Wise Convolution
class DConv7(nn.Cell):
    def __init__(self , f_number , pad_mode='pad') -> None:
        super().__init__()
        self.dconv = nn.Conv2d(f_number , f_number , kernel_size=7,padding=3,group=f_number,pad_mode=pad_mode, has_bias=True)

    def construct(self, x):
        return self.dconv(x)


# Point-Wise Convolution in a MLP
class MLP(nn.Cell):
    def __init__(self , f_number , excitation_factor) -> None:
        super().__init__()
        super.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(f_number , f_number * excitation_factor , kernel_size=1 , pad_mode="valid" , has_bias=True)
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor , f_number , kernel_size=1 , pad_mode="valid" , has_bias=True)

    def construct(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x

# Ensemble the above layers into CID block
class CID(nn.Cell):
    def __init__(self , f_number , pad_mode) -> None:
        super().__init__()
        self.channel_independent = DConv7(f_number , pad_mode=pad_mode)
        self.channel_dependent = MLP(f_number , excitation_factor=2)

    def construct(self, x):
        return self.channel_dependent(self.channel_independent(x))