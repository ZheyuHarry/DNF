from mindspore import nn


class ResidualSwitchBlock(nn.Cell):
    def __init__(self , block) -> None:
        super().__init__()
        self.block = block

    def construct(self, x , residual_switch):
        return self.block(x) + residual_switch * x