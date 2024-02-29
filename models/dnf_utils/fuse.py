import mindspore
from mindspore import nn , ops


class PDConvFuse(nn.Cell):
    def __init__(self , in_channels = None , f_number = None , feature_num = 2 , has_bias = True , **kwargs) -> None:
        super().__init__()
        if in_channels is None:
            assert f_number is not None
            in_channels = f_number
        self.feature_num = feature_num
        self.act = nn.GELU()
        self.pwconv = nn.Conv2d(feature_num * in_channels , in_channels , kernel_size=1 , stride=1 , pad_mode='valid',has_bias=has_bias)
        self.dwconv = nn.Conv2d(in_channels , in_channels , kernel_size=3 , stride=1,padding=1,has_bias=has_bias,group=in_channels,pad_mode='pad')

    def construct(self, *input_features):
        assert len(input_features) == self.feature_num
        return self.dwconv(self.act(self.pwconv(ops.concat(input_x=input_features , axis=1))))


class GFM(nn.Cell):
    def __init__(self , in_channels , feature_num=2, has_bias=True , pad_mode='pad',**kwargs) -> None:
        super().__init__()
        self.feature_num = feature_num

        hidden_features = in_channels * feature_num
        self.pwconv = nn.Conv2d(hidden_features , hidden_features * 2 , kernel_size=1 , stride=1 , pad_mode='valid' , has_bias=has_bias)
        self.dwconv = nn.Conv2d(hidden_features*2 , hidden_features*2 , kernel_size=3 , stride=1, padding=1 , pad_mode=pad_mode,has_bias=has_bias,group=hidden_features*2)
        self.project_out = nn.Conv2d(hidden_features , in_channels , kernel_size=1,has_bias=has_bias,pad_mode='valid')
        self.mlp = nn.Conv2d(in_channels , in_channels , kernel_size=1 , stride=1,pad_mode='valid',has_bias=has_bias)

    def construct(self, *input_features):
        assert len(input_features) == self.feature_num
        shortcut = input_features[0]
        x = ops.concat(input_x=input_features , axis=1)
        x = self.pwconv(x)
        x1 , x2 = ops.split(axis=1,output_num=2)(self.dwconv(x))
        x = nn.GELU()(x1) * x2
        x = self.project_out(x)
        return self.mlp(x + shortcut)