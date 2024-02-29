import mindspore
from mindspore import nn , ops
import numpy as np

class LayerNorm(nn.Cell):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self , normalized_shape , eps=1e-6,data_format="channels_last"):
        super().__init__()
        self.weight = mindspore.Parameter(mindspore.Tensor(np.ones(normalized_shape)))
        self.bias = mindspore.Parameter(mindspore.Tensor(np.zeros(normalized_shape)))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def construct(self, x):
        if self.data_format == "channels_last":
            return ops.LayerNorm(epsilon=self.eps)(x , self.weight , self.bias)[0]  # The return value is a tuple
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / ops.Sqrt()(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x