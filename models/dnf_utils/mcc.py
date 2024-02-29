import mindspore
from mindspore import nn , ops
import numpy as np

from ..utils import LayerNorm

class MCC(nn.Cell):
    def __init__(self , f_number , num_heads , pad_mode , has_bias=False) -> None:
        super().__init__()
        self.norm = LayerNorm(f_number , eps=1e-6,data_format="channel_first")
        self.num_heads = num_heads
        self.temperature = mindspore.Parameter(mindspore.Tensor(np.ones((num_heads, 1, 1))))
        self.pwconv = nn.Conv2d(f_number , f_number*3, kernel_size=1 , has_bias=has_bias,pad_mode='valid')
        self.dwconv = nn.Conv2d(f_number*3,f_number*3, kernel_size=3 , stride=1, padding=1, pad_mode='pad' , group=f_number*3)
        self.project_out = nn.Conv2d(f_number, f_number , kernel_size=1 , has_bias=has_bias , pad_mode='valid')
        self.feedforward = nn.SequentialCell(
            nn.Conv2d(f_number, f_number, kernel_size=1 , stride=1 , padding=0 , has_bias=has_bias , pad_mode='valid'),
            nn.GELU(),
            nn.Conv2d(f_number , f_number , kernel_size=3 , stride=1 , padding=1 , has_bias=has_bias , group=f_number , pad_mode='pad'),
            nn.GELU()
        )

    def construct(self, x):
        attn = self.norm(x)
        _ , _ , h, w = attn.shape

        qkv = self.dwconv(self.pwconv(attn))
        q , k , v = mindspore.ops.split(qkv , axis=1 ,output_num=3)

        head = self.num_heads
        q = ops.reshape(q , (q.shape[0] , head , q.shape[1] // head , q.shape[2]*q.shape[3]))
        k = ops.reshape(k, (k.shape[0], head, k.shape[1] // head, k.shape[2] * k.shape[3]))
        v = ops.reshape(v, (v.shape[0], head, v.shape[1] // head, v.shape[2] * v.shape[3]))

        q = ops.L2Normalize(q , axis=-1)
        k = ops.L2Normalize(k, axis=-1)

        attn = (ops.mul(q , k.tranpose(-2 , -1))) * self.temperature
        attn = ops.Softmax(axis=-1)(attn)

        out = ops.mul(attn , v)

        out = ops.reshape(out , (out.shape[0] , out.shape[1]*out.shape[2] , h , out.shape[3] // h ))

        out = self.project_out(out)
        out = self.feedforward(out + x)
        return out