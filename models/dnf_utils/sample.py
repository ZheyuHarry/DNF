from mindspore import nn

class SimpleDownsample(nn.Cell):
    def __init__(self , dim , * , pad_mode='pad'):
        super().__init__()
        self.body = nn.Conv2d(dim , dim*2 , kernel_size=2 , stride=2,padding=0,has_bias=False,pad_mode='valid')

    def construct(self, x):
        return self.body(x)

class SimpleUpsample(nn.Cell):
    def __init__(self , dim , * , pad_mode='pad'):
        super().__init__()
        self.body = nn.Conv2dTranspose(dim , dim//2 , kernel_size=2 , stride=2 , padding=0 , pad_mode='valid' , has_bias=False)

    def construct(self, x):
        return self.body(x)