import torch
from torch import Tensor
from torch.nn import functional as F
import sys
sys.path.append('./')
from models.base import BaseModel
# from fpn_head import FPNHead
from models.upernet_head import UPerHead
from models.segformer_head import SegFormerHead
# from lawin_head import LawinHead


class CONVNEXTMODEL(BaseModel):
    def __init__(self, backbone: str = 'ConvNeXt-T', num_classes: int = 4): #Change number of classes and backbone netwrok accordingly
        super().__init__(backbone, num_classes)
        self.decode_head = UPerHead(self.backbone.channels, 128, num_classes) 
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        ms_feature = y
        inp = self.decode_head(y) #4x reduction in image size
        out = F.interpolate(inp, size=x.shape[2:], mode='bilinear', align_corners=False) #to original image shape
        return ms_feature, inp, out
##
class SEGFORMER(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 4) -> None: #Change the decoder type accordingly {PVT, MIT, ResT, and others}
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 128, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        ms_feature = y
        inp = self.decode_head(y)   #4x reduction in image size
        out = F.interpolate(inp, size=x.shape[2:], mode='bilinear', align_corners=False) #to original image shape #Uniformly resize to 224*224
        return ms_feature, inp, out


# if __name__ == '__main__':
#     net = CONVNEXTMODEL('ConvNeXt-T', 4)
#     net.init_pretrained("ProMSC/models/weights/convnext_tiny_1k_224_ema.pth")
#     model = net
#     bck = model.backbone.parameters()
#     print (bck)
    # x = torch.zeros(2, 3, 224, 224)
    # msf, inp, y = model(x)
    # print(model)
    # print(y.shape)
    # print(inp.shape)
    # for f in msf:
    #     print(f.shape)
        
    
# if __name__ == '__main__':
#     model = CNN_based('ConvNeXt-S', 3)
#     model.init_pretrained('ProMSC/models/weights/convnext_small_1k_224_ema.pth')
#     x = torch.randn(2, 3, 224, 224)
#     msf, inp, y = model(x)
#     # print(model)
#     print(inp.shape)
#     print(y.shape)
#     for f in msf:
#         print(f.shape)