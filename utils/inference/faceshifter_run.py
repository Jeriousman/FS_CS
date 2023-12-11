'''

'''

import torch
import numpy as np


def faceshifter_batch(source_emb: torch.tensor, 
                      target: torch.tensor,
                      G: torch.nn.Module) -> np.ndarray:
    """
    Apply faceshifter model for batch of target images
    """
    
    bs = target.shape[0] ##batch_size
    assert target.ndim == 4, "target should have 4 dimentions -- B x C x H x W"
    
    if bs > 1:
        source_emb = torch.cat([source_emb]*bs)  ##source_emb를 batch_size만큼 복사를 해서 B C H W로 브로드캐스팅한다
    
    with torch.no_grad():
        Y_st, _ = G(target, source_emb)
        Y_st = (Y_st.permute(0, 2, 3, 1)*0.5 + 0.5)*255  ##torch tensor를 디노말라이징하고 255곱해줘서 이미지화 시켜줌
        Y_st = Y_st[:, :, :, [2,1,0]].type(torch.uint8)  ##RGB로 만들어 주는 것으로 보인다
        Y_st = Y_st.cpu().detach().numpy()    
    return Y_st