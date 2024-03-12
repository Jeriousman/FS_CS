'''

'''

import torch
import numpy as np


def faceshifter_batch(id_embedding: torch.tensor, 
                      target: torch.tensor,
                      source: torch.tensor,
                      G: torch.nn.Module) -> np.ndarray:
    """
    Apply faceshifter model for batch of target images
    """
    
    bs = target.shape[0] ##batch_size
    assert target.ndim == 4, "target should have 4 dimentions -- B x C x H x W"
    
    if bs > 1:
        id_embedding = torch.cat([id_embedding]*bs)  ##source_emb를 batch_size만큼 복사를 해서 B C H W로 브로드캐스팅한다
        source = torch.cat([source]*bs)
    
    with torch.no_grad():
        Y_st, _, _ = G(target, source, id_embedding) 
        Y_st = (Y_st.permute(0, 2, 3, 1)*0.5 + 0.5)*255  ##torch tensor를 디노말라이징하고 255곱해줘서 이미지화 시켜줌
        Y_st = Y_st[:, :, :, [2,1,0]].type(torch.uint8)  ##RGB로 만들어 주는 것으로 보인다
        Y_st = Y_st.cpu().detach().numpy()    
    return Y_st