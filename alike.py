import os
from typing import Tuple
import torch
import math

from .alnet import ALNet
from .soft_detect import DKD


class ALike(ALNet):
    def __init__(
            self,
            radius: int = 2,
            top_k: int = -1, scores_th: float = 0.3,
            n_limit: int = 500,
            device: str = 'cuda',
        ):

        super().__init__(32, 64, 128, 128, 128, False)

        self.radius = radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th
        self.dkd = DKD(radius=self.radius, top_k=self.top_k,
                       scores_th=self.scores_th, n_limit=self.n_limit)
        self.device = device

        model_path = os.path.join(os.path.split(__file__)[0], 'alike-l.pth')

        if model_path != '':
            state_dict = torch.load(model_path, self.device)
            self.load_state_dict(state_dict)
            self.eval()
            
    def extract_dense_map(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
        if h_ != h:
            h_padding = torch.zeros(b, c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        scores_map, descriptor_map = self.super_forward(image)

        # ====================================================
        if h_ != h or w_ != w:
            descriptor_map = descriptor_map[:, :, :h, :w]
            scores_map = scores_map[:, :, :h, :w]  # Bx1xHxW
        # ====================================================

        # BxCxHxW
        descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2.0, dim=1)

        return descriptor_map, scores_map
    
    def super_forward(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32

        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample8(x3)  # B x dim//4 x H x W
        x4_up = self.upsample32(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        x = self.convhead2(x1234)  # B x dim+1 x H x W

        descriptor_map = x[:, :-1, :, :]
        scores_map = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)

        return scores_map, descriptor_map

    def forward(self, img: torch.Tensor, sort:bool=True, sub_pixel:bool=True, n_keypoints:int=0) -> torch.Tensor:
        """
        :param img: torch.Tensor Bx3xHxW, RGB
        :param sort: sort keypoints by scores
        :param sub_pixel: whether to use sub-pixel accuracy
        """
        B, three, H, W = img.shape
        assert three == 3, "input image shape should be [HxWx3]"

        # ==================== extract keypoints
        with torch.no_grad():
            descriptor_map, scores_map = self.extract_dense_map(img)
            keypoints, _, scores, _ = self.dkd(scores_map, descriptor_map,
                                                         sub_pixel=sub_pixel)

        result_keypoints = []

        for b in range(B):
            score = scores[b]
            keypoint = keypoints[b]

            if sort:
                indices = torch.argsort(score, descending=True)
                keypoint = keypoint[indices]

            if n_keypoints > 0:
                if n_keypoints < len(score):
                    keypoint = keypoint[:n_keypoints]
                else:
                    while len(keypoint) < n_keypoints:
                        # pad keypoints
                        diff = min(n_keypoints - len(keypoint), len(keypoint))
                        keypoint = torch.cat([keypoint, keypoint[:diff]])

            result_keypoints.append(keypoint)

        keypoints = torch.stack(result_keypoints, dim=0)

        return keypoints