import os
import sys
import cv2
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
from copy import deepcopy
from torchvision.transforms import ToTensor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from alike import ALike, configs

dataset_root = 'hseq/hpatches-sequences-release'
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
methods = ['alike-n', 'alike-l', 'alike-n-ms', 'alike-l-ms']


class HPatchesDataset(data.Dataset):
    def __init__(self, root: str = dataset_root, alteration: str = 'all'):
        """
        Args:
            root: dataset root path
            alteration: # 'all', 'i' for illumination or 'v' for viewpoint
        """
        assert (Path(root).exists()), f"Dataset root path {root} dose not exist!"
        self.root = root

        # get all image file name
        self.image0_list = []
        self.image1_list = []
        self.homographies = []
        folders = [x for x in Path(self.root).iterdir() if x.is_dir()]
        self.seqs = []
        for folder in folders:
            if alteration == 'i' and folder.stem[0] != 'i':
                continue
            if alteration == 'v' and folder.stem[0] != 'v':
                continue

            self.seqs.append(folder)

        self.len = len(self.seqs)
        assert (self.len > 0), f'Can not find PatchDataset in path {self.root}'

    def __getitem__(self, item):
        folder = self.seqs[item]

        imgs = []
        homos = []
        for i in range(1, 7):
            img = cv2.imread(str(folder / f'{i}.ppm'), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HxWxC
            imgs.append(img)

            if i != 1:
                homo = np.loadtxt(str(folder / f'H_1_{i}')).astype('float32')
                homos.append(homo)

        return imgs, homos, folder.stem

    def __len__(self):
        return self.len

    def name(self):
        return self.__class__


def extract_multiscale(model, img, scale_f=2 ** 0.5,
                       min_scale=1., max_scale=1.,
                       min_size=0., max_size=99999.,
                       image_size_max=99999,
                       n_k=0, sort=False):
    H_, W_, three = img.shape
    assert three == 3, "input image shape should be [HxWx3]"

    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # ==================== image size constraint
    image = deepcopy(img)
    max_hw = max(H_, W_)
    if max_hw > image_size_max:
        ratio = float(image_size_max / max_hw)
        image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

    # ==================== convert image to tensor
    H, W, three = image.shape
    image = ToTensor()(image).unsqueeze(0)
    image = image.to(device)

    s = 1.0  # current scale factor
    keypoints, descriptors, scores, scores_maps, descriptor_maps = [], [], [], [], []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = image.shape[2:]

            # extract descriptors
            with torch.no_grad():
                descriptor_map, scores_map = model.extract_dense_map(image)
                keypoints_, descriptors_, scores_, _ = model.dkd(scores_map, descriptor_map)

            keypoints.append(keypoints_[0])
            descriptors.append(descriptors_[0])
            scores.append(scores_[0])

        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        image = torch.nn.functional.interpolate(image, (nh, nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    keypoints = torch.cat(keypoints)
    descriptors = torch.cat(descriptors)
    scores = torch.cat(scores)
    keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W_ - 1, H_ - 1]])

    if sort or 0 < n_k < len(keypoints):
        indices = torch.argsort(scores, descending=True)
        keypoints = keypoints[indices]
        descriptors = descriptors[indices]
        scores = scores[indices]

    if 0 < n_k < len(keypoints):
        keypoints = keypoints[0:n_k]
        descriptors = descriptors[0:n_k]
        scores = scores[0:n_k]

    return {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}


def extract_method(m):
    hpatches = HPatchesDataset(root=dataset_root, alteration='all')
    model = m[:7]
    min_scale = 0.3 if m[8:] == 'ms' else 1.0

    model = ALike(**configs[model], device=device, top_k=0, scores_th=0.2, n_limit=5000)

    progbar = tqdm(hpatches, desc='Extracting for {}'.format(m))
    for imgs, homos, seq_name in progbar:
        for i in range(1, 7):
            img = imgs[i - 1]
            pred = extract_multiscale(model, img, min_scale=min_scale, max_scale=1, sort=False, n_k=5000)
            kpts, descs, scores = pred['keypoints'], pred['descriptors'], pred['scores']

            with open(os.path.join(dataset_root, seq_name, f'{i}.ppm.{m}'), 'wb') as f:
                np.savez(f, keypoints=kpts.cpu().numpy(),
                         scores=scores.cpu().numpy(),
                         descriptors=descs.cpu().numpy())


if __name__ == '__main__':
    for method in methods:
        extract_method(method)
