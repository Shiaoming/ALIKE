import cv2
import os
from tqdm import tqdm
import torch
import numpy as np
from extract import extract_method

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

methods = ['d2', 'lfnet', 'superpoint', 'r2d2', 'aslfeat', 'disk',
           'alike-n', 'alike-l', 'alike-n-ms', 'alike-l-ms']
names = ['D2-Net(MS)', 'LF-Net(MS)', 'SuperPoint', 'R2D2(MS)', 'ASLFeat(MS)', 'DISK',
         'ALike-N', 'ALike-L', 'ALike-N(MS)', 'ALike-L(MS)']

top_k = None
n_i = 52
n_v = 56
cache_dir = 'hseq/cache'
dataset_path = 'hseq/hpatches-sequences-release'


def generate_read_function(method, extension='ppm'):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(dataset_path, seq_name, '%d.%s.%s' % (im_idx, extension, method)))
        if top_k is None:
            return aux['keypoints'], aux['descriptors']
        else:
            assert ('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k:]
            return aux['keypoints'][ids, :], aux['descriptors'][ids, :]

    return read_function


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def homo_trans(coord, H):
    kpt_num = coord.shape[0]
    homo_coord = np.concatenate((coord, np.ones((kpt_num, 1))), axis=-1)
    proj_coord = np.matmul(H, homo_coord.T).T
    proj_coord = proj_coord / proj_coord[:, 2][..., None]
    proj_coord = proj_coord[:, 0:2]
    return proj_coord


def benchmark_features(read_feats):
    lim = [1, 5]
    rng = np.arange(lim[0], lim[1] + 1)

    seq_names = sorted(os.listdir(dataset_path))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    i_err_homo = {thr: 0 for thr in rng}
    v_err_homo = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        # =========== compute homography
        ref_img = cv2.imread(os.path.join(dataset_path, seq_name, '1.ppm'))
        ref_img_shape = ref_img.shape

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device)
            )

            homography = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))

            pos_a = keypoints_a[matches[:, 0], : 2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2:]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

            # =========== compute homography
            gt_homo = homography
            pred_homo, _ = cv2.findHomography(keypoints_a[matches[:, 0], : 2], keypoints_b[matches[:, 1], : 2],
                                              cv2.RANSAC)
            if pred_homo is None:
                homo_dist = np.array([float("inf")])
            else:
                corners = np.array([[0, 0],
                                    [ref_img_shape[1] - 1, 0],
                                    [0, ref_img_shape[0] - 1],
                                    [ref_img_shape[1] - 1, ref_img_shape[0] - 1]])
                real_warped_corners = homo_trans(corners, gt_homo)
                warped_corners = homo_trans(corners, pred_homo)
                homo_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err_homo[thr] += np.mean(homo_dist <= thr)
                else:
                    v_err_homo[thr] += np.mean(homo_dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, i_err_homo, v_err_homo, [seq_type, n_feats, n_matches]


if __name__ == '__main__':
    errors = {}
    for method in methods:
        output_file = os.path.join(cache_dir, method + '.npy')
        read_function = generate_read_function(method)
        if os.path.exists(output_file):
            errors[method] = np.load(output_file, allow_pickle=True)
        else:
            extract_method(method)
            errors[method] = benchmark_features(read_function)
            np.save(output_file, errors[method])

    for name, method in zip(names, methods):
        i_err, v_err, i_err_hom, v_err_hom, _ = errors[method]

        print(f"====={name}=====")
        print(f"MMA@1 MMA@2 MMA@3 MHA@1 MHA@2 MHA@3: ", end='')
        for thr in range(1, 4):
            err = (i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5)
            print(f"{err * 100:.2f}%", end=' ')
        for thr in range(1, 4):
            err_hom = (i_err_hom[thr] + v_err_hom[thr]) / ((n_i + n_v) * 5)
            print(f"{err_hom * 100:.2f}%", end=' ')
        print('')
