import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler

import model.functional as LF
import model.normalization as normalization




class MAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.adaptive_max_pool2d(x, (1, 1))

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SPoC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))

    def __repr__(self):
        return self.__class__.__name__ + "()"


class GeM(nn.Module):
    def __init__(self, p: float=3.0, eps: float=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(
        1.0 / self.p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class RMAC(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        super().__init__()
        self.L = L
        self.eps = eps
        self.steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    def forward(self, x):
        ovr = 0.4  # desired overlap of neighboring regions
        
        W = x.size(3)
        H = x.size(2)
        w = min(W, H)
        # w2 = math.floor(w/2.0 - 1)
        tmp = (max(H, W) - w) / (self.steps - 1)
        (tmp, idx) = torch.min(
            torch.abs(((w**2 - w * tmp) / w**2) - ovr), 0
        )  # steps(idx) regions for long dimension
        # region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1
        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + self.eps).expand_as(v)
        for l in range(1, self.L + 1):
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)
            if l + Wd == 1:
                b = 0.
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = (
                torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2
            )  # center coordinates
            if l + Hd == 1:
                b = 0.
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = (
                torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2
            )  # center coordinates
            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                    R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + self.eps).expand_as(vt)
                    v += vt
        return v
    
    def __repr__(self):
        return self.__class__.__name__ + "(" + "L=" + "{}".format(self.L) + ")"


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class RRM(nn.Module):
    """Residual Retrieval Module as described in the paper
    `Leveraging EfficientNet and Contrastive Learning for AccurateGlobal-scale
    Location Estimation <https://arxiv.org/pdf/2105.07645.pdf>`
    """

    def __init__(self, dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.fc1 = nn.Linear(in_features=dim, out_features=dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=dim, out_features=dim)
        self.ln2 = nn.LayerNorm(normalized_shape=dim)
        self.l2 = normalization.L2Norm()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.ln1(x)
        identity = x
        out = self.fc2(self.relu(self.fc1(x)))
        out += identity
        out = self.l2(self.ln2(out))
        return out

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(
        self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False
    ):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1))
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        self.conv.weight = nn.Parameter(
            torch.from_numpy(self.alpha * centroids_assign)
            .unsqueeze(2)
            .unsqueeze(3)
        )


    def initialize_netvlad_layer(self, args, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(
            np.random.choice(len(cluster_ds), images_num, replace=False)
        )
        random_dl = DataLoader(
            dataset=cluster_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            sampler=random_sampler,
        )
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(
                shape=(descriptors_num, args.features_dim), dtype=np.float32
            )
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(args.device)
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(
                    norm_outputs.shape[0], args.features_dim, -1
                ).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(
                        image_descriptors.shape[1], descs_num_per_image, replace=False
                    )
                    startix = batchix + ix * descs_num_per_image
                    descriptors[
                        startix : startix + descs_num_per_image, :
                    ] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(
            args.features_dim, self.clusters_num, niter=100, verbose=False
        )
        kmeans.train(descriptors)
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)

        if args.fc_output_dim is not None:
            args.features_dim = args.fc_output_dim

    def set_output_dim(self, args):
        if args.fc_output_dim is not None:
            args.features_dim = args.fc_output_dim

  
    def forward(self, x):
        N, D, H, W = x.shape[:]
        x = F.normalize(x, p=2.0, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # Compute residuals for all clusters simultaneously
        residual = x_flatten.unsqueeze(1) - self.centroids.unsqueeze(0).unsqueeze(-1)
        residual *= soft_assign.unsqueeze(2)
        
        vlad = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2.0, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2.0, dim=1)  # L2 normalize
        
        return vlad

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class OrigNetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(
        self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False
    ):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2)
            )
        else:
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * centroids_assign)
                .unsqueeze(2)
                .unsqueeze(3)
            )
        self.conv.bias = None

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2.0, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros(
            [N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device
        )
        for D in range(
            self.clusters_num
        ):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[
                D : D + 1, :
            ].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D : D + 1, :].unsqueeze(2)
            vlad[:, D : D + 1, :] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2.0, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2.0, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(
            np.random.choice(len(cluster_ds), images_num, replace=False)
        )
        random_dl = DataLoader(
            dataset=cluster_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            sampler=random_sampler,
        )
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(
                shape=(descriptors_num, args.features_dim), dtype=np.float32
            )
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(args.device)
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(
                    norm_outputs.shape[0], args.features_dim, -1
                ).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(
                        image_descriptors.shape[1], descs_num_per_image, replace=False
                    )
                    startix = batchix + ix * descs_num_per_image
                    descriptors[
                        startix : startix + descs_num_per_image, :
                    ] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(
            args.features_dim, self.clusters_num, niter=100, verbose=False
        )
        kmeans.train(descriptors)
        logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)

        if args.fc_output_dim is not None:
            args.features_dim = args.fc_output_dim



class CRNModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Downsample pooling
        self.downsample_pool = nn.AvgPool2d(
            kernel_size=3, stride=(2, 2), padding=0, ceil_mode=True
        )

        # Multiscale Context Filters
        self.filter_3_3 = nn.Conv2d(
            in_channels=dim, out_channels=32, kernel_size=(3, 3), padding=1
        )
        self.filter_5_5 = nn.Conv2d(
            in_channels=dim, out_channels=32, kernel_size=(5, 5), padding=2
        )
        self.filter_7_7 = nn.Conv2d(
            in_channels=dim, out_channels=20, kernel_size=(7, 7), padding=3
        )

        # Accumulation weight
        self.acc_w = nn.Conv2d(in_channels=84, out_channels=1, kernel_size=(1, 1))
        # Upsampling
        self.upsample = F.interpolate

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize Context Filters
        torch.nn.init.xavier_normal_(self.filter_3_3.weight)
        torch.nn.init.constant_(self.filter_3_3.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_5_5.weight)
        torch.nn.init.constant_(self.filter_5_5.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_7_7.weight)
        torch.nn.init.constant_(self.filter_7_7.bias, 0.0)

        torch.nn.init.constant_(self.acc_w.weight, 1.0)
        torch.nn.init.constant_(self.acc_w.bias, 0.0)
        self.acc_w.weight.requires_grad = False
        self.acc_w.bias.requires_grad = False

    def forward(self, x):
        # Contextual Reweighting Network
        x_crn = self.downsample_pool(x)

        # Compute multiscale context filters g_n
        g_3 = self.filter_3_3(x_crn)
        g_5 = self.filter_5_5(x_crn)
        g_7 = self.filter_7_7(x_crn)
        g = torch.cat((g_3, g_5, g_7), dim=1)
        g = F.relu(g)

        w = F.relu(self.acc_w(g))  # Accumulation weight
        mask = self.upsample(w, scale_factor=2, mode="bilinear")  # Reweighting Mask

        return mask


class CRN(NetVLAD):
    def __init__(self, clusters_num=64, dim=128, normalize_input=True):
        super().__init__(clusters_num, dim, normalize_input)
        self.crn = CRNModule(dim)

    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim

        mask = self.crn(x)

        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # Weight soft_assign using CRN's mask
        soft_assign = soft_assign * mask.view(N, 1, H * W)

        vlad = torch.zeros(
            [N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device
        )
        for D in range(
            self.clusters_num
        ):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[
                D : D + 1, :
            ].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D : D + 1, :].unsqueeze(2)
            vlad[:, D : D + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad
