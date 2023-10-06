import torch
import numpy as np
from torchvision.models import resnet50, resnet34, resnet18
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.cuda.amp import autocast
from torch import nn
import logging
import torch_tensorrt
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import faiss


import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'






    
class MyNetVLAD(nn.Module):
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

import parser
args = parser.parse_arguments()

backbone = torchvision.models.resnet18(pretrained=True)
layers = list(backbone.children())[:-3]
model = torch.nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Size after three convolutions and three max pooling layers: 64x60x80
        self.fc1 = nn.Linear(64 * 60 * 80, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 features as output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 60 * 80)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    

mymodel = MyNetVLAD()
image = torch.randn(10, 128, 7, 7)

out = mymodel(image)
print("====================================", out.shape)

from util import quantize_model


qmodel = quantize_model(mymodel, precision="fp16_comp", args=args)