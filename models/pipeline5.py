import torch.nn as nn
import torch.nn.functional as F
from models.facenet2 import InceptionResnetV1
import torch

class Pipeline(nn.Module):
    """
    DLN model without high-order here. Pretrained weights can be found in checkpoints directory.
    """
    def __init__(self, out_dim=512):
        super(Pipeline,self).__init__()
        self.faceNet = InceptionResnetV1(pretrained="vggface2").eval()
        self.R_net = InceptionResnetV1(pretrained="vggface2")
        self.BN1 = nn.BatchNorm1d(512)
        self.BN2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512,16,bias=False)
        self.out_dim = out_dim

        if out_dim == 1:
            self.last_D = nn.Linear(512,1,bias=True)

    def forward(self, x , c=1):
        """
        Calculate expression embeddings or logits given a batch of input image tensors.
        :param x: Batch of image tensors representing faces.
        :return: Batch of embedding vectors or multinomial logits.
        """
        with torch.no_grad():
            id_feature_ = self.faceNet(x)
        id_feature = torch.sigmoid(id_feature_)
        x = self.R_net(x)
        x = torch.sigmoid(x)
        x = x-id_feature
        emb_16 = self.linear2(x)
        emb_16 = F.normalize(emb_16, dim=1)

        if self.out_dim == 1:
            x = self.last_D(x)
        if self.out_dim == 512 + 512:
            x = torch.cat([id_feature, x], dim=1)
        return x, emb_16

    def forward2(self, x):
        with torch.no_grad():
            id_feature = self.faceNet(x)
        id_feature = torch.sigmoid(id_feature)
        x = self.R_net(x)
        x = torch.sigmoid(x)
        x = x - id_feature
        global_feature = x
        x = self.linear2(x)
        x = F.normalize(x, dim=1)
        return global_feature, x


if __name__ == '__main__':
    net = Pipeline().cuda()
    x = torch.rand([16,3,224,224]).cuda()
    res= net(x)
    print(res.shape)
    print(sum(param.numel() for param in net.parameters()))