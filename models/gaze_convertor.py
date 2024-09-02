import torch
import torch.nn as nn
from models.resnet import resnet18

class gaze_adaptor(nn.Module):
    def __init__(self):
        super(gaze_adaptor, self).__init__()

        # self.face_backbone = resnet18(pretrained=True)
        # feat_size = self.face_backbone.output_channel
        # self.fc_face = nn.Sequential(
        #     nn.Linear(feat_size, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 128),
        #     nn.ReLU(True)
        # )

        self.fc_delta = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 2),
        )

    def forward(self, x_feat, data_type):

        d_gaze = self.fc_delta(x_feat) * data_type
        # print(d_gaze)
        
        return d_gaze
