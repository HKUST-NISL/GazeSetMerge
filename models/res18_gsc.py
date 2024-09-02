import torch
import torch.nn as nn
import torchvision
from models.gaze_convertor import gaze_adaptor
from models.resnet import resnet18

from models.attention import TransformerEncoding
from models.multispacial import MultiSpacial


class res18_gs_convertor(nn.Module):
    def __init__(self, with_la=True, pretrained=True):
        super(res18_gs_convertor, self).__init__()

        # feature extract
        self.face_backbone = resnet18(pretrained=pretrained)
        self.leye_backbone = resnet18(pretrained=pretrained)#, replace_stride_with_dilation=[True, True, True])
        self.reye_backbone = resnet18(pretrained=pretrained)#, replace_stride_with_dilation=[True, True, True])
        # self.reye_backbone = self.leye_backbone

        feat_size = self.face_backbone.output_channel

        # fc output
        self.fc_eye = nn.Sequential(
            nn.Linear(feat_size * 2, 128),
            nn.ReLU(True)
        )
        self.fc_face = nn.Sequential(
            nn.Linear(feat_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )

        self.with_la = with_la
        self.convertor = gaze_adaptor()

    def encode_input(self, data):
        face_data = data['image']
        leye_box = data['left_eye_box'].float()
        reye_box = data['right_eye_box'].float()
        data_type = data['type'].float()

        B = face_data.shape[0]
        batch_order = torch.arange(B, dtype=leye_box.dtype, device=leye_box.device).view(B, 1)

        leye_box_ = torch.cat([batch_order, leye_box], dim=1)
        reye_box_ = torch.cat([batch_order, reye_box], dim=1)

        leye_data = torchvision.ops.roi_align(face_data, leye_box_, 128, aligned=True)
        reye_data = torchvision.ops.roi_align(face_data, reye_box_, 128, aligned=True)

        encoded_data = {
            'face': face_data,
            'type': data_type.clone(),
            'left_eye': leye_data.clone(),
            'right_eye': reye_data.clone()
        }

        return encoded_data

    def forward(self, data):
        data = self.encode_input(data)

        face = data['face']
        left_eye = data['left_eye']
        right_eye = data['right_eye']
        data_type = data['type'].unsqueeze(-1)

        B = face.shape[0]
        x_leye = self.leye_backbone(left_eye).view(B, -1)
        x_reye = self.reye_backbone(right_eye).view(B, -1)
        x_face = self.face_backbone(face).view(B, -1)

        x_eye = torch.cat([x_leye, x_reye], dim=1)
        x_eye = self.fc_eye(x_eye)

        x_face = self.fc_face(x_face)

        x = torch.cat([x_eye, x_face], dim=1)
        x = self.fc_out(x)

        if self.with_la:
            x = self.convertor(x, data_type)

        return x