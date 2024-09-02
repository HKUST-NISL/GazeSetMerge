import torch
import torch.nn as nn
import torchvision
from models.resnet import resnet18

from models.attention import TransformerEncoding
from models.multispacial import MultiSpacial


class gaze_res18_itracker(nn.Module):
    def __init__(self, pretrained=True):
        super(gaze_res18_itracker, self).__init__()

        # feature extract
        self.face_backbone = resnet18(pretrained=pretrained)
        self.leye_backbone = resnet18(pretrained=pretrained)#, replace_stride_with_dilation=[True, True, True])
        self.reye_backbone = resnet18(pretrained=pretrained)#, replace_stride_with_dilation=[True, True, True])

        feat_size = self.face_backbone.output_channel

        # multi-head attention
        self.trans = TransformerEncoding(feat_size)

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
            nn.Linear(128 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )

        self.msp_leye = MultiSpacial()
        self.msp_reye = MultiSpacial()
        self.trans_special = TransformerEncoding(448, out_size=8)

        self.norm_sp = nn.LayerNorm(feat_size*2)

        self.fc_eye_sp = nn.Sequential(
            nn.Linear(feat_size * 2, 128),
            nn.ReLU(True)
        )

        self.type_A = nn.Parameter(torch.zeros((2, 2)), requires_grad=True)
        self.type_b = nn.Parameter(torch.zeros((2, 2)), requires_grad=True)

    def encode_input(self, data):
        face_data = data['image']
        leye_box = data['left_eye_box'].float()
        reye_box = data['right_eye_box'].float()

        B = face_data.shape[0]
        batch_order = torch.arange(B, dtype=leye_box.dtype, device=leye_box.device).view(B, 1)

        leye_box_ = torch.cat([batch_order, leye_box], dim=1)
        reye_box_ = torch.cat([batch_order, reye_box], dim=1)

        leye_data = torchvision.ops.roi_align(face_data, leye_box_, 128, aligned=True)
        reye_data = torchvision.ops.roi_align(face_data, reye_box_, 128, aligned=True)

        encoded_data = {
            'face': face_data,
            'left_eye': leye_data.clone(),
            'right_eye': reye_data.clone()
        }

        return encoded_data

    def forward(self, data):
        data = self.encode_input(data)

        face = data['face']
        left_eye = data['left_eye']
        right_eye = data['right_eye']

        B = face.shape[0]
        x_leye, ep_leye = self.leye_backbone(left_eye)
        x_reye, ep_reye = self.reye_backbone(right_eye)
        x_face, ep_face = self.face_backbone(face)

        msp_leye = self.msp_leye(ep_leye)
        msp_reye = self.msp_reye(ep_reye)
        sb, sc, sh, sw = msp_leye.shape

        msp_leye = msp_leye.reshape(sb, sc, -1).permute(0, 2, 1)
        msp_reye = msp_reye.reshape(sb, sc, -1).permute(0, 2, 1)
        msp_seq = torch.cat([msp_leye, msp_reye], dim=1)
        msp_seq = self.trans_special(msp_seq)

        msp_leye = msp_seq[:, :sh*sw].reshape(sb, -1)
        msp_reye = msp_seq[:, sh*sw:].reshape(sb, -1)

        msp_eye = self.norm_sp(torch.cat([msp_leye, msp_reye], dim=1))
        msp_eye = self.fc_eye_sp(msp_eye)

        x_leye = x_leye.view(B, 1, -1)
        x_reye = x_reye.view(B, 1, -1)
        x_face = x_face.view(B, 1, -1)

        x_seq = torch.cat([x_leye, x_reye, x_face], dim=1)
        x_ffn = self.trans(x_seq)
        x_leye, x_reye, x_face = torch.unbind(x_ffn, dim=1)

        x_eye = torch.cat([x_leye, x_reye], dim=1)
        x_eye = self.fc_eye(x_eye)

        x_face = self.fc_face(x_face)

        x = torch.cat([x_eye, x_face, msp_eye], dim=1)
        # x = torch.cat([x_eye, x_face], dim=1)
        x = self.fc_out(x)

        return x