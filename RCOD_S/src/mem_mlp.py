import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import default_init_weights


class MetricEstimationModule(nn.Module):
    """
    Metric Estimation Module (MEM) for adaptive timestep selection in RCOD_S.
    """
    def __init__(self, pretrained_path=None, args=None, use_clip_score=False):
        super().__init__()
        self.args = args

        if use_clip_score:
            self.clip_feature_dim = 256 + 768
        else:
            self.clip_feature_dim = 256

        self.clip_projection = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 1),
            nn.ReLU(True),
        )

        default_init_weights([self.clip_projection], 1e-5)
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            if "state_dict_clip_proj" in sd:
                _clip_proj = self.clip_projection.state_dict()
                for k in sd["state_dict_clip_proj"]:
                    _clip_proj[k] = sd["state_dict_clip_proj"][k]
                self.clip_projection.load_state_dict(_clip_proj)

    def set_eval(self):
        self.clip_projection.eval()
        self.clip_projection.requires_grad_(False)

    def set_train(self):
        self.clip_projection.requires_grad_(True)
    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        x1_ = x1.reshape(x1.shape[0], -1)
        x2_ = x2.reshape(x2.shape[0], -1)
        output = F.cosine_similarity(x1_, x2_, dim=1)
       
        return output
 
    def div_step(self, cs_gt_bc):
        step_list = []
        if self.args.n_div == 3:
            for cs_gt in cs_gt_bc:
                if cs_gt<0.5:
                    cs_time_step = 749
                elif cs_gt<0.7 and cs_gt>=0.5:
                    cs_time_step = 499
                elif cs_gt>=0.7:
                    cs_time_step=249
                else:
                    cs_time_step = 499
                step_list.append(cs_time_step)
        elif self.args.n_div == 4:
            for cs_gt in cs_gt_bc:
                if cs_gt < 0.3:
                    cs_time_step = 999
                elif cs_gt < 0.5 and cs_gt >= 0.3:
                    cs_time_step = 749
                elif cs_gt < 0.7 and cs_gt >= 0.5:
                    cs_time_step = 499
                elif cs_gt >= 0.7:
                    cs_time_step = 249
                else:
                    cs_time_step = 499
                step_list.append(cs_time_step)

        return step_list

    def forward(self, deg_score, clip_score=None):
        deg_score = deg_score.view(deg_score.shape[0], -1)
        if clip_score is not None:
            clip_score = clip_score.view(clip_score.shape[0], -1)
            deg_score = torch.cat([deg_score, clip_score], dim=1)
        cs_score_pred = self.clip_projection(deg_score)
        return cs_score_pred

    def save_model(self, outf):
        sd = {}
        sd["state_dict_clip_proj"] = {k: v for k, v in self.clip_projection.state_dict().items()}
        torch.save(sd, outf)


# Backward compatibility aliases
S3Diff = MetricEstimationModule
MEM_MLP = MetricEstimationModule

