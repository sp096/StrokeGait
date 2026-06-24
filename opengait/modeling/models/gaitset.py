import torch
import copy
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SeparateFCs, SeparateBNNecks, BasicConv2d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper


class GaitSet(BaseModel):
    """
        GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition
        Arxiv:  https://arxiv.org/abs/1811.06186
        Github: https://github.com/AbnerHqC/GaitSet
    """

    def build_network(self, model_cfg):
        in_c = model_cfg['in_channels']
        self.set_block1 = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2 = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3 = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))

        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_pooling = PackSequenceWrapper(torch.max)

        self.Head = SeparateFCs(**model_cfg['SeparateFCs'])
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        # 分类头（可选，用于分类任务）
        if 'SeparateBNNecks' in model_cfg:
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.class_num = model_cfg['SeparateBNNecks']['class_num']
        else:
            self.BNNecks = None
            self.class_num = None

        # 固定标签映射（从 trainer_cfg 读取）
        self.label_mapping = self.cfgs.get('trainer_cfg', {}).get('label_mapping', None)

    def forward(self, inputs):
        ipts, labs, labels, _, seqL = inputs
        """
        inputs 结构:
        - labs: 身份ID索引（用于triplet loss）
        - labels: 类型/类别名（用于分类，格式为 类别名/身份ID/视角 时为类别名）
        """
        sils = ipts[0]  # [n, s, h, w]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        outs = self.set_block1(sils)
        gl = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        feature2 = self.HPP(gl)  # [n, c, p]
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs = self.Head(feature)

        n, _, s, h, w = sils.size()

        # 构建返回值
        if self.BNNecks is not None:
            # 分类模式：labels 就是类别名（如 003, 004, 005）
            # 只在训练时需要构建 label_ids，测试时直接用 labs 作为标签
            import numpy as np
            if self.training:
                # 使用固定标签映射（优先）或自动映射
                if hasattr(self, 'label_mapping') and self.label_mapping is not None:
                    label_map = {}
                    for k, v in self.label_mapping.items():
                        key_str = str(k)
                        label_map[key_str] = int(v)
                        try:
                            label_map[str(int(key_str))] = int(v)
                        except (TypeError, ValueError):
                            pass
                else:
                    unique_labels = sorted(list(set(labels)))
                    label_map = {str(label): idx for idx, label in enumerate(unique_labels)}
                
                # 灵活的标签查询
                def get_label_id(status):
                    candidates = [str(status)]
                    try:
                        candidates.append(str(int(str(status))))
                    except (TypeError, ValueError):
                        pass
                    for candidate in dict.fromkeys(candidates):
                        if candidate in label_map:
                            return label_map[candidate]
                    raise KeyError(f"Label '{status}' not found in label_map: {label_map}")

                label_ids = np.array([get_label_id(status) for status in labels])
                label_ids = torch.from_numpy(label_ids).cuda().long()
            else:
                label_ids = labs  # 测试时 labs 就是身份索引

            feature_smote_cfg = self.cfgs.get('trainer_cfg', {}).get('feature_smote', {})
            _, logits = self.BNNecks(embs)
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embs, 'labels': labs},
                    'softmax': {
                        'logits': logits,
                        'labels': label_ids,
                        'embeddings': embs,
                        'classifier': self.BNNecks,
                        'feature_smote': feature_smote_cfg,
                    },
                },
                'visual_summary': {
                    'image/sils': sils.view(n*s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': logits  # 分类模式下使用logits作为特征
                }
            }
        else:
            # 原始识别模式
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embs, 'labels': labs}
                },
                'visual_summary': {
                    'image/sils': sils.view(n*s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': embs
                }
            }
        return retval
