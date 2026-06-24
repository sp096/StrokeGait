import torch
import torch.nn.functional as F

from .base import BaseLoss
from utils.feature_smote import feature_smote_embeddings


class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False, weight=None):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float32)
        else:
            self.weight = None

    def forward(self, logits=None, labels=None, embeddings=None, classifier=None, feature_smote=None):
        """
            logits: [n, c, p]
            labels: [n]
        """
        smote_info = None
        if logits is None:
            if embeddings is None or classifier is None:
                raise ValueError("CrossEntropyLoss expects either logits or (embeddings, classifier).")
            embeddings = embeddings.float()
            _, logits = classifier(embeddings)

        if embeddings is None and feature_smote is not None and feature_smote.get('enabled', False):
            raise ValueError("feature_smote.enabled requires embeddings and classifier in training_feat['softmax'].")

        n, c, p = logits.size()
        logits = logits.float()
        weight = self.weight.to(logits.device) if self.weight is not None else None
        labels_2d = labels.unsqueeze(1)

        def compute_ce(_logits, _labels):
            _labels = _labels.unsqueeze(1)
            _p = _logits.size(2)
            if self.label_smooth:
                return F.cross_entropy(
                    _logits * self.scale, _labels.repeat(1, _p), label_smoothing=self.eps, weight=weight)
            return F.cross_entropy(_logits * self.scale, _labels.repeat(1, _p), weight=weight)

        base_loss = compute_ce(logits, labels)
        loss = base_loss


        if feature_smote is not None and feature_smote.get('enabled', False):
            smote_embeddings, smote_labels, smote_info = feature_smote_embeddings(
                embeddings,
                labels,
                sampling_strategy=feature_smote.get('sampling_strategy', 'auto'),
                k_neighbors=feature_smote.get('k_neighbors', 1),
                random_state=feature_smote.get('random_state', 0),
                fallback=feature_smote.get('fallback', 'RandomOverSampler'),
                safe_mode=feature_smote.get('safe_mode', True),
                normalize_embeddings=feature_smote.get('normalize_embeddings', True),
                clamp_value=feature_smote.get('clamp_value', 5.0),
                min_samples_for_smote=feature_smote.get('min_samples_for_smote', 4),
                nan_to_num=feature_smote.get('nan_to_num', True),
                mode=feature_smote.get('mode', 'smote'),
                max_synthetic_per_class=feature_smote.get('max_synthetic_per_class', None),
                synthetic_ratio=feature_smote.get('synthetic_ratio', 1.0))

            if smote_embeddings.size(0) < 2:
                smote_info = dict(smote_info or {})
                smote_info.update({
                    'enabled': False,
                    'reason': 'batch_too_small_for_bn_classifier',
                    'resampled_samples': int(smote_embeddings.size(0)),
                    'synthetic_samples': int(smote_info.get('synthetic_samples', 0) if smote_info else 0),
                })
                self.info.update({
                    'smote_enabled': torch.tensor(0.0, device=logits.device),
                    'smote_synthetic': torch.tensor(float(smote_info.get('synthetic_samples', 0.0)), device=logits.device),
                })
            elif smote_info.get('enabled', False) and smote_info.get('resampled_samples', smote_embeddings.size(0)) != embeddings.size(0):
                smote_embeddings = smote_embeddings.float()
                _, smote_logits = classifier(smote_embeddings)
                smote_loss = compute_ce(smote_logits, smote_labels)
                loss = 0.5 * (base_loss + smote_loss)
                self.info.update({
                    'smote_loss': smote_loss.detach().clone(),
                    'smote_enabled': torch.tensor(1.0, device=logits.device),
                    'smote_synthetic': torch.tensor(float(smote_info.get('synthetic_samples', 0.0)), device=logits.device),
                })
            else:
                self.info.update({
                    'smote_enabled': torch.tensor(0.0, device=logits.device),
                    'smote_synthetic': torch.tensor(float(smote_info.get('synthetic_samples', 0.0)), device=logits.device),
                })

        self.info.update({'base_loss': base_loss.detach().clone(), 'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels_2d).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info
