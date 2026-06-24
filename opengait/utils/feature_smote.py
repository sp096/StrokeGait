import numpy as np
import torch

# NOTE:
# - imbalanced-learn is an optional dependency and can be expensive to import
#   (scikit-learn / OpenMP). We import lazily inside the function below so that
#   importing this module does not trigger heavy imports during process spawn
#   (which can cause crashes on Windows when using torch.distributed.launch).


def _as_long_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    return torch.as_tensor(x, device=device, dtype=torch.long)


def feature_smote_embeddings(embeddings, labels, sampling_strategy='auto', k_neighbors=1,
                             random_state=0, fallback='RandomOverSampler', safe_mode=True,
                             normalize_embeddings=True, clamp_value=5.0,
                             min_samples_for_smote=4, nan_to_num=True,
                             mode='smote', max_synthetic_per_class=None, synthetic_ratio=1.0):
    """Apply SMOTE on flattened feature embeddings.

    Args:
        embeddings: Tensor with shape [N, C, P].
        labels: Tensor with shape [N].
    Returns:
        (aug_embeddings, aug_labels, info)
    """
    if not torch.is_tensor(embeddings):
        raise TypeError(f"embeddings should be a torch.Tensor, got {type(embeddings)}")
    if embeddings.ndim != 3:
        raise ValueError(f"embeddings should have shape [N, C, P], got {tuple(embeddings.shape)}")

    device = embeddings.device
    labels = _as_long_tensor(labels, device)
    if labels.ndim != 1:
        labels = labels.view(-1)

    if not torch.isfinite(embeddings).all():
        info = {
            'enabled': False,
            'method': 'none',
            'reason': 'non_finite_input',
            'original_samples': int(embeddings.size(0)),
            'resampled_samples': int(embeddings.size(0)),
            'synthetic_samples': 0,
        }
        return embeddings, labels, info

    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    info = {
        'enabled': False,
        'method': 'none',
        'original_samples': int(embeddings.size(0)),
        'resampled_samples': int(embeddings.size(0)),
        'synthetic_samples': 0,
    }

    if len(unique_labels) < 2:
        info['reason'] = 'single_class'
        return embeddings, labels, info

    # Lazy import imbalanced-learn classes required for selected mode. This
    # prevents heavy top-level imports and allows graceful fallback when the
    # package is not available.
    RandomOverSampler = None
    SMOTE = None
    TomekLinks = None
    SMOTETomek = None
    if mode in ('smote', 'random_over', 'smote_tomek', 'smotetomek', 'smote+tomek', 'tomek', 'tomek_links', 'tomeklinks'):
        try:
            from imblearn.over_sampling import RandomOverSampler, SMOTE
            from imblearn.under_sampling import TomekLinks
            from imblearn.combine import SMOTETomek
        except Exception:
            # keep variables as None to handle gracefully below
            RandomOverSampler = None
            SMOTE = None
            TomekLinks = None
            SMOTETomek = None

    # Validate availability for the chosen mode and provide a graceful
    # informative error (do not crash the whole process unexpectedly).
    if mode == 'smote' and SMOTE is None:
        raise ImportError("imbalanced-learn is required for feature SMOTE. Please install it with `pip install imbalanced-learn`.")
    if mode in ('tomek', 'tomek_links', 'tomeklinks') and TomekLinks is None:
        raise ImportError("imbalanced-learn is required for TomekLinks. Please install it with `pip install imbalanced-learn`.")
    if mode in ('smote_tomek', 'smotetomek', 'smote+tomek') and SMOTETomek is None:
        raise ImportError("imbalanced-learn is required for SMOTETomek. Please install it with `pip install imbalanced-learn`.")

    min_count = int(counts.min())
    # Decide method based on mode and class counts
    method = 'none'
    if mode == 'smote':
        if min_count < 2:
            # fallback to random over-sampling when SMOTE not applicable
            if fallback == 'RandomOverSampler' and RandomOverSampler is not None:
                sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
                method = 'RandomOverSampler'
            else:
                info['reason'] = 'min_class_count_lt_2'
                return embeddings, labels, info
        else:
            effective_k = min(int(k_neighbors), min_count - 1)
            if effective_k < 1:
                effective_k = 1
            sampler = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=effective_k,
                            random_state=random_state)
            method = 'SMOTE'
    elif mode == 'random_over':
        if RandomOverSampler is None:
            info['reason'] = 'random_over_not_available'
            return embeddings, labels, info
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        method = 'RandomOverSampler'
    elif mode == 'interpolate':
        # lightweight interpolation-based synthesis (cheap SMOTE-like) implemented below
        method = 'interpolate'
    elif mode in ('tomek', 'tomek_links', 'tomeklinks'):
        # TomekLinks is an under-sampling / cleaning method
        sampler = TomekLinks(sampling_strategy=sampling_strategy)
        method = 'TomekLinks'
    elif mode in ('smote_tomek', 'smotetomek', 'smote+tomek'):
        # Combined SMOTE + Tomek cleaning (uses imblearn.combine.SMOTETomek)
        # This will first synthesize then clean border samples. It's heavier but often more robust.
        sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
        method = 'SMOTETomek'
    else:
        info['reason'] = f'unknown_mode_{mode}'
        return embeddings, labels, info

    flat_embeddings = embeddings.detach().permute(0, 2, 1).contiguous().view(embeddings.size(0), -1)
    flat_embeddings = flat_embeddings.float()
    if normalize_embeddings:
        norms = flat_embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        flat_embeddings = flat_embeddings / norms
    if nan_to_num:
        flat_embeddings = torch.nan_to_num(flat_embeddings, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
    flat_embeddings_np = flat_embeddings.cpu().numpy()

    if safe_mode and min_count < max(int(min_samples_for_smote), int(k_neighbors) + 1):
        if mode == 'smote':
            # try fallback to random over sampler
            if fallback == 'RandomOverSampler' and RandomOverSampler is not None:
                sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
                method = 'RandomOverSampler'
            else:
                # allow interpolate fallback
                if mode != 'interpolate':
                    info['reason'] = 'min_class_count_too_small_for_smote'
                    return embeddings, labels, info

    # three possible paths: SMOTE (uses imblearn), RandomOverSampler, TomekLinks (cleaning), SMOTETomek (combine), or interpolate (cheap)
    if method in ('SMOTE', 'RandomOverSampler', 'TomekLinks', 'SMOTETomek'):
        # fit_resample can raise various exceptions (e.g. unexpected data
        # shape/ dtype or internal sklearn errors). We catch exceptions and
        # attempt a safe fallback to RandomOverSampler when possible; if
        # fallback is not available we return the original embeddings rather
        # than let the whole process crash.
        try:
            resampled_embeddings_np, resampled_labels_np = sampler.fit_resample(flat_embeddings_np, labels_np)
        except Exception as e:
            # attempt fallback to RandomOverSampler
            if fallback == 'RandomOverSampler' and 'RandomOverSampler' in globals() and RandomOverSampler is not None and method != 'RandomOverSampler':
                try:
                    sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
                    method = 'RandomOverSampler'
                    resampled_embeddings_np, resampled_labels_np = sampler.fit_resample(flat_embeddings_np, labels_np)
                except Exception:
                    # last resort: return original embeddings with info
                    info.update({'reason': f'fit_resample_failed: {str(e)}'})
                    return embeddings, labels, info
            else:
                info.update({'reason': f'fit_resample_failed: {str(e)}'})
                return embeddings, labels, info

        resampled_embeddings = torch.from_numpy(resampled_embeddings_np).to(device=device, dtype=torch.float32)
        resampled_embeddings = resampled_embeddings.view(-1, embeddings.size(2), embeddings.size(1)).permute(0, 2, 1).contiguous()
        resampled_labels = torch.from_numpy(resampled_labels_np).to(device=device, dtype=labels.dtype)
    elif method == 'interpolate':
        # interpolation-based lightweight synthesis (no sklearn, low memory)
        # operate in torch on CPU to reduce GPU memory pressure
        flat = flat_embeddings.cpu()
        labels_cpu = torch.from_numpy(labels_np)
        unique, counts = torch.unique(labels_cpu, return_counts=True)
        majority = int(counts.max().item())
        synth_list = []
        synth_labels_list = []
        for ul, cnt in zip(unique.tolist(), counts.tolist()):
            cnt = int(cnt)
            if cnt >= majority:
                continue
            needed = int(majority - cnt)
            # cap by max_synthetic_per_class if provided
            if max_synthetic_per_class is not None:
                needed = min(needed, int(max_synthetic_per_class))
            # optionally reduce by synthetic_ratio (0..1)
            needed = int(needed * float(synthetic_ratio)) if synthetic_ratio < 1.0 else needed
            if needed <= 0:
                continue
            idxs = (labels_cpu == ul).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                continue
            # sample pairs with replacement
            for _ in range(needed):
                a = idxs[torch.randint(0, idxs.numel(), (1,)).item()]
                b = idxs[torch.randint(0, idxs.numel(), (1,)).item()]
                alpha = torch.rand(1)
                synth = flat[a] + alpha * (flat[b] - flat[a])
                if nan_to_num:
                    synth = torch.nan_to_num(synth, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
                synth_list.append(synth.unsqueeze(0))
                synth_labels_list.append(ul)

        if len(synth_list) == 0:
            # nothing synthesized
            resampled_embeddings = embeddings
            resampled_labels = labels
            info.update({'method': 'interpolate', 'enabled': False, 'reason': 'no_synthetic_generated'})
            return resampled_embeddings, resampled_labels, info

        synth_flat = torch.cat(synth_list, dim=0)  # [S, D]
        synth_labels_np = np.array(synth_labels_list, dtype=np.int64)
        # convert synth back to original shape
        synth_flat = synth_flat.to(dtype=torch.float32)
        synth_embeddings = synth_flat.view(-1, embeddings.size(2), embeddings.size(1)).permute(0, 2, 1).contiguous()
        resampled_embeddings = torch.cat([embeddings.cpu(), synth_embeddings], dim=0)
        resampled_labels = torch.cat([labels.cpu(), torch.from_numpy(synth_labels_np)], dim=0)
        # move back to original device if needed
        resampled_embeddings = resampled_embeddings.to(device=device, dtype=torch.float32)
        resampled_labels = resampled_labels.to(device=device, dtype=labels.dtype)
    else:
        info['reason'] = 'unsupported_method'
        return embeddings, labels, info

    original_size = int(embeddings.size(0))
    # If method is TomekLinks, this is a cleaning operation: the resampled set may be
    # smaller than the original. We should return the cleaned set directly.
    if method == 'TomekLinks':
        info.update({
            'enabled': True,
            'method': method,
            'original_samples': original_size,
            'resampled_samples': int(resampled_embeddings.size(0)),
            'synthetic_samples': 0,
        })
        return resampled_embeddings, resampled_labels, info

    # For over-sampling methods (SMOTE/RandomOverSampler) we expect resampled set to be larger
    if resampled_embeddings.size(0) <= original_size:
        info.update({
            'enabled': False,
            'method': 'none',
            'resampled_samples': original_size,
            'synthetic_samples': 0,
        })
        return embeddings, labels, info

    synthetic_embeddings = resampled_embeddings[original_size:].detach()
    synthetic_labels = resampled_labels[original_size:].detach()
    resampled_embeddings = torch.cat([embeddings, synthetic_embeddings], dim=0)
    resampled_labels = torch.cat([labels, synthetic_labels], dim=0)

    if nan_to_num:
        resampled_embeddings = torch.nan_to_num(resampled_embeddings, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
    if clamp_value is not None:
        resampled_embeddings = resampled_embeddings.clamp(min=-float(clamp_value), max=float(clamp_value))

    info.update({
        'enabled': True,
        'method': method,
        'original_samples': original_size,
        'resampled_samples': int(resampled_embeddings.size(0)),
        'synthetic_samples': int(resampled_embeddings.size(0) - original_size),
    })
    return resampled_embeddings, resampled_labels, info

