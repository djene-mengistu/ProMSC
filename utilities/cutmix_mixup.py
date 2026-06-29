
import torch
import numpy as np
import random
#For cross-data augumentaiton
# ----- Step 4: MixUp function for hard labels -----
def mixup_images_and_labels(x1, x2, y1, y2, alpha=0.75):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Optional: enforce dominance
    mixed_x = lam * x1 + (1 - lam) * x2

    # NOTE: y1, y2 are now integers (B, H, W)
    # Randomly choose pixel-wise from y1 or y2
    mask = torch.rand_like(y1.float()) < lam
    mixed_y = torch.where(mask, y1, y2)
    return mixed_x, mixed_y

def mixup_data(x1, x2, y1, y2, alpha=0.7):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Optional: enforce larger weight for stability
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y

#CUTMIX
def cutmix_tensor(img1, img2, label1, label2, beta=1.0):
    """
    Perform CutMix between two tensors:
    - img1, img2: [3, H, W]
    - label1, label2: [H, W] (integer labels)
    """
    _, H, W = img1.shape
    lam = np.random.beta(beta, beta)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Clone for mixing
    mixed_img = img1.clone()
    mixed_img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]

    mixed_label = label1.clone()
    mixed_label[y1:y2, x1:x2] = label2[y1:y2, x1:x2]

    return mixed_img, mixed_label, lam

##MORE GENERAL CUTMIX FOR BATCHES
def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio
    w = np.random.randint(img_size[1] // ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)
    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()

def generate_crossmix_data(
    data_l, labels_l,
    data_wk, data_st,
    pseudo_labels,
    mode='cutmix',
    p=0.3
):
    """
    Mix labeled and unlabeled (wk, st) images using CutMix.
    Labels_l: GT for labeled.
    Pseudo_labels: hard pseudo-labels for unlabeled wk.
    """
    batch_size, _, im_h, im_w = data_l.shape
    device = data_l.device

    new_data_wk, new_data_st, new_labels = [], [], []

    for i in range(batch_size):
        if mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
            mix_mask_img = mix_mask  # [H, W], float
            mix_mask_lbl = mix_mask.bool()  # for indexing labels

        if random.random() < p:
            # Mix images: labeled + unlabeled wk/st
            img_wk = data_wk[i] * mix_mask_img + data_l[i] * (1 - mix_mask_img)
            img_st = data_st[i] * mix_mask_img + data_l[i] * (1 - mix_mask_img)

            # Mix labels: use labeled label and pseudo-labels
            mixed_label = pseudo_labels[i].clone()
            mixed_label[mix_mask_lbl == 0] = labels_l[i][mix_mask_lbl == 0]
        else:
            img_wk = data_wk[i]
            img_st = data_st[i]
            mixed_label = pseudo_labels[i]

        new_data_wk.append(img_wk.unsqueeze(0))
        new_data_st.append(img_st.unsqueeze(0))
        new_labels.append(mixed_label.unsqueeze(0))

    new_data_wk = torch.cat(new_data_wk)
    new_data_st = torch.cat(new_data_st)
    new_labels = torch.cat(new_labels).long()

    return new_data_wk, new_data_st, new_labels
