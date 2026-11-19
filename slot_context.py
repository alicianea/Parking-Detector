"""Residual occupancy refinement with one graph per camera frame."""
from hashlib import sha256

import torch
from torch import nn
from torch.nn import functional as F


def slot_geometry(valid_spaces, image_shape):
     # Store each rectangle as normalized (x, y, width, height) in crop order.
    height, width = image_shape[:2]
    rectangles = [
        (space["x"] / width, space["y"] / height,
         space["w"] / width, space["h"] / height)
        for space, *_ in valid_spaces
    ]
    return torch.tensor(rectangles, dtype=torch.float32)


def nearest_slots(centers, count):
    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(float("inf"))
    return distances.topk(count, largest=False).indices


class SlotContext(nn.Module):
    def __init__(self, feature_size, neighbors=4):
        super().__init__()
        self.neighbors = neighbors
        # Each slot contributes its crop features, two class scores, and four
        # rectangle values. Project that input to a 64-value node representation.
        self.project = nn.Sequential(nn.Linear(feature_size + 6, 64), nn.ReLU())
        # Each message uses the neighbor node plus four relative geometry values.
        self.message = nn.Sequential(nn.Linear(68, 64), nn.ReLU())
        # Combine the slot and its pooled neighbor context to adjust both logits.
        self.adjust = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
        # Start with zero corrections so initial predictions match MobileNetV2.
        nn.init.zeros_(self.adjust[-1].weight)
        nn.init.zeros_(self.adjust[-1].bias)

    @staticmethod
    @torch.no_grad()
    def encode(model, crops):
        # Pool each crop into one feature vector and obtain its original logits.
        # The caller puts MobileNetV2 in evaluation mode. No gradients flow here.
        features = F.adaptive_avg_pool2d(model.features(crops), 1).flatten(1)
        return features, model.classifier(features)

    def forward(self, features, baseline, geometry):
        # One call processes one camera frame. Each row represents one slot.
        # The feature, logit, and geometry rows all follow the same slot order.
        slots = self.project(torch.cat((features, baseline, geometry), dim=1))
        # Zero context covers single-slot frames and the neighbors=0 comparison.
        context = torch.zeros_like(slots)
        count = min(self.neighbors, len(slots) - 1)
        if count > 0:
            # Find the closest slots by center distance in normalized image space.
            # An infinite self-distance keeps each slot out of its own neighbor list.
            centers = geometry[:, :2] + geometry[:, 2:] / 2
            indices = nearest_slots(centers, count)
            # Describe each neighbor relative to the receiving slot using center
            # offsets and log ratios for width and height.
            offsets = centers[indices] - centers[:, None]
            size_ratios = (geometry[indices, 2:] / geometry[:, None, 2:]).log()
            messages = self.message(torch.cat((slots[indices], offsets, size_ratios), dim=-1))
            # Average the neighbor messages into one context vector per slot.
            context = messages.mean(dim=1)
        # Add the learned correction to the original MobileNetV2 logits.
        return baseline + self.adjust(torch.cat((slots, context), dim=1))


def checkpoint_digest(path):
    # Hash the checkpoint in chunks so its exact contents can be checked.
    digest = sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_context(path, baseline_path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    # The head was trained on features and scores from this exact baseline.
    if checkpoint["baseline_sha256"] != checkpoint_digest(baseline_path):
        raise ValueError("This context checkpoint requires the baseline used during its training.")
    # Rebuild the head with the settings saved during training.
    model = SlotContext(checkpoint["feature_size"], checkpoint["neighbors"]).to(device)
    model.load_state_dict(checkpoint["context_state_dict"])
    return model.eval()
