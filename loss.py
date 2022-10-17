# Hard Triplet Loss

# The following class implements the hard triplet loss. Hard means that the closest negative and furthest positive are choosen instead of random.

class HardTripletLoss():

    def __init__(self, device, margin=1.0):
        self.margin = margin
        self.device = device

    def _get_anchor_positive_triplet_mask(self, labels):
        labels_mat = labels.unsqueeze(0).repeat(labels.shape[0], 1)
        res = (labels_mat == labels_mat.T).int().to(self.device)
        return res

    def _get_anchor_negative_triplet_mask(self, labels):
        labels_mat = labels.unsqueeze(0).repeat(labels.shape[0], 1)
        res = (labels_mat != labels_mat.T).int().to(self.device)
        return res

    def _get_dist_matrix(self, embeddings):
        return torch.cdist(embeddings, embeddings).to(self.device)**2

    def __call__(self, embeddings, labels):
        """Build the triplet loss over a batch of embeddings.

        For each anchor, we get the hardest positive and hardest negative to form a triplet.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._get_dist_matrix(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = torch.max(
            anchor_positive_dist, axis=1, keepdims=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = torch.max(
            pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + \
            max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = torch.min(
            anchor_negative_dist, axis=1, keepdims=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = torch.max(hardest_positive_dist -
                                 hardest_negative_dist + self.margin, torch.zeros_like(hardest_negative_dist))

        # Get final mean triplet loss
        triplet_loss = torch.mean(triplet_loss)

        return triplet_loss

