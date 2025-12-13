from torch.utils.data import Sampler
import numpy as np

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, anom_frac=1/3, seed=42):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.anom_bs = max(1, int(batch_size * anom_frac))
        self.norm_bs = batch_size - self.anom_bs

        self.norm_idx = np.where(self.labels == 0)[0]
        self.anom_idx = np.where(self.labels == 1)[0]

        self.rng = np.random.RandomState(seed)
        self.num_batches = int(np.ceil(len(self.norm_idx) / self.norm_bs))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        norm = self.norm_idx.copy()
        anom = self.anom_idx.copy()
        self.rng.shuffle(norm)
        self.rng.shuffle(anom)

        n_ptr, a_ptr = 0, 0

        for _ in range(self.num_batches):
            if n_ptr + self.norm_bs > len(norm):
                self.rng.shuffle(norm)
                n_ptr = 0
            if a_ptr + self.anom_bs > len(anom):
                self.rng.shuffle(anom)
                a_ptr = 0

            batch = np.concatenate([
                norm[n_ptr:n_ptr + self.norm_bs],
                anom[a_ptr:a_ptr + self.anom_bs]
            ])

            n_ptr += self.norm_bs
            a_ptr += self.anom_bs

            self.rng.shuffle(batch)
            yield batch.tolist()
