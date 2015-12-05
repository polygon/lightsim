import numpy as np


class LightsimConfig(object):
    def __init__(self, width=30, thickness=1, etch_pattern=None, light_dist=5):
        self.width = float(width)
        self.thickness = float(thickness)
        self.etch_pattern = etch_pattern
        self.light_dist = float(light_dist)
        self.lit_top = True
        self.lit_bottom = True
        self.lit_left = True
        self.lit_right = True
        self.n_air = 1.00
        self.n_mat = 1.5

    @property
    def length(self):
        if self.etch_pattern is None:
            length = None
        else:
            num_x, num_y = self.etch_pattern.shape
            length = self.width / num_x * num_y
        return length

    @property
    def alpha_max(self):
        alpha_max = np.pi/2 - np.arccos(self.n_air / self.n_mat)
        return alpha_max


class Lightsim(object):
    def __init__(self, cfg: LightsimConfig):
        self.cfg = cfg


class LightSource(object):
    def __init__(self, cfg: LightsimConfig, pos_x, pos_y, theta_min, theta_max):
        self.cfg = cfg
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.hits = np.zeros_like(self.cfg.etch_pattern, dtype=np.int)

    def simulate_rays(self, N):
        dirs_x, dirs_y = self._random_direction_vectors(N)
        dists, offsets = self._random_angles(N)
        min_dist = 2 * self.cfg.thickness * np.cos(self.cfg.alpha_max)
        max_hops = np.ceil(np.sqrt(self.cfg.width**2 + self.cfg.length**2) / min_dist) + 1
        offsets_x = dirs_x * offsets
        offsets_y = dirs_y * offsets
        hops_x = np.dot(np.arange(max_hops)[:, np.newaxis], dirs_x[np.newaxis, :]) + offsets_x[np.newaxis, :]
        hops_y = np.dot(np.arange(max_hops)[:, np.newaxis], dirs_y[np.newaxis, :]) + offsets_y[np.newaxis, :]
        coord_x = np.round(hops_x / self.cfg.width * self.hits.shape[0]).astype(np.int)
        coord_y = np.round(hops_y / self.cfg.length * self.hits.shape[1]).astype(np.int)
        hit_coords = self.cfg.etch_pattern[coord_x, coord_y]
        first_hits = np.argmax(hit_coords, axis=1)
        first_hits_idx_x = coord_x[:, first_hits]

    def _random_direction_vectors(self, N):
        thetas = np.random.uniform(self.theta_min, self.theta_max, N)
        dirs_x = np.cos(thetas)
        dirs_y = np.sin(thetas)
        return dirs_x, dirs_y

    def _random_angles(self, N):
        alphas = np.random.uniform(0.0, self.cfg.alpha_max, N)
        dists = 2 * self.cfg.thickness / np.sin(alphas)
        offsets = np.random.uniform(0., 1., N) * dists
        return dists, offsets

