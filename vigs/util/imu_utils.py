import numpy as np
import torch


def up_axis(Rwg, g, eps=1e-12):
    """The up axis e_z = -g/||g|| of the paper (Sec. 3.1), from the IMU-init gravity:
    -(Rwg @ g) / ||.||. `g` (init_g) points DOWN in the gravity-aligned frame, so the negation
    gives up. This is the ONE place the elevator stack's up-axis sign convention lives -- call
    it instead of re-deriving the idiom, where a sign or normalization slip is easy to make."""
    v = -(np.asarray(Rwg) @ np.asarray(g))
    return v / (np.linalg.norm(v) + eps)


def up_axis_tensor(video, device):
    """`up_axis(video.Rwg, video.init_g)` as a float32 tensor on `device`, cached on `video`.

    Rwg is fixed once the IMU init has run, and the in-loop BA and the ride hooks ask for this
    vector several times per keyframe; the cache is keyed on the values, so a re-run of the
    init invalidates it."""
    key = (np.asarray(video.Rwg).tobytes(), np.asarray(video.init_g, dtype=np.float64).tobytes(),
           str(device))
    cache = getattr(video, "_up_axis_cache", None)
    if cache is None or cache[0] != key:
        cache = (key, torch.tensor(up_axis(video.Rwg, video.init_g), dtype=torch.float32,
                                   device=device))
        video._up_axis_cache = cache
    return cache[1]
