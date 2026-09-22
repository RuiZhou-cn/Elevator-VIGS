"""The final bundle adjustment over the whole keyframe graph (App. C), run by `--offline`.

Its edge budget is capped by the free GPU memory, and no edge touches a keyframe from a folded
ride, so those keyframes keep the rise the fold wrote into their poses.
"""
import torch
from factor_graph import FactorGraph


class TrackBackend:
    def __init__(self, net, video, config):
        self.video = video
        self.update_op = net.update

        self.backend_thresh = config["backend_thresh"]
        self.backend_radius = config["backend_radius"]
        self.backend_nms = config["backend_nms"]

    @torch.no_grad()
    def __call__(self, steps=12, inertial=False):
        """ main update """
        torch.cuda.empty_cache()

        t = self.video.counter.value
        # edge budget: upstream's min(1e4, 20 t), capped by what fits. Per edge the graph holds
        # the fp16 hidden state (256 B/px) + target, weight, coords0 (24 B/px); each step adds
        # coords1, mask, motn (28 B/px) and the BA kernel its Eii/Eij/Cii/wi blocks (56 B/px):
        # ~1.4 MB per edge at 43x80. add_proximity_factors keeps the CLOSEST candidates, so a
        # cap degrades the pass gracefully instead of running out of memory after the whole run.
        ht, wd = self.video.disps.shape[1:]
        free = torch.cuda.mem_get_info()[0]
        max_factors = min(int(1e4), 20 * t, int(0.85 * free / (384 * ht * wd)))
        if max_factors < min(int(1e4), 20 * t):
            print(f"[backend] edge budget capped at {max_factors} by free GPU memory ({free / 2**30:.1f} GB)")
        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=max_factors)
        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, backend=True)

        graph.update_lowmem(steps=steps, inertial=inertial)
        graph.clear_edges()
        self.video.dirty[:t] = True
