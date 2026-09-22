import numpy as np
import torch
import lietorch
from lietorch import SE3
from factor_graph import FactorGraph


class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses """

    def __init__(self, net, video, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image)

    @torch.no_grad()
    def fill(self, tstamps, images):
        """ fill operator """

        tt = torch.as_tensor(tstamps, device="cuda")
        images = torch.stack(images, 0)
        inputs = images.to(self.device) / 255.0
        
        # linear pose interpolation
        N = self.video.counter.value
        M = len(tstamps)

        ts = self.video.tstamp[:N]
        Ps = SE3(self.video.poses[:N])

        t0 = torch.as_tensor([ts[ts<=t].shape[0] - 1 for t in tstamps])
        t1 = torch.where(t0<N-1, t0+1, t0)

        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(-1)
        w = v * (tt - ts[t0]).unsqueeze(-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        self.video.counter.value += M
        # 8-element item: no keyframe stamp (index 10), so these temporary rows get no IMU
        # preintegration
        self.video[N:N+M] = (tt, images[:,0], Gs.data, 1, None, None, None, fmap)

        graph = FactorGraph(self.video, self.update)
        graph.add_factors(t0.cuda(), torch.arange(N, N+M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N+M).cuda())

        for itr in range(6):
            graph.update(N, N+M, motion_only=True)
    
        Gs = SE3(self.video.poses[N:N+M].clone())
        self.video.counter.value -= M
        return [ Gs ]

    @torch.no_grad()
    def fill_in_elevator(self, tstamps, images, rec):
        """fill() for a batch whose neighbour keyframes all lie inside ONE folded elevator ride.

        After the fold of Eq. (9) those keyframes sit at different heights while the images
        show the same static interior, so a frame's two flow factors contradict each other by
        the height difference: the solve snaps to one neighbour (a staircase in height, the
        remainder taken up by pitch) and consecutive frames flip between the two. Solve in the
        elevator frame E instead -- undo the fold on the ride's keyframes (poses flat at the
        departure height, the state the ride was tracked in), fill, then give every frame the
        ride's h at its own time, linear between the keyframes. The keyframe poses are restored
        untouched."""
        N = self.video.counter.value
        ts = self.video.tstamp[:N]
        e_z = torch.tensor(rec["e_z"], dtype=torch.float, device=self.video.poses.device)
        rows, hvals = [], []
        for t_k, h_k in zip(rec["ts"], rec["h"]):
            m = (ts == t_k).nonzero()
            if len(m):
                rows.append(int(m[0]))
                hvals.append(float(h_k))
        saved = self.video.poses[rows].clone()
        for k, h_k in zip(rows, hvals):
            R = SE3(self.video.poses[k][None]).matrix()[0, :3, :3]
            self.video.poses[k, :3] += h_k * (R @ e_z)          # undo the fold's -h_k lift
        try:
            Gs = self.fill(tstamps, images)[0]
        finally:
            self.video.poses[rows] = saved
        h_t = torch.tensor(np.interp(tstamps, rec["ts"], rec["h"]), dtype=torch.float, device=e_z.device)
        data = Gs.data.clone()
        R = Gs.matrix()[:, :3, :3]
        data[:, :3] -= h_t[:, None] * (R @ e_z)                 # the fold, at each frame's own h
        return [SE3(data)]

    @torch.no_grad()
    def __call__(self, image_stream):
        """ fill in poses of non-keyframe images """
        N = self.video.counter.value
        ts = self.video.tstamp[:N].cpu().numpy()
        recs = self.video.elev_fold_records

        def ride_of(t):
            """Index of the folded ride whose keyframes bracket frame stamp t on both sides."""
            k0 = int(np.searchsorted(ts, t, "right")) - 1
            if k0 >= 0:
                for r, rec in enumerate(recs):
                    if rec["depart_ts"] <= ts[k0] and t < rec["arrive_ts"]:
                        return r
            return None

        # store all camera poses
        pose_list = []
        # batches of 16 frames, never straddling a change of bracketing ride
        tstamps, images, ride = [], [], None
        for (tstamp, image) in image_stream.items():
            r = ride_of(tstamp)
            if tstamps and (r != ride or len(tstamps) == 16):
                pose_list += (self.fill(tstamps, images) if ride is None
                              else self.fill_in_elevator(tstamps, images, recs[ride]))
                tstamps, images = [], []
            ride = r
            tstamps.append(tstamp)
            images.append(image)

        if len(tstamps) > 0:
            pose_list += (self.fill(tstamps, images) if ride is None
                          else self.fill_in_elevator(tstamps, images, recs[ride]))

        # stitch pose segments together
        return lietorch.cat(pose_list, 0)
