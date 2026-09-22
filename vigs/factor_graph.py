"""The co-visibility factor graph of the dense bundle adjustment.

Edges (i, j) carry the ConvGRU's dense correspondences and confidences (Eq. 1); `update` runs the
operator over the edges and hands the visual factors to the solver in `depth_video`. Edge
proposal skips pairs that straddle a ride not yet folded (Sec. 3.3, first rule), and the final
bundle adjustment adds no edge to a keyframe from a folded ride (App. C).
"""
import torch
import numpy as np

from tqdm import trange
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops


class FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)
        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """remove duplicate edges"""
        # filter if (ii, jj) insersect with any active edge
        if len(self.ii) > 0:
            mask = ((ii[:, None] == self.ii) & (jj[:, None] == self.jj)).any(dim=-1)
            ii = ii[~mask]
            jj = jj[~mask]

        # filter if (ii, jj) intersect with any inactive edge
        if len(self.ii_inac) > 0:
            mask = ((ii[:, None] == self.ii_inac) & (jj[:, None] == self.jj_inac)).any(
                dim=-1
            )
            ii = ii[~mask]
            jj = jj[~mask]

        return ii, jj


    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.amp.autocast("cuda", enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)


        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.amp.autocast("cuda", enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.amp.autocast("cuda", enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            if hasattr(self.video, 'pgobuf'):
                self.video.pgobuf.add_rel_poses(self.ii[mask], self.jj[mask], self.target[:,mask], self.weight[:,mask])
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.amp.autocast("cuda", enabled=True)
    def rm_keyframe(self, ix):
        """ remove a keyframe: shift later keyframe state down by one index, reintegrate IMU, and drop edges touching this keyframe """

        with self.video.get_lock():
            self.video.tstamp[ix] = self.video.tstamp[ix+1]
            if self.video.store_images:
                self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_prior[ix] = self.video.disps_prior[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]
            if self.video.dense_outputs:
                self.video.disps_up[ix] = self.video.disps_up[ix+1]
                self.video.normals[ix] = self.video.normals[ix+1]
            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]
            self.video.rm_and_reintegrate(ix)


        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    @torch.amp.autocast("cuda", enabled=True)
    def get_network_update_full_graph(self, t0=None, EP=1e-7):
        """Run the update operator over EVERY edge and return the BA inputs.

        Used by IMU-init step 3 (joint visual-inertial optimization), which needs the whole
        graph rather than the sliding window.
        """
        indices = torch.cat([self.ii, self.jj])
        iu, iu_exp = torch.unique(indices, return_inverse=True)
        iu_exp, ju_exp = torch.split(iu_exp, [self.ii.shape[0], self.jj.shape[0]])

        # alternate corr implementation
        num, rig, ch, ht, wd = self.video.fmaps[:,:2].shape
        num = iu.shape[0]
        corr_op = AltCorrBlock(self.video.fmaps[iu.cpu(),:2].reshape(1, num*rig, ch, ht, wd).cuda())

        with torch.amp.autocast("cuda", enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

        s = 2
        # upmask is indexed compactly (one row per unique source KF, sorted), not by absolute KF
        # index, so it doesn't grow unbounded when imu_late_init_from runs on a long static start.
        uii = torch.unique(self.ii)
        upslot = torch.full((int(uii[-1]) + 1,), -1, dtype=torch.long, device=uii.device)
        upslot[uii] = torch.arange(uii.numel(), device=uii.device)
        self.upmask = torch.ones(uii.numel(), 576, ht, wd, device="cuda", dtype=torch.float16)
        for i in range(0, self.jj.max()+1, s):
            v = (self.ii >= i) & (self.ii < i + s)
            if not torch.any(v):
                continue
            iis = self.ii[v]
            jjs = self.jj[v]

            corr1 = corr_op(coords1[:,v], rig * iu_exp[v], rig * ju_exp[v] + (iis == jjs).long())
            with torch.amp.autocast("cuda", enabled=True):
                net, delta, weight, damping, upmask = \
                    self.update_op(self.net[:,v], self.video.inps[iis.cpu()].cuda()[None, ...], corr1, motn[:,v], iis)

            self.net[:,v] = net
            self.target[:,v] = coords1[:,v] + delta.float()
            self.weight[:,v] = weight.float()
            self.damping[torch.unique(iis)] = damping
            self.upmask[upslot[torch.unique(iis)]] = upmask

        damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP
        upmask = self.upmask.contiguous()   # already one row per unique(self.ii), sorted
        return t0, self.target, self.weight, damping[None], self.ii, self.jj, self.ii, upmask
        
    
    @torch.amp.autocast("cuda", enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False, use_mono=False, tracking=False, inertial=False, disable_vision=False):
        """ run update operator on factor graph """

        if tracking: # tracking mode is for speed, only optimize the last keyframe
            mask = self.ii == self.ii.max()
            t0 = self.ii.max().item()
            corr = self.corr.view(mask)
        else:
            # every edge: a slice is a view, an all-true boolean mask is a nonzero() device
            # sync plus a copy of every graph tensor (correlation pyramid included) per update
            mask = slice(None)
            corr = self.corr
            
        # motion features
        with torch.amp.autocast("cuda", enabled=False):
            coords1, _ = self.video.reproject(self.ii[mask], self.jj[mask])
            motn = torch.cat([coords1 - self.coords0, self.target[:,mask] - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
            
        # correlation features
        corr = corr(coords1)
        self.net[:,mask], delta, weight, damping, upmask = \
            self.update_op(self.net[:,mask], self.inp[:,mask], corr, motn, self.ii[mask])
        if disable_vision:
            weight = torch.zeros_like(weight)
            
        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.amp.autocast("cuda", enabled=False):
            self.target[:,mask] = coords1 + delta.to(dtype=torch.float)
            self.weight[:,mask] = weight.to(dtype=torch.float)

            self.damping[torch.unique(self.ii[mask])] = damping

            if tracking:
                ii, jj, target, weight = self.ii[mask], self.jj[mask], self.target[:,mask], self.weight[:,mask]
            else:
                if use_inactive:
                    m = (self.ii_inac >= t0 - 5) & (self.jj_inac >= t0 - 5)
                    ii = torch.cat([self.ii_inac[m], self.ii], 0)
                    jj = torch.cat([self.jj_inac[m], self.jj], 0)
                    target = torch.cat([self.target_inac[:,m], self.target], 1)
                    weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

                else:
                    ii, jj, target, weight = self.ii[mask], self.jj[mask], self.target, self.weight


            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP
            # dense bundle adjustment
            if inertial:
                self.video.inertial_ba(target, weight, damping, ii, jj, t0, t1, itrs, lm=1e-5, ep=0.01, use_mono=use_mono)
            else:
                self.video.cuda_ba(target, weight, damping, ii, jj, t0, t1, 
                    itrs=itrs, lm=1e-5, ep=0.01, motion_only=motion_only, use_mono=use_mono)
            if self.video.dense_outputs:   # a tracking run has no upsampled product to keep
                self.video.upsample(torch.unique(self.ii[mask]), upmask)

        self.age += 1


    @torch.amp.autocast("cuda", enabled=False)
    def update_lowmem(self, itrs=2, EP=1e-7, steps=8, inertial=False):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in (pbar := trange(steps)):
            pbar.set_description(f"Global BA Iteration #{step+1} with {t} keyframes {len(self.ii)} edges")
            with torch.amp.autocast("cuda", enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 2
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                if not v.any():                # keyframes with no edge (a folded ride)
                    continue
                iis = self.ii[v]
                jjs = self.jj[v]

                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.amp.autocast("cuda", enabled=True):
                 
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis)

                    if self.video.dense_outputs:
                        self.video.upsample(torch.unique(iis), upmask)

                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            # one entry per keyframe in [0, t) -- the kernel indexes eta by keyframe, and the
            # keyframes of a folded ride carry no edge, so unique(ii) would leave gaps
            damping = .2 * self.damping[:t].contiguous() + EP
            if inertial:
                self.video.inertial_ba(self.target, self.weight, damping, self.ii, self.jj, 10, t, 
                itrs=itrs, lm=1e-5, ep=1e-2, use_mono=False)
            else:
                # dense bundle adjustment
                self.video.cuda_ba(self.target, self.weight, damping, self.ii, self.jj, 10, t, 
                    itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False, use_mono=False)

            self.video.dirty[:t] = True

    @torch.amp.autocast("cuda", enabled=False)
    def update_pgba(self, t0=None, t1=None, itrs=2, EP=1e-7, steps=6):
        """ run update operator on factor graph - Sim3 pose-graph BA, chunked over edges """

        # alternate corr implementation
        if t1 is None:
            t1 = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in range(steps):
            with torch.amp.autocast("cuda", enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj, sim3=True)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 120
            E = self.ii.numel()
            for start in range(0, E, s):
                end = min(start + s, E)
                v = torch.arange(start, end, device='cuda')

                iis = self.ii[v]
                jjs = self.jj[v]

                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.amp.autocast("cuda", enabled=True):
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis)
                    if self.video.dense_outputs:
                        self.video.upsample(torch.unique(iis), upmask)

                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            ii = torch.cat([self.ii_inac, self.ii], 0)
            jj = torch.cat([self.jj_inac, self.jj], 0)
            target = torch.cat([self.target_inac, self.target], 1)
            weight = torch.cat([self.weight_inac, self.weight], 1)
            damping = .2 * self.damping[torch.unique(torch.cat((torch.arange(t0, t1, device='cuda'), ii)))].contiguous() + EP

            self.video.cuda_pgba(target, weight, damping, ii, jj, t0, t1, itrs=itrs, lm=1e-5, ep=1e-3)

        self.video.poses[:t1] = self.video.poses_sim3[:t1,:7]
        self.video.poses[:t1,:3] /= self.video.poses_sim3[:t1,-1:]
        self.video.pgobuf.rel_poses[:self.video.pgobuf.rel_N.value,:3] /= self.video.poses_sim3[self.video.pgobuf.rel_ii[:self.video.pgobuf.rel_N.value],-1:].cpu()

        ss = self.video.poses_sim3[:t1,-1:,None]
        self.video.disps[:t1] *= ss
        if self.video.dense_outputs:
            self.video.disps_up[:t1] *= ss.cpu()
        self.video.dscales[:t1] *= ss

    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1), indexing='ij')
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    @torch.no_grad()
    def nms_invalidate_(self, d, ii, jj, t0, t1, t, nms):
        """
        In-place: d[ (i1-t0)*(t-t1) + (j1-t1) ] = +inf
        for all (i1,j1) in L1-ball around (i,j) with radius r=clamp(|i-j|-2,0,nms),
        respecting bounds t0<=i1<t and t1<=j1<t.
        """

        device = d.device
        ii = ii.to(device=device, dtype=torch.long)
        jj = jj.to(device=device, dtype=torch.long)

        W = (t - t1)

        # radius per edge
        r = (ii - jj).abs()
        r = (r - 2).clamp(min=0, max=nms).to(torch.long)

        # precompute offsets for each radius (diamond / L1 ball)
        offsets = [None] * (nms + 1)
        offsets[0] = torch.tensor([[0, 0]], device=device, dtype=torch.long)
        for rv in range(1, nms + 1):
            # all integer (di,dj) with |di|+|dj| <= rv
            di = torch.arange(-rv, rv + 1, device=device)
            dj = torch.arange(-rv, rv + 1, device=device)
            DII, DJJ = torch.meshgrid(di, dj, indexing="ij")
            m = (DII.abs() + DJJ.abs()) <= rv
            offsets[rv] = torch.stack([DII[m], DJJ[m]], dim=1).to(torch.long)  # [K,2]

        # only loop over small number of radii
        for rv in range(0, nms + 1):
            sel = (r == rv)
            if not sel.any():
                continue

            base_i = ii[sel]  # [E]
            base_j = jj[sel]  # [E]
            off = offsets[rv] # [K,2]

            # broadcast to [E,K]
            i1 = base_i[:, None] + off[None, :, 0]
            j1 = base_j[:, None] + off[None, :, 1]

            # bounds
            m = (i1 >= t0) & (i1 < t) & (j1 >= t1) & (j1 < t)
            if not m.any():
                continue

            i1 = i1[m]
            j1 = j1[m]

            lin = (i1 - t0) * W + (j1 - t1)   # [N]
            d[lin] = np.inf  # duplicates are fine

        return d

    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.3, thresh=16.0, remove=False, backend=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        # Skip visual edges spanning a ride that has not folded yet: both ends still read the
        # departure-floor height (Sec. 3.3, first rule).
        _unfolded_rides = [(k0, k1) for k0, k1 in self.video.elev_ride_spans(unfolded_only=True) if k1 is not None]
        # The final bundle adjustment (backend=True) adds no edge to a keyframe of a FOLDED ride
        # (App. C): those frames see the static elevator interior, so a visual edge between them
        # reports no motion and a world-frame solve would flatten the folded rise (on
        # Office2 it pulled mid-ride keyframes 6 m up onto the arrival floor). With no edge
        # touching them their poses get no update and the floors stay where the fold put them.
        _in_ride = set()
        if backend:
            for k0, k1 in self.video.elev_folded_rides:
                _in_ride.update(range(int(k0), int(k1) + 1))
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx, indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_inac], 0)
        
        d = self.nms_invalidate_(d, ii1, jj1, t0, t1, t, nms)
        # Candidate selection below is scalar, per-pair work: one host copy here instead of a
        # device sync per candidate and a one-element kernel per cell.
        d = d.cpu()

        es = []
        for i in range(t0, t):
            for j in range(max(i-rad-1,0), i):
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf
                if i in _in_ride or j in _in_ride:
                    continue
                es.append((i,j))
                es.append((j,i))

        ix = torch.argsort(d)
        d_sorted = d[ix]
        stop = torch.searchsorted(d_sorted, torch.tensor(thresh, dtype=d.dtype), right=True)
        ix = ix[:stop]                        # only candidates within threshold
        # No suppression among the picked candidates themselves: ix is fixed before the loop
        # (the base system's behaviour).
        for k in ix.tolist():
            if len(es) > self.max_factors:
                break
            i, j = int(ii[k]), int(jj[k])
            _lo, _hi = (i, j) if i < j else (j, i)
            if any(_lo < k0 and _hi > k1 for k0, k1 in _unfolded_rides):
                continue                       # straddles a ride that has not folded yet
            if _lo in _in_ride or _hi in _in_ride:
                continue                       # folded ride: poses held fixed in the final BA
            # bidirectional
            es.append((i, j))
            es.append((j, i))

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)
