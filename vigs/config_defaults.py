"""Algorithm hyper-parameters shared by every dataset.

A config/<dataset>.yaml holds only what differs between datasets -- the sensor (IMU noise
model, camera-IMU extrinsics, time units and offset), the IMU-init keyframe and loop closure.
load_config (util/utils.py) starts from a copy of DEFAULTS and merges the yaml over it, so a key
given in the yaml overrides its default and an upstream-style full yaml still loads unchanged.
Section and key names are upstream VIGS-SLAM's; `Elevator` is this work's.
"""

DEFAULTS = dict(
    Dataset=dict(
        pcd_downsample_init=32,     # pixel stride of the Gaussians seeded from the first keyframe
        adaptive_pointsize=True,    # scale point_size by the keyframe's median depth
        point_size=0.05,
    ),
    IMU=dict(
        init_bg=[0.0, 0.0, 0.0],    # gyro / accel bias before the inertial init
        init_ba=[0.0, 0.0, 0.0],
    ),
    Tracking=dict(
        disable_mono=False,
        motion_filter=dict(
            init_thresh=4.0,
            thresh=2.4,             # motion required before a new keyframe is considered
        ),
        frontend=dict(
            keyframe_thresh=4.0,    # threshold to create a new keyframe
            keep_kf_once_deleted=False,
            frontend_thresh=16.0,   # add edges between frames within this distance
            frontend_window=25,     # optimization window
            frontend_radius=2,      # force edges between frames within radius
            frontend_nms=1,         # non-maximal suppression of edges
            mono_depth_alpha=0.01,
        ),
        backend=dict(
            backend_thresh=22.0,    # loop-closure edge distance (larger = more edges)
            backend_radius=2,       # connect neighbors within radius (larger = more edges)
            backend_nms=3,          # non-maximal suppression of edges (smaller = more edges)
            covis_thresh=0.3,
        ),
        pgba=dict(
            pgba_thresh=22.0,       # loop-closure edge distance (larger = more edges)
        ),
    ),
    Training=dict(
        parallel=False,             # run the Gaussian mapper in its own thread
        queue_size=2,               # keyframes buffered for that thread
        gs_defer_until_kf=0,        # packets whose newest keyframe is below this index are not
                                    # mapped (0 = map from the first keyframe; FAST-LIVO2: 9)
        alpha=0.95,                 # RGB vs depth weight of the RGB-D mapping loss
        init_itr_num=1050,
        gaussian_update_every=150,
        gaussian_update_offset=50,
        gaussian_th=0.7,
        gaussian_extent=1.0,
        gaussian_reset=2000000001,
        size_threshold=20,
        window_size=10,
        rgb_boundary_threshold=0.01,
        lambda_dnormal=0.5,
        compensate_exposure=True,
    ),
    opt_params=dict(
        pose_lr=0.0001,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_max_steps=26000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.001,
        rotation_lr=0.001,
        exposure_lr=0.01,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densify_grad_threshold=0.0002,
    ),
    Elevator=dict(
        # The elevator path is always live -- nothing tells a run whether a ride is coming, so an
        # elevator-free sequence runs the same code with the same numbers.
        kf_dt=1.0,                  # s, keyframe gap forced while the armed window is open (3 s
                                    # otherwise): sparse IMU edges span the whole ride and lose the rise
        detect=dict(
            # Armed window (elevator/detect.py::ArmedWindowFSM): the semantic cue P_elev and the
            # geometric cue rho_sp per sampled frame -> causal ARMED / DISARMED. The armed window
            # gates the ride FSM that detects the departure and the arrival.
            armed=dict(
                sample_dt=0.29,         # s, min gap between scored frames; the EMA and
                                        # hysteresis below are tuned at this rate
                ema_tau=1.0,            # s, smoothing on P_elev and rho_sp
                max_rho_sp=1.0,         # smoothed depth spread rho_sp <= this = "enclosed"
                                        # (gates arming only)
                p_arm=0.5,              # hysteresis on the smoothed P_elev
                p_disarm=0.3,
                p_depth_wake=0.35,      # smoothed P_elev above which the ~10 ms geometric cue is
                                        # worth running
                sustain_in=1.0,         # s, how long the arm / disarm condition must hold
                sustain_out=2.0,
            ),
            # Ride FSM (elevator/detect.py::RideDetector), IMU rate: departure and arrival.
            ride=dict(
                # signal chain: a_z (world-vertical acceleration) -> a_lp (double EMA)
                #               -> a_z - c (minus the constant error c, a trailing median)
                lp_tau=0.30,        # s, EMA time constant, applied twice: ~2 Hz gait is attenuated ~15x
                                    # while a 1-1.5 s elevator ramp passes almost unattenuated
                c_win=15.0,         # s, trailing-median window of a_lp = the constant error c of a_z
                warmup=8.0,         # s, no decisions until c has this much history
                # body-rest predicate: answers "is the BODY still", never "is this an elevator"
                rest_acc_max=0.15,  # m/s^2, |a_z - c| below this counts as quiet
                rest_dwell=0.3,     # s, quiet must last this long before the body counts as at rest
                # Rest pins v_z to 0 (still body + still elevator => v_z is exactly 0). A gentle
                # elevator ramp looks like rest too, so a sustained push suspends the pin:
                ramp_acc=0.04,      # m/s^2, 1 s-mean |a_z - c| at or above this = sustained push
                ramp_max=6.0,       # s, longest continuous suspension
                rest_clear=2.0,     # s, rest with no push after which v_z is zeroed regardless of
                                    # |v_z|; before that only |v_z| <= depart_v is pinned, so a real
                                    # ride's v_z survives
                # departure criterion, ARMED -> RIDING. Velocity domain: the robot's own vertical
                # motion is zero-mean, the elevator adds a DC offset.
                depart_v=0.40,      # m/s, |v_z| must reach this ...
                depart_win=1.0,     # s, ... and hold it without dipping. A floor against spurious
                                    # departures only; elevator-vs-stairs is the armed window's job.
                # arrival criterion, RIDING -> ARMED
                arrive_v=0.25,      # m/s, zero band on the 0.5 s-mean of v_z (wide enough to admit
                                    # the drift a real ride accumulates under a frozen c)
                arrive_dwell=0.4,   # s, sustained in-band before the arrival is called
                redepart_min=3.0,   # s, v_z is HELD at 0 this long after an arrival: an elevator
                                    # cannot depart again before its doors cycle, and the decel tail
                                    # + post-arrival motion phantom ~0.8 m/s
                # leaving the elevator, ARMED -> DISARMED: the only transition that ends a visit
                leave_dist=2.0,     # m, net horizontal displacement from the last arrival (motion
                                    # inside the elevator stays under ~1 m)
                leave_dwell=1.0,    # s, sustained -- also rides out a single-KF pose jump
                min_rise_m=2.0,     # m, a ride whose rise stays under this is discarded -- a floor
                                    # is >= 3.35 m on every set
                depart_dh_max=0.5,  # m, sideways displacement over the departure window above
                                    # which a departure is refused -- a rider stands still as the
                                    # elevator leaves; a drone or a walker does not. Judged once,
                                    # before the ride is committed: during a ride the horizontal
                                    # estimate wanders metres on real rides (detect.py docstring)
                # endpoints
                backdate_max=4.0,   # s, cap on moving the departure stamp back to where v_z left
                                    # the zero band
                replay_win=30.0,    # s, how far back a late arming edge may replay buffered IMU; a
                                    # glass elevator can arm ~17 s after the departure
                replay_prewin=2.0,  # s, minimum in-band stretch the replay must see before the
                                    # crossing it commits to; guards against a phantom departure
                                    # from a c the trailing median has not relearned yet
            ),
        ),
    ),
)
