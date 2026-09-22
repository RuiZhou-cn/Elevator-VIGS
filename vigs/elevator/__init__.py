"""The elevator handling of Sec. 3, in the order the data flows through it:

* `detect`      -- WHEN (Sec. 3.4). Two layers that only ever produce a time window: the
                   semantic cue P_elev of Eq. (10) and the geometric cue rho_sp of Eq. (11),
                   scored per sampled frame and sustained into an ARMED window (run by the
                   motion filter), and inside that window an IMU-rate three-state FSM on the
                   open-loop vertical velocity v_z that detects the departure and the arrival
                   (run by the tracking frontend). Open-loop throughout; nothing here is ever
                   fed back into the estimator.
* `transport`   -- HOW FAR, and where it is written (Sec. 3.2, App. C.1). `TransportSolver`
                   re-reads the same IMU window under BA-converged attitudes and solves ONE
                   ride -- t_d, t_a, and the constant error c of a_z that brings u back to zero
                   in the arrival rest -- read-only. `seed_transport` (at the departure) and
                   `resolve_transport` (once the arrival rest closes the ride) are the ONLY
                   writers of the per-KF transport state (u, h) on the shared DepthVideo, and
                   `assemble_factors_17w` carries it through the in-loop BA at the 17-wide
                   state [pose6|vel3|bias6|u@15|h@16], with the departure and arrival
                   constraints of Eq. (7).
* `ride_manager`-- the frontend's ride lifecycle: pin k_0 at the departure, pin k_1 at the
                   arrival, hold the rise in h, and run the deferred fold of Eq. (9) once the
                   ride has left the sliding window.

The path is always live: no config says whether a ride is coming, the detector decides, and an
elevator-free sequence runs the same code with h = u = 0 pinned on every keyframe.
"""
