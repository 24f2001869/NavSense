package com.sih26168.idr.engine;

import java.util.ArrayList;
import java.util.List;

/**
 * Unified Smartphone Dead-Reckoning Navigation Engine.
 * Direct Java port of src/navigation/dead_reckoning_engine.py.
 */
public class DeadReckoningEngine {

    public enum NavigationState {
        ALIGNING,
        GNSS_LOCKED,
        DEAD_RECKONING,
        REACQUIRING
    }

    public static class EngineConfig {
        public double sigma_accel = 0.291;
        public double sigma_gyro = 0.015;
        public double gravity = 9.80665;
        public double dt_default = 0.1;

        public double sigma_speed = 0.60;
        public double huber_speed_k = 3.0;

        public double sigma_lat_0 = 0.50;
        public double nhc_gate_chi2 = 9.0;
        public double turn_rate_scale_deg_s = 5.0;

        public double mag_norm_tol = 0.08;
        public double mag_db_dt_tol = 5.0;
        public double sigma_compass_deg = 5.0;
        public double compass_gate_deg = 25.0;

        public double map_search_radius_m = 25.0;
        public double map_heading_gate_deg = 30.0;
        public double sigma_map_heading_deg = 5.0;
        public double map_update_interval_s = 1.0;

        public boolean enable_vert_nhc = true;
        public double sigma_vert = 0.50;
        public double turn_rate_vnhc_gate_deg_s = 3.0;
        public double turn_rate_compass_gate_deg_s = 3.0;
        public boolean enable_causal_speed_bias = false;
        public double max_speed_bias_ms = 1.5;

        public double zupt_sigma_vel = 0.05;
        public double causal_lookback_s = 30.0;
        public int min_straight_samples = 10;
    }

    public static class GNSSMeasurement {
        public final boolean valid;
        public final Vector3 pos_enu;
        public final Vector3 vel_enu;
        public final double accuracy_m;

        public GNSSMeasurement(boolean valid) {
            this(valid, new Vector3(), new Vector3(), 3.0);
        }

        public GNSSMeasurement(boolean valid, Vector3 pos_enu, Vector3 vel_enu, double accuracy_m) {
            this.valid = valid;
            this.pos_enu = pos_enu != null ? pos_enu : new Vector3();
            this.vel_enu = vel_enu != null ? vel_enu : new Vector3();
            this.accuracy_m = accuracy_m;
        }
    }

    public static class StepDiagnostics {
        public String nav_state;
        public double outage_duration_s;
        public double b_accel_x_applied;
        public double b_gyro_z_applied;
        public double b_speed_applied;
        public boolean nhc_active;
        public boolean vnhc_active;
        public boolean compass_active;
        public boolean map_active;
        public boolean zupt_active;
        public boolean gnss_active;
        public double gnss_nis;
        public double nhc_nis;
        public double vnhc_nis;
    }

    public static class NavigationTelemetry {
        public double time_s;
        public Vector3 pos_enu;
        public Vector3 vel_enu;
        public double heading_deg;
        public double pitch_deg;
        public double roll_deg;
        public double pos_sigma_m;
        public double heading_sigma_deg;
        public StepDiagnostics diagnostics;
    }

    public final EngineConfig cfg;
    public final Matrix R_vp;
    public final Vector3 ba_stat;
    public final Vector3 bg_stat;
    public final double baseline_B;

    public ESKF3D eskf;
    public MapConstraintManager map_mgr;

    public NavigationState nav_state = NavigationState.GNSS_LOCKED;
    public double time_now_s = 0.0;
    public double outage_duration_s = 0.0;
    public double last_map_update_time = -1000.0;

    private final int lookback_pts;
    private final List<Double> buf_time = new ArrayList<>();
    private final List<Double> buf_phone_ax = new ArrayList<>();
    private final List<Double> buf_gyro_z = new ArrayList<>();
    private final List<Double> buf_speed = new ArrayList<>();
    private final List<Double> buf_turn_rate = new ArrayList<>();
    private final List<Double> buf_ml_speed = new ArrayList<>();

    public double b_accel_x = 0.0;
    public double b_gyro_z = 0.0;
    public double b_speed = 0.0;

    private Vector3 last_mag = null;

    public DeadReckoningEngine(
        EngineConfig config,
        RoadNetworkIndex roadIndex,
        Vector3 initPosEnu,
        Vector3 initVelEnu,
        double initHeadingDeg,
        Matrix R_vp,
        Vector3 ba_stat,
        Vector3 bg_stat,
        double baseline_mag_uT
    ) {
        this.cfg = config != null ? config : new EngineConfig();
        this.R_vp = R_vp != null ? R_vp.copy() : Matrix.identity(3);
        this.ba_stat = ba_stat != null ? ba_stat.copy() : new Vector3();
        this.bg_stat = bg_stat != null ? bg_stat.copy() : new Vector3();
        this.baseline_B = baseline_mag_uT;

        this.lookback_pts = (int) (this.cfg.causal_lookback_s / this.cfg.dt_default);

        this.eskf = new ESKF3D(
            initPosEnu,
            initVelEnu,
            initHeadingDeg,
            0.0,
            0.0,
            Matrix.identity(3),
            this.ba_stat,
            this.bg_stat,
            this.cfg.sigma_accel,
            this.cfg.sigma_gyro,
            0.001,
            0.0001,
            1.5,
            3.0,
            0.1,
            0.2,
            this.cfg.gravity
        );

        if (roadIndex != null) {
            this.map_mgr = new MapConstraintManager(
                roadIndex,
                this.cfg.map_search_radius_m,
                this.cfg.map_heading_gate_deg,
                this.cfg.sigma_map_heading_deg,
                this.cfg.map_update_interval_s
            );
        }
    }

    public NavigationTelemetry step(
        Vector3 accel_raw,
        Vector3 gyro_raw,
        double speed_est,
        double dt,
        Vector3 mag_raw,
        GNSSMeasurement gnss,
        Double psi_mag_cal_deg,
        Double db_dt
    ) {
        time_now_s += dt;

        // 1. Transform raw phone IMU to vehicle body frame
        double[] acc_v_arr = R_vp.multiply(new double[]{accel_raw.x, accel_raw.y, accel_raw.z});
        double[] gyro_v_arr = R_vp.multiply(new double[]{gyro_raw.x, gyro_raw.y, gyro_raw.z});
        Vector3 acc_v = new Vector3(acc_v_arr[0], acc_v_arr[1], acc_v_arr[2]);
        Vector3 gyro_v = new Vector3(gyro_v_arr[0], gyro_v_arr[1], gyro_v_arr[2]);

        double ax_raw = acc_v.x, ay_raw = acc_v.y, az_raw = acc_v.z;
        double gx_raw = gyro_v.x, gy_raw = gyro_v.y, gz_raw = gyro_v.z;

        double turn_rate_deg_s = Math.abs(Math.toDegrees(gz_raw));

        // 2. State machine transitions
        boolean is_outage = (gnss == null || !gnss.valid);
        if (!is_outage) {
            if (nav_state == NavigationState.DEAD_RECKONING) {
                nav_state = NavigationState.REACQUIRING;
            } else {
                nav_state = NavigationState.GNSS_LOCKED;
            }
            outage_duration_s = 0.0;
        } else {
            nav_state = NavigationState.DEAD_RECKONING;
            outage_duration_s += dt;
        }

        // 3. Pre-Outage Causal Bias Tracking (Only during valid GNSS)
        if (!is_outage && gnss != null) {
            double v_gnss = Math.sqrt(gnss.vel_enu.x * gnss.vel_enu.x + gnss.vel_enu.y * gnss.vel_enu.y);
            buf_time.add(time_now_s);
            buf_phone_ax.add(ax_raw);
            buf_gyro_z.add(gz_raw);
            buf_speed.add(v_gnss);
            buf_turn_rate.add(turn_rate_deg_s);
            buf_ml_speed.add(speed_est);

            if (buf_time.size() > lookback_pts) {
                buf_time.remove(0);
                buf_phone_ax.remove(0);
                buf_gyro_z.remove(0);
                buf_speed.remove(0);
                buf_turn_rate.remove(0);
                buf_ml_speed.remove(0);
            }

            if (buf_speed.size() >= cfg.min_straight_samples) {
                int N = buf_speed.size();
                List<Integer> st_indices = new ArrayList<>();
                for (int i = 0; i < N; i++) {
                    if (buf_speed.get(i) >= 3.0 && buf_turn_rate.get(i) < 2.0) {
                        st_indices.add(i);
                    }
                }

                if (st_indices.size() >= cfg.min_straight_samples) {
                    // Compute gradient of speeds
                    double[] a_gnss = new double[N];
                    for (int i = 0; i < N; i++) {
                        if (i == 0) {
                            a_gnss[i] = (buf_speed.get(1) - buf_speed.get(0)) / dt;
                        } else if (i == N - 1) {
                            a_gnss[i] = (buf_speed.get(N - 1) - buf_speed.get(N - 2)) / dt;
                        } else {
                            a_gnss[i] = (buf_speed.get(i + 1) - buf_speed.get(i - 1)) / (2.0 * dt);
                        }
                    }

                    double sum_accel_diff = 0.0;
                    double sum_gyro_z = 0.0;
                    double sum_speed_diff = 0.0;
                    for (int idx : st_indices) {
                        sum_accel_diff += (buf_phone_ax.get(idx) - a_gnss[idx]);
                        sum_gyro_z += buf_gyro_z.get(idx);
                        sum_speed_diff += (buf_speed.get(idx) - buf_ml_speed.get(idx));
                    }
                    int M = st_indices.size();
                    this.b_accel_x = sum_accel_diff / M;
                    this.b_gyro_z = sum_gyro_z / M;
                    double raw_b_spd = sum_speed_diff / M;
                    this.b_speed = Math.max(-cfg.max_speed_bias_ms, Math.min(cfg.max_speed_bias_ms, raw_b_spd));
                }
            }
        }

        // 4. Strapdown Inertial Prediction (accel_raw and gyro_raw in phone frame)
        eskf.predict(accel_raw.x, accel_raw.y, accel_raw.z, gyro_raw.x, gyro_raw.y, gyro_raw.z, dt);

        // 5. Gravity Leveling (tilt stabilization)
        eskf.attitude.updateGravity(accel_raw.x, accel_raw.y, accel_raw.z, 0.02, 1.5);

        // 6. Diagnostics initialization
        StepDiagnostics diag = new StepDiagnostics();
        diag.nav_state = nav_state.name();
        diag.outage_duration_s = outage_duration_s;
        diag.b_accel_x_applied = is_outage ? b_accel_x : 0.0;
        diag.b_gyro_z_applied = is_outage ? b_gyro_z : 0.0;
        diag.b_speed_applied = is_outage ? b_speed : 0.0;

        // 7. Measurement Updates
        if (!is_outage && gnss != null) {
            if (nav_state == NavigationState.REACQUIRING) {
                eskf.reacquireGnss(gnss.pos_enu, gnss.vel_enu, gnss.accuracy_m);
            } else {
                eskf.updateGnss(gnss.pos_enu, gnss.vel_enu);
            }
            diag.gnss_active = true;
        } else {
            // DEAD RECKONING MODE

            // A. Forward Velocity Update: TCN-kin (1 Hz or whenever speed_est provided)
            double spd_use = speed_est;
            if (cfg.enable_causal_speed_bias) {
                spd_use = Math.max(0.0, speed_est + b_speed);
            }
            if (spd_use > 0.0) {
                double nis_x = eskf.updateTcnSpeed(spd_use, cfg.sigma_speed);
                diag.nhc_nis = nis_x;
            }

            // B. Decoupled Pure-Velocity Lateral Damping (Variant V1: 10 Hz)
            double nis_y = eskf.updateDecoupledLateralVelocityDamping(cfg.sigma_lat_0);
            diag.nhc_active = true;

            // C. Downstream OSM Map Snapping (purely telemetry display, ZERO closed loop filter twist)
            if (map_mgr != null && (time_now_s - last_map_update_time >= cfg.map_update_interval_s - 1e-4)) {
                last_map_update_time = time_now_s;
                RoadNetworkIndex.Candidate cand = map_mgr.selectCandidate(eskf.pos_n.x, eskf.pos_n.y, eskf.attitude.getYawDeg());
                if (cand != null) {
                    diag.map_active = true;
                }
            }
        }

        // 7. Telemetry packaging
        NavigationTelemetry tel = new NavigationTelemetry();
        tel.time_s = time_now_s;
        tel.pos_enu = eskf.pos_n.copy();
        tel.vel_enu = eskf.vel_n.copy();
        tel.heading_deg = eskf.attitude.getYawDeg();
        tel.pitch_deg = eskf.attitude.getPitchDeg();
        tel.roll_deg = eskf.attitude.getRollDeg();

        double p00 = eskf.P.get(0, 0);
        double p11 = eskf.P.get(1, 1);
        tel.pos_sigma_m = Math.sqrt(p00 + p11);
        tel.heading_sigma_deg = Math.toDegrees(Math.sqrt(eskf.P.get(8, 8)));
        tel.diagnostics = diag;

        return tel;
    }

    private void updateForwardVelocity(double v_forward) {
        Matrix C_v_n = eskf.attitude.getDcm();
        Matrix C_n_v = C_v_n.transpose();
        double[] vel_v = C_n_v.multiply(new double[]{eskf.vel_n.x, eskf.vel_n.y, eskf.vel_n.z});

        double r_fwd = v_forward - vel_v[0];
        Matrix H_fwd = new Matrix(1, 15);
        for (int i = 0; i < 3; i++) {
            H_fwd.set(0, 3 + i, C_n_v.get(0, i));
        }
        H_fwd.set(0, 6, 0.0);
        H_fwd.set(0, 7, -vel_v[2]);
        H_fwd.set(0, 8,  vel_v[1]);

        double R_1d = cfg.sigma_speed * cfg.sigma_speed;
        Matrix S_mat = H_fwd.multiply(eskf.P).multiply(H_fwd.transpose());
        double S = S_mat.get(0, 0) + R_1d;
        double nis = (r_fwd * r_fwd) / Math.max(S, 1e-6);

        double R_eff = R_1d;
        double nis_gate_1d = 6.635;
        if (nis > nis_gate_1d) {
            double scale = Math.sqrt(nis / nis_gate_1d);
            R_eff = R_1d * (scale * scale);
            S = S_mat.get(0, 0) + R_eff;
        }

        Matrix P_Ht = eskf.P.multiply(H_fwd.transpose()); // 15x1
        Matrix K = P_Ht.scale(1.0 / Math.max(S, 1e-6));

        // STRICT ATTITUDE & BIAS FREEZE
        for (int i = 6; i < 15; i++) {
            K.set(i, 0, 0.0);
        }

        double[] delta_x = new double[15];
        for (int i = 0; i < 15; i++) {
            delta_x[i] = K.get(i, 0) * r_fwd;
        }

        eskf.pos_n.x += delta_x[0];
        eskf.pos_n.y += delta_x[1];
        eskf.pos_n.z += delta_x[2];

        eskf.vel_n.x += delta_x[3];
        eskf.vel_n.y += delta_x[4];
        eskf.vel_n.z += delta_x[5];

        Matrix IKH = Matrix.identity(15).sub(K.multiply(H_fwd));
        Matrix R_mat = new Matrix(1, 1);
        R_mat.set(0, 0, R_eff);
        Matrix P_new = IKH.multiply(eskf.P).multiply(IKH.transpose()).add(K.multiply(R_mat).multiply(K.transpose()));
        P_new.symmetrize();
        eskf.P = P_new;
    }

    private void updateLateralNhc(double v_fwd_est, double tr_deg_s, StepDiagnostics diag) {
        Matrix C_v_n = eskf.attitude.getDcm();
        Matrix C_n_v = C_v_n.transpose();
        double[] vel_v = C_n_v.multiply(new double[]{eskf.vel_n.x, eskf.vel_n.y, eskf.vel_n.z});

        double r_lat = -vel_v[1];
        double scale_tr = tr_deg_s / cfg.turn_rate_scale_deg_s;
        double sigma_lat = cfg.sigma_lat_0 * (1.0 + scale_tr * scale_tr);
        double R = sigma_lat * sigma_lat;

        Matrix H = new Matrix(1, 15);
        for (int i = 0; i < 3; i++) {
            H.set(0, 3 + i, C_n_v.get(1, i));
        }
        double vx_use = Math.abs(vel_v[0]) > 0.5 ? vel_v[0] : v_fwd_est;
        H.set(0, 6, vel_v[2]);
        H.set(0, 7, 0.0);
        H.set(0, 8, -vx_use);

        Matrix S_mat = H.multiply(eskf.P).multiply(H.transpose());
        double S = S_mat.get(0, 0) + R;
        double S_inv = 1.0 / S;
        double nis = (r_lat * r_lat) * S_inv;

        if (nis > cfg.nhc_gate_chi2 || Math.abs(r_lat) > 2.5) {
            return;
        }

        double k_huber = 2.0;
        double gamma = (nis > (k_huber * k_huber)) ? Math.min(1.0, k_huber / Math.sqrt(Math.max(nis, 1e-12))) : 1.0;

        Matrix P_Ht = eskf.P.multiply(H.transpose());
        Matrix K = P_Ht.scale(S_inv * gamma);

        // Strict Bias Freeze
        for (int i = 9; i < 15; i++) {
            K.set(i, 0, 0.0);
        }

        double[] dx = new double[15];
        for (int i = 0; i < 15; i++) {
            dx[i] = K.get(i, 0) * r_lat;
        }

        eskf.pos_n.x += dx[0];
        eskf.pos_n.y += dx[1];
        eskf.pos_n.z += dx[2];

        eskf.vel_n.x += dx[3];
        eskf.vel_n.y += dx[4];
        eskf.vel_n.z += dx[5];

        Vector3 dtheta = new Vector3(dx[6], dx[7], dx[8]);
        Quaternion dq = Quaternion.fromRotationVector(dtheta);
        eskf.attitude.q_nv = Quaternion.multiply(eskf.attitude.q_nv, dq);
        eskf.attitude.q_nv.normalize();

        Matrix IKH = Matrix.identity(15).sub(K.multiply(H));
        Matrix R_mat = new Matrix(1, 1);
        R_mat.set(0, 0, R);
        Matrix P_new = IKH.multiply(eskf.P).multiply(IKH.transpose()).add(K.multiply(R_mat).multiply(K.transpose()));
        P_new.symmetrize();
        eskf.P = P_new;

        diag.nhc_active = true;
        diag.nhc_nis = nis;
    }

    private void updateVerticalNhc(double turn_rate_deg_s, StepDiagnostics diag) {
        if (turn_rate_deg_s > cfg.turn_rate_vnhc_gate_deg_s) {
            return;
        }

        Matrix C_v_n = eskf.attitude.getDcm();
        Matrix C_n_v = C_v_n.transpose();
        double[] vel_v = C_n_v.multiply(new double[]{eskf.vel_n.x, eskf.vel_n.y, eskf.vel_n.z});

        double r_vert = -vel_v[2];
        double R = cfg.sigma_vert * cfg.sigma_vert;

        Matrix H = new Matrix(1, 15);
        for (int i = 0; i < 3; i++) {
            H.set(0, 3 + i, C_n_v.get(2, i));
        }

        Matrix S_mat = H.multiply(eskf.P).multiply(H.transpose());
        double S = S_mat.get(0, 0) + R;
        double S_inv = 1.0 / S;
        double nis = (r_vert * r_vert) * S_inv;

        if (nis > cfg.nhc_gate_chi2 || Math.abs(r_vert) > 3.0) {
            return;
        }

        double k_huber = 2.0;
        double gamma = (nis > (k_huber * k_huber)) ? Math.min(1.0, k_huber / Math.sqrt(Math.max(nis, 1e-12))) : 1.0;

        Matrix P_Ht = eskf.P.multiply(H.transpose());
        Matrix K = P_Ht.scale(S_inv * gamma);

        // Strict Position, Attitude & Bias Freeze
        for (int i = 0; i < 3; i++) {
            K.set(i, 0, 0.0);
        }
        for (int i = 6; i < 15; i++) {
            K.set(i, 0, 0.0);
        }

        double[] dx = new double[15];
        for (int i = 0; i < 15; i++) {
            dx[i] = K.get(i, 0) * r_vert;
        }

        eskf.vel_n.x += dx[3];
        eskf.vel_n.y += dx[4];
        eskf.vel_n.z += dx[5];

        Matrix IKH = Matrix.identity(15).sub(K.multiply(H));
        Matrix R_mat = new Matrix(1, 1);
        R_mat.set(0, 0, R);
        Matrix P_new = IKH.multiply(eskf.P).multiply(IKH.transpose()).add(K.multiply(R_mat).multiply(K.transpose()));
        P_new.symmetrize();
        eskf.P = P_new;

        diag.vnhc_active = true;
        diag.vnhc_nis = nis;
    }

    private void updateSelectiveCompass(
        Vector3 mag_raw,
        double psi_mag_cal_deg,
        double dt,
        double turn_rate_deg_s,
        Double db_dt_override,
        StepDiagnostics diag
    ) {
        if (turn_rate_deg_s > cfg.turn_rate_compass_gate_deg_s) {
            return;
        }

        double mag_norm = mag_raw.norm();
        double db_dt = 0.0;
        if (db_dt_override != null) {
            db_dt = db_dt_override;
        } else if (last_mag != null && dt > 0.0) {
            db_dt = mag_raw.sub(last_mag).norm() / dt;
        }
        this.last_mag = mag_raw.copy();

        double norm_diff = Math.abs(mag_norm - baseline_B) / baseline_B;
        if (norm_diff > cfg.mag_norm_tol || db_dt > cfg.mag_db_dt_tol) {
            return;
        }

        double cur_yaw = eskf.attitude.getYawDeg();
        double yaw_diff = Math.abs(((psi_mag_cal_deg - cur_yaw + 180.0) % 360.0 + 360.0) % 360.0 - 180.0);
        if (yaw_diff > cfg.compass_gate_deg) {
            return;
        }

        double r_psi = Math.toRadians(((psi_mag_cal_deg - cur_yaw + 180.0) % 360.0 + 360.0) % 360.0 - 180.0);
        Matrix C_v_n = eskf.attitude.getDcm();
        Matrix H = new Matrix(1, 15);
        H.set(0, 6, -C_v_n.get(2, 0));
        H.set(0, 7, -C_v_n.get(2, 1));
        H.set(0, 8, -C_v_n.get(2, 2));

        double sig_rad = Math.toRadians(cfg.sigma_compass_deg);
        double R = sig_rad * sig_rad;

        Matrix S_mat = H.multiply(eskf.P).multiply(H.transpose());
        double S = S_mat.get(0, 0) + R;
        double S_inv = 1.0 / S;

        Matrix P_Ht = eskf.P.multiply(H.transpose());
        Matrix K = P_Ht.scale(S_inv);

        // Strict Bias Freeze
        for (int i = 9; i < 15; i++) {
            K.set(i, 0, 0.0);
        }

        double[] dx = new double[15];
        for (int i = 0; i < 15; i++) {
            dx[i] = K.get(i, 0) * r_psi;
        }

        eskf.pos_n.x += dx[0];
        eskf.pos_n.y += dx[1];
        eskf.pos_n.z += dx[2];

        eskf.vel_n.x += dx[3];
        eskf.vel_n.y += dx[4];
        eskf.vel_n.z += dx[5];

        Vector3 dtheta = new Vector3(dx[6], dx[7], dx[8]);
        Quaternion dq = Quaternion.fromRotationVector(dtheta);
        eskf.attitude.q_nv = Quaternion.multiply(eskf.attitude.q_nv, dq);
        eskf.attitude.q_nv.normalize();

        Matrix IKH = Matrix.identity(15).sub(K.multiply(H));
        Matrix R_mat = new Matrix(1, 1);
        R_mat.set(0, 0, R);
        Matrix P_new = IKH.multiply(eskf.P).multiply(IKH.transpose()).add(K.multiply(R_mat).multiply(K.transpose()));
        P_new.symmetrize();
        eskf.P = P_new;

        diag.compass_active = true;
    }

    private void updateRoadHeading(StepDiagnostics diag) {
        if (map_mgr == null) return;

        double cur_yaw = eskf.attitude.getYawDeg();
        RoadNetworkIndex.Candidate cand = map_mgr.selectCandidate(eskf.pos_n.x, eskf.pos_n.y, cur_yaw);
        if (cand == null) return;

        MapConstraintManager.HeadingInnovation innov = map_mgr.computeHeadingInnovation(cur_yaw, cand);
        double R_psi = map_mgr.sigmaPsi * map_mgr.sigmaPsi;

        Matrix S_mat = innov.H_psi.multiply(eskf.P).multiply(innov.H_psi.transpose());
        double S = S_mat.get(0, 0) + R_psi;
        double S_inv = 1.0 / S;
        double nis = (innov.y_psi * innov.y_psi) * S_inv;

        if (nis > map_mgr.nisGate1dof) {
            return;
        }

        Matrix P_Ht = eskf.P.multiply(innov.H_psi.transpose());
        Matrix K = P_Ht.scale(S_inv);

        // Strict Bias Freeze
        for (int i = 9; i < 15; i++) {
            K.set(i, 0, 0.0);
        }

        double[] dx = new double[15];
        for (int i = 0; i < 15; i++) {
            dx[i] = K.get(i, 0) * innov.y_psi;
        }

        eskf.pos_n.x += dx[0];
        eskf.pos_n.y += dx[1];
        eskf.pos_n.z += dx[2];

        eskf.vel_n.x += dx[3];
        eskf.vel_n.y += dx[4];
        eskf.vel_n.z += dx[5];

        Vector3 dtheta = new Vector3(dx[6], dx[7], dx[8]);
        Quaternion dq = Quaternion.fromRotationVector(dtheta);
        eskf.attitude.q_nv = Quaternion.multiply(eskf.attitude.q_nv, dq);
        eskf.attitude.q_nv.normalize();

        Matrix IKH = Matrix.identity(15).sub(K.multiply(innov.H_psi));
        Matrix R_mat = new Matrix(1, 1);
        R_mat.set(0, 0, R_psi);
        Matrix P_new = IKH.multiply(eskf.P).multiply(IKH.transpose()).add(K.multiply(R_mat).multiply(K.transpose()));
        P_new.symmetrize();
        eskf.P = P_new;

        last_map_update_time = time_now_s;
        diag.map_active = true;
    }

    public double getPitchDeg() {
        return eskf.attitude.getPitchDeg();
    }

    public double getRollDeg() {
        return eskf.attitude.getRollDeg();
    }

    public double getHeadingDeg() {
        return eskf.attitude.getYawDeg();
    }

    public void setHeadingDeg(double headingDeg) {
        if (eskf != null && eskf.attitude != null) {
            eskf.attitude.setHeadingDeg(headingDeg);
        }
    }

    public void alignHeading(double targetHeadingDeg, double alpha) {
        if (eskf != null && eskf.attitude != null) {
            eskf.attitude.alignHeading(targetHeadingDeg, alpha);
        }
    }
}
