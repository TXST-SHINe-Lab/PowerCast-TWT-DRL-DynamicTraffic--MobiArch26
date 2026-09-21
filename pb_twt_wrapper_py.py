#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
pb-twt-wrapper-py.py - Python-Side TWT Wrapper for NS-3 Simulation

This wrapper handles ALL NS-3/NS3-AI details and exposes a simple Python API
for controllers (RL agents, heuristics, etc.) that only see Python dicts.

Author: Ahmed Maksud <amaks002@ucr.edu>
Lab: SHINE Lab, Texas State University

Usage:
    from pb_twt_wrapper_py import TWTWrapper, create_default_action  # File: pb-twt-wrapper-py.py

    # 1. Create wrapper
    wrapper = TWTWrapper(verbose=True)

    # 2. Start simulation (returns initial env observation)
    env = wrapper.reset(seed=9000)
    print(f"Started with {env['num_sta']} STAs")

    # 3. Control loop
    while True:
        # Your controller decides action based on env
        # env['sta_observations'] contains per-STA metrics
        action = create_default_action(env['num_sta'], num_groups=2)

        # Send action, receive next env
        env, done = wrapper.step(action)

        if done:
            break

    # 4. Cleanup
    wrapper.close()
    # Check data-log/ for py-wrapper-env-*.csv and py-wrapper-action-*.csv
"""

import os
import sys
import csv
import traceback
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any

# Add paths for Python bindings
_script_dir = os.path.dirname(os.path.abspath(__file__))
# NS-3 root: twt/ -> examples/ -> ai/ -> contrib/ -> ns-3.44/
_ns3_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, "..", "..", "..", "python_utils"))
# the .so is also deployed under cmake-cache/ in this example's own dir; derive the dir name (not hardcoded) so a clone under any directory name still resolves.
_ex_name = os.path.basename(_script_dir)
sys.path.insert(
    0, os.path.join(_ns3_root, "cmake-cache", "contrib", "ai", "examples", _ex_name)
)


class TWTWrapper:
    """
    Python-side wrapper for NS-3 TWT simulation.

    Handles:
    - NS3-AI initialization and communication
    - C++ struct ↔ Python dict conversion
    - CSV logging of all env/action data
    - Simulation lifecycle management

    Exposes simple API:
    - reset(seed) → env_dict
    - step(action_dict) → (env_dict, done)
    - close()
    """

    # Field definitions for StaRealisticMetrics (from pb-twt-core.h)
    REALISTIC_FIELDS = [
        "sta_id",
        "is_active",
        "bsr_queue_ac_be",
        "bsr_queue_ac_bk",
        "bsr_queue_ac_vi",
        "bsr_queue_ac_vo",
        "bsr_scaling_factor",
        "rx_fragment_count",
        "fcs_error_count",
        "rcpi",
        "rsni",
        "rssi_dbm",
        "snr_db",
        "link_margin_db",
        "last_rx_frame_type",
        "last_rx_frame_subtype",
        "last_rx_mcs",
        "last_rx_nss",
        "channel_width_mhz",
        "guard_interval_ns",
        "power_mgmt_bit",
        "last_rx_timestamp_us",
        "bytes_received_at_ap",
        "packets_received_at_ap",
        "bytes_received_ac_vo",
        "bytes_received_ac_vi",
        "bytes_received_ac_be",
        "bytes_received_ac_bk",
        "packets_received_ac_vo",
        "packets_received_ac_vi",
        "packets_received_ac_be",
        "packets_received_ac_bk",
        "airtime_used_us",
        # QoEH-realistic per-SP UL accounting + AP DL accounting (cumulative).
        "bytes_rx_at_ap_in_sp",
        "packets_rx_at_ap_in_sp",
        "sp_completed_count",
        "sp_with_demand_count",
        "sp_with_starvation_count",
        "dl_bytes_to_sta",
        "dl_duration_us_to_sta",
        "dl_unicast_count_to_sta",
        "user_priority",
        "observation_time_ms",
        "observation_sequence_num",
    ]

    # Field definitions for StaOracleMetrics (from pb-twt-core.h)
    ORACLE_FIELDS = [
        # Device characteristics — moved from realistic 2026-06-07 (signal audit):
        # not AP-observable in this sim (TSPEC/TPC signaling not modeled; class blind).
        "device_class",
        "nominal_msdu_size",
        "mean_data_rate_kbps",
        "delay_bound_ms",
        "tx_power_dbm",
        "tx_fragment_count",
        "tx_failed_count",
        "tx_retry_count",
        "ack_failure_count",
        "total_energy_consumed_mj",
        "current_times_time_ma_ms",
        "awake_time_ms",
        "sleep_time_ms",
        "duty_cycle",
        "packets_generated",
        "packets_enqueued",
        "bytes_generated",
        "mpdu_drops_expired",
        "mpdu_drops_queue_full",
        "psdu_response_timeouts",
        "packets_transmitted",
        "bytes_transmitted",
        "ampdu_count",
        "ampdu_mpdus_total",
        "ampdu_bytes_total",
        "queue_size_packets",
        "queue_size_bytes",
        "queue_max_size",
        "avg_latency_ms",
        "queue_delay_sum_ms",
        "queue_delay_count",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        # PowerCast energy oracle (REHD-only; 0 for non-REHD STAs).
        # Reward-only — NEVER exposed to the agent observation.
        # Needed by the QoEH reward.
        "vcap_v",
        "vcap_max_v",
        "e_nominal_j",
        "e_sunk_j",
        "e_avail_j",
        "harvested_total_j",
        "consumed_total_j",
        "t_active_us",
        "unpowered_tx_events",
        "output_enabled",
        "harvester_phase",
        "est_rf_energy_delivered_j",
    ]

    def __init__(
        self,
        ns3_root: str = None,
        log_dir: str = "data-log",
        enable_logging: bool = True,
        verbose: bool = True,
        quiet_ns3: bool = True,
        disable_ns3_traces: bool = True,
        n_stations: int = None,
        n_rehd: int = None,
        traffic_scale: float = None,
    ):
        """
        Initialize TWT Wrapper.

        Args:
            ns3_root: Absolute path to NS-3 root directory (where ./ns3 lives).
                      If None, auto-computed from this module's location.
            log_dir: Directory for CSV log files (can be relative or absolute)
            enable_logging: Whether to log env/action to CSV
            verbose: Whether to print progress to terminal
            quiet_ns3: If True, pass --quietMode=true to NS-3 to suppress its
                       hot-path std::cout (does NOT affect env data via SHM).
                       Default True for fast training; set False for debug.
            disable_ns3_traces: If True, pass --disableTraces=true so NS-3
                       skips opening BI/call-level/E2E/queue/ampdu/etc CSV files.
                       Trace callbacks still fire to populate env arrays.
                       Default True for fast training; set False to keep CSVs.
        """
        self.ns3_root = ns3_root if ns3_root is not None else _ns3_root
        # Handle both absolute and relative log_dir paths
        if os.path.isabs(log_dir):
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(_script_dir, log_dir)
        self.enable_logging = enable_logging
        self.verbose = verbose
        self.quiet_ns3 = quiet_ns3
        self.disable_ns3_traces = disable_ns3_traces
        # Topology: number of non-REHD STAs and REHD energy-harvesting sensors.
        # None => omit the CLI flag and let NS-3 use its compiled default (nStations=ACTIVE_NUM_STA=16, nRehd=0).
        # Set both to spawn REHDs so the energy/QoEH machinery is active in training.
        # total active = n_stations + n_rehd must be <= MAX_NUM_STA (32).
        self.n_stations = n_stations
        self.n_rehd = n_rehd
        self.traffic_scale = traffic_scale

        # State
        self.initialized = False
        self.step_count = 0
        self.current_seed = None

        # NS3-AI interface (lazy loaded)
        self.py_binding = None
        self.exp = None
        self.msgInterface = None

        # CSV logging
        self.env_csv_file = None
        self.env_csv_writer = None
        self.action_csv_file = None
        self.action_csv_writer = None
        self.log_timestamp = None

        # Cached constants
        self.MAX_NUM_STA = None
        self.MAX_NUM_TWT_GROUPS = None

    def _log(self, msg: str, color: str = None):
        """Print message if verbose mode is on."""
        if self.verbose:
            colors = {
                "green": "\033[32m",
                "red": "\033[31m",
                "yellow": "\033[33m",
                "blue": "\033[34m",
                "cyan": "\033[36m",
                "magenta": "\033[35m",
                "reset": "\033[0m",
            }
            if color and color in colors:
                print(f"{colors[color]}{msg}{colors['reset']}")
            else:
                print(msg)

    def _import_bindings(self):
        """Import NS3-AI Python bindings."""
        if self.py_binding is not None:
            return

        try:
            import pb_twt_powercast_interface_py as py_binding

            self.py_binding = py_binding
            self.MAX_NUM_STA = py_binding.MAX_NUM_STA
            self.MAX_NUM_TWT_GROUPS = py_binding.MAX_NUM_TWT_GROUPS
            self._log("✓ Imported pb_twt_powercast_interface_py", "green")
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import pb_twt_powercast_interface_py: {e}\n"
                f"Make sure the binding is built: ./ns3 build"
            )

    def _init_logging(self):
        """Initialize CSV log files."""
        if not self.enable_logging:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        self.log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Environment log (one row per STA per step)
        env_log_path = os.path.join(
            self.log_dir, f"py-wrapper-env-{self.log_timestamp}.csv"
        )
        self.env_csv_file = open(env_log_path, "w", newline="")
        self.env_csv_writer = csv.writer(self.env_csv_file)

        # Write env header
        env_header = [
            "step",
            "sim_time_sec",
            "num_sta",
            "observation_count",
            "beacon_interval_ms",
            "sta_id",
        ]
        env_header += [f"realistic_{f}" for f in self.REALISTIC_FIELDS if f != "sta_id"]
        env_header += [f"oracle_{f}" for f in self.ORACLE_FIELDS]
        self.env_csv_writer.writerow(env_header)

        # Action log (one row per STA per step)
        action_log_path = os.path.join(
            self.log_dir, f"py-wrapper-action-{self.log_timestamp}.csv"
        )
        self.action_csv_file = open(action_log_path, "w", newline="")
        self.action_csv_writer = csv.writer(self.action_csv_file)

        # Write action header
        action_header = [
            "step",
            "sim_time_sec",
            "num_sta",
            "num_active_twt_groups",
            "action_timestamp_ms",
            "sta_id",
            "assigned_twt_group",
            "enable_twt",
            "group_id",
            "group_wake_interval_ms",
            "group_wake_duration_ms",
            "group_sp_offset_ms",
            "group_num_stas_assigned",
        ]
        self.action_csv_writer.writerow(action_header)

        self._log(f"✓ Logging enabled: {env_log_path}", "green")

    def _close_logging(self):
        """Close CSV log files."""
        if self.env_csv_file:
            self.env_csv_file.close()
            self.env_csv_file = None
        if self.action_csv_file:
            self.action_csv_file.close()
            self.action_csv_file = None

    def _log_env(self, env_dict: Dict, sim_time: float):
        """Log environment observation to CSV."""
        if not self.enable_logging or not self.env_csv_writer:
            return

        num_sta = env_dict.get("num_sta", 0)
        obs_count = env_dict.get("observation_count", 0)
        beacon_interval = env_dict.get("beacon_interval_ms", 0)

        for sta_obs in env_dict.get("sta_observations", []):
            row = [self.step_count, sim_time, num_sta, obs_count, beacon_interval]

            # STA ID
            sta_id = sta_obs.get("realistic", {}).get("sta_id", 0)
            row.append(sta_id)

            # Realistic fields (skip sta_id, already added)
            realistic = sta_obs.get("realistic", {})
            for field in self.REALISTIC_FIELDS:
                if field != "sta_id":
                    row.append(realistic.get(field, 0))

            # Oracle fields
            oracle = sta_obs.get("oracle", {})
            for field in self.ORACLE_FIELDS:
                row.append(oracle.get(field, 0))

            self.env_csv_writer.writerow(row)

        self.env_csv_file.flush()

    def _log_action(self, action_dict: Dict, sim_time: float):
        """Log action to CSV."""
        if not self.enable_logging or not self.action_csv_writer:
            return

        num_sta = action_dict.get("num_sta", 0)
        num_groups = action_dict.get("num_active_twt_groups", 0)
        action_ts = action_dict.get("action_timestamp_ms", 0)

        # Build group lookup
        groups = {g["group_id"]: g for g in action_dict.get("twt_group_configs", [])}

        for sta_assign in action_dict.get("sta_group_assignments", []):
            sta_id = sta_assign.get("sta_id", 0)
            assigned_group = sta_assign.get("assigned_twt_group", 0)
            enable_twt = sta_assign.get("enable_twt", 0)

            # Get group params for this STA's assigned group
            group = groups.get(assigned_group, {})

            row = [
                self.step_count,
                sim_time,
                num_sta,
                num_groups,
                action_ts,
                sta_id,
                assigned_group,
                enable_twt,
                group.get("group_id", assigned_group),
                group.get("twt_wake_interval_ms", 0),
                group.get("twt_wake_duration_ms", 0),
                group.get("twt_sp_offset_ms", 0),
                group.get("num_stas_assigned", 0),
            ]
            self.action_csv_writer.writerow(row)

        self.action_csv_file.flush()

    def _env_struct_to_dict(self, env) -> Dict:
        """Convert C++ EnvStruct to Python dict."""
        result = {
            "num_sta": env.num_sta,
            "simulation_time_sec": env.simulation_time_sec,
            "observation_timestamp_ms": env.observation_timestamp_ms,
            "observation_count": env.observation_count,
            "beacon_interval_ms": env.beacon_interval_ms,
            # Global AP TX state (REALISTIC, diagnostic — not in obs/reward).
            "ap_tx_power_dbm": env.ap_tx_power_dbm,
            "dl_broadcast_bytes": env.dl_broadcast_bytes,
            "dl_broadcast_duration_us": env.dl_broadcast_duration_us,
            "dl_broadcast_count": env.dl_broadcast_count,
            "sta_observations": [],
        }

        for i in range(env.num_sta):
            sta_env = env.sta_observations[i]

            # Realistic metrics
            r = sta_env.realistic
            realistic = {
                "sta_id": r.sta_id,
                "is_active": r.is_active,
                "bsr_queue_ac_be": r.bsr_queue_ac_be,
                "bsr_queue_ac_bk": r.bsr_queue_ac_bk,
                "bsr_queue_ac_vi": r.bsr_queue_ac_vi,
                "bsr_queue_ac_vo": r.bsr_queue_ac_vo,
                "bsr_scaling_factor": r.bsr_scaling_factor,
                "rx_fragment_count": r.rx_fragment_count,
                "fcs_error_count": r.fcs_error_count,
                "rcpi": r.rcpi,
                "rsni": r.rsni,
                "rssi_dbm": r.rssi_dbm,
                "snr_db": r.snr_db,
                "link_margin_db": r.link_margin_db,
                "last_rx_frame_type": r.last_rx_frame_type,
                "last_rx_frame_subtype": r.last_rx_frame_subtype,
                "last_rx_mcs": r.last_rx_mcs,
                "last_rx_nss": r.last_rx_nss,
                "channel_width_mhz": r.channel_width_mhz,
                "guard_interval_ns": r.guard_interval_ns,
                "power_mgmt_bit": r.power_mgmt_bit,
                "last_rx_timestamp_us": r.last_rx_timestamp_us,
                "bytes_received_at_ap": r.bytes_received_at_ap,
                "packets_received_at_ap": r.packets_received_at_ap,
                "bytes_received_ac_vo": r.bytes_received_ac_vo,
                "bytes_received_ac_vi": r.bytes_received_ac_vi,
                "bytes_received_ac_be": r.bytes_received_ac_be,
                "bytes_received_ac_bk": r.bytes_received_ac_bk,
                "packets_received_ac_vo": r.packets_received_ac_vo,
                "packets_received_ac_vi": r.packets_received_ac_vi,
                "packets_received_ac_be": r.packets_received_ac_be,
                "packets_received_ac_bk": r.packets_received_ac_bk,
                "airtime_used_us": r.airtime_used_us,
                # QoEH-realistic per-SP UL accounting + AP DL accounting (cumulative).
                "bytes_rx_at_ap_in_sp": r.bytes_rx_at_ap_in_sp,
                "packets_rx_at_ap_in_sp": r.packets_rx_at_ap_in_sp,
                "sp_completed_count": r.sp_completed_count,
                "sp_with_demand_count": r.sp_with_demand_count,
                "sp_with_starvation_count": r.sp_with_starvation_count,
                "dl_bytes_to_sta": r.dl_bytes_to_sta,
                "dl_duration_us_to_sta": r.dl_duration_us_to_sta,
                "dl_unicast_count_to_sta": r.dl_unicast_count_to_sta,
                "user_priority": r.user_priority,
                "observation_time_ms": r.observation_time_ms,
                "observation_sequence_num": r.observation_sequence_num,
            }

            # Oracle metrics
            o = sta_env.oracle
            oracle = {
                "sta_id": o.sta_id,
                # Device characteristics — moved from realistic 2026-06-07 (signal audit).
                "device_class": o.device_class,
                "nominal_msdu_size": o.nominal_msdu_size,
                "mean_data_rate_kbps": o.mean_data_rate_kbps,
                "delay_bound_ms": o.delay_bound_ms,
                "tx_power_dbm": o.tx_power_dbm,
                "tx_fragment_count": o.tx_fragment_count,
                "tx_failed_count": o.tx_failed_count,
                "tx_retry_count": o.tx_retry_count,
                "ack_failure_count": o.ack_failure_count,
                "total_energy_consumed_mj": o.total_energy_consumed_mj,
                "current_times_time_ma_ms": o.current_times_time_ma_ms,
                "awake_time_ms": o.awake_time_ms,
                "sleep_time_ms": o.sleep_time_ms,
                "duty_cycle": o.duty_cycle,
                "packets_generated": o.packets_generated,
                "packets_enqueued": o.packets_enqueued,
                "bytes_generated": o.bytes_generated,
                "mpdu_drops_expired": o.mpdu_drops_expired,
                "mpdu_drops_queue_full": o.mpdu_drops_queue_full,
                "psdu_response_timeouts": o.psdu_response_timeouts,
                "packets_transmitted": o.packets_transmitted,
                "bytes_transmitted": o.bytes_transmitted,
                "ampdu_count": o.ampdu_count,
                "ampdu_mpdus_total": o.ampdu_mpdus_total,
                "ampdu_bytes_total": o.ampdu_bytes_total,
                "queue_size_packets": o.queue_size_packets,
                "queue_size_bytes": o.queue_size_bytes,
                "queue_max_size": o.queue_max_size,
                "avg_latency_ms": o.avg_latency_ms,
                "queue_delay_sum_ms": o.queue_delay_sum_ms,
                "queue_delay_count": o.queue_delay_count,
                "position_x_m": o.position_x_m,
                "position_y_m": o.position_y_m,
                "position_z_m": o.position_z_m,
                # PowerCast energy oracle (REHD-only; 0 for non-REHD).
                # Reward-only, NEVER in the observation.
                # Consumed by the QoEH reward.
                "vcap_v": o.vcap_v,
                "vcap_max_v": o.vcap_max_v,
                "e_nominal_j": o.e_nominal_j,
                "e_sunk_j": o.e_sunk_j,
                "e_avail_j": o.e_avail_j,
                "harvested_total_j": o.harvested_total_j,
                "consumed_total_j": o.consumed_total_j,
                "t_active_us": o.t_active_us,
                "unpowered_tx_events": o.unpowered_tx_events,
                "output_enabled": o.output_enabled,
                "harvester_phase": o.harvester_phase,
                "est_rf_energy_delivered_j": o.est_rf_energy_delivered_j,
            }

            result["sta_observations"].append(
                {"realistic": realistic, "oracle": oracle}
            )

        return result

    def _dict_to_action_struct(self, action_dict: Dict):
        """Convert Python dict to C++ ActionStruct."""
        self.msgInterface.PySendBegin()
        action = self.msgInterface.GetPy2CppStruct()

        # Zero-initialize all arrays
        for i in range(self.MAX_NUM_TWT_GROUPS):
            action.twt_group_configs[i].group_id = 0
            action.twt_group_configs[i].twt_wake_interval_ms = 0.0
            action.twt_group_configs[i].twt_wake_duration_ms = 0.0
            action.twt_group_configs[i].twt_sp_offset_ms = 0.0
            action.twt_group_configs[i].num_stas_assigned = 0

        for i in range(self.MAX_NUM_STA):
            action.sta_group_assignments[i].sta_id = 0
            action.sta_group_assignments[i].assigned_twt_group = 0
            action.sta_group_assignments[i].enable_twt = 0

        # Set metadata
        action.num_sta = action_dict.get("num_sta", 0)
        action.num_active_twt_groups = action_dict.get("num_active_twt_groups", 0)
        action.action_timestamp_ms = action_dict.get("action_timestamp_ms", 0)
        # PDW 3rd action head (Phase 2). 0.0 => PDW off.
        # The Python decoder has already scale-shifted the group offsets/wakes into [D,95]; C++ just applies them + (re)installs the broadcast window for this D.
        action.pdw_duration_ms = float(action_dict.get("pdw_duration_ms", 0.0))

        # Set TWT group configs
        for i, group in enumerate(action_dict.get("twt_group_configs", [])):
            if i >= self.MAX_NUM_TWT_GROUPS:
                break
            action.twt_group_configs[i].group_id = group.get("group_id", i)
            action.twt_group_configs[i].twt_wake_interval_ms = group.get(
                "twt_wake_interval_ms", 102.4
            )
            action.twt_group_configs[i].twt_wake_duration_ms = group.get(
                "twt_wake_duration_ms", 5.0
            )
            action.twt_group_configs[i].twt_sp_offset_ms = group.get(
                "twt_sp_offset_ms", 2.0
            )
            action.twt_group_configs[i].num_stas_assigned = group.get(
                "num_stas_assigned", 0
            )

        # Set STA assignments
        for i, assign in enumerate(action_dict.get("sta_group_assignments", [])):
            if i >= self.MAX_NUM_STA:
                break
            action.sta_group_assignments[i].sta_id = assign.get("sta_id", i)
            action.sta_group_assignments[i].assigned_twt_group = assign.get(
                "assigned_twt_group", 0
            )
            action.sta_group_assignments[i].enable_twt = assign.get("enable_twt", 1)

        self.msgInterface.PySendEnd()

    def _receive_env(self) -> Tuple[Dict, bool]:
        """Receive environment from NS-3."""
        try:
            # Check if finished before receiving
            if self.msgInterface.PyGetFinished():
                return {}, True

            self.msgInterface.PyRecvBegin()
            env = self.msgInterface.GetCpp2PyStruct()
            self.msgInterface.PyRecvEnd()

            if env is None:
                return {}, True

            env_dict = self._env_struct_to_dict(env)
            return env_dict, False

        except Exception as e:
            self._log(f"Error receiving env: {e}", "red")
            return {}, True

    def reset(self, seed: int = 9000, rand_seed: int = None) -> Dict:
        """
        Start/restart NS-3 simulation.

        Args:
            seed: Random seed for NS-3 simulation

        Returns:
            Initial environment observation as dict
        """
        # Clean up previous run if any
        if self.initialized:
            self.close()

        # Generate unique shared memory segment names based on seed
        # This prevents collision between parallel/subprocess episodes
        # Use underscores instead of spaces to avoid command-line parsing issues
        self._segmentName = f"MySeg_{seed}"
        self._cpp2pyMsgName = f"MyCpp2PyMsg_{seed}"
        self._py2cppMsgName = f"MyPy2CppMsg_{seed}"
        self._lockableName = f"MyLockable_{seed}"

        # Extra safety: ensure shared memory is clean even if close() wasn't called
        try:
            import subprocess

            # Clean both old-style fixed names AND new unique names
            subprocess.run(
                [
                    "rm",
                    "-f",
                    "/dev/shm/My Seg",
                    "/dev/shm/My Cpp to Python Msg",
                    "/dev/shm/My Python to Cpp Msg",
                    "/dev/shm/My Lockable",
                    f"/dev/shm/{self._segmentName}",
                    f"/dev/shm/{self._cpp2pyMsgName}",
                    f"/dev/shm/{self._py2cppMsgName}",
                    f"/dev/shm/{self._lockableName}",
                ],
                capture_output=True,
                timeout=5,
            )
            # Reset singleton flag
            from ns3ai_utils import Experiment

            Experiment._created = False
        except:
            pass

        self.current_seed = seed
        self.step_count = 0

        # Import bindings
        self._import_bindings()

        # Initialize logging
        self._init_logging()

        # Start NS-3 simulation
        self._log(f"\nStarting NS-3 simulation with seed {seed}...", "blue")
        self._log(f"  Segment name: {self._segmentName}", "cyan")

        try:
            from ns3ai_utils import Experiment

            self.exp = Experiment(
                "twt-powercast-main-simulation",
                self.ns3_root,
                self.py_binding,
                handleFinish=True,
                useVector=False,
                shmSize=262144,
                segName=self._segmentName,
                cpp2pyMsgName=self._cpp2pyMsgName,
                py2cppMsgName=self._py2cppMsgName,
                lockableName=self._lockableName,
            )

            run_setting = {
                # CRN: rand_seed (shared per scenario-group) drives the NS-3 scenario; SHM names use `seed` (unique per worker).
                # None => scenario == seed.
                "randSeed": seed if rand_seed is None else rand_seed,
                "segmentName": self._segmentName,
                "cpp2pyMsgName": self._cpp2pyMsgName,
                "py2cppMsgName": self._py2cppMsgName,
                "lockableName": self._lockableName,
                # Quiet/disable-traces: passed as lower-case strings because NS-3 CommandLine accepts "true"/"false" for bool args.
                "quietMode": "true" if self.quiet_ns3 else "false",
                "disableTraces": "true" if self.disable_ns3_traces else "false",
            }
            # Topology overrides (only when explicitly set; otherwise NS-3 uses its compiled defaults).
            # nRehd>0 activates the energy/QoEH machinery.
            if self.n_stations is not None:
                run_setting["nStations"] = int(self.n_stations)
            if self.n_rehd is not None:
                run_setting["nRehd"] = int(self.n_rehd)
            if self.traffic_scale is not None:
                run_setting["trafficScale"] = float(self.traffic_scale)

            self.msgInterface = self.exp.run(
                setting=run_setting,
                show_output=True,  # Must be True: show_output=False causes a 64 KB pipe-buffer deadlock once the C++ side writes enough stdout
            )

            self.initialized = True
            self._log("✓ Connected to NS-3", "green")

            # Receive initial environment
            env_dict, done = self._receive_env()

            if done:
                raise RuntimeError("Simulation ended immediately after start")

            # Log initial env
            sim_time = env_dict.get("simulation_time_sec", 0)
            self._log_env(env_dict, sim_time)

            return env_dict

        except Exception as e:
            self._log(f"Failed to start simulation: {e}", "red")
            traceback.print_exc()
            raise

    def step(self, action_dict: Dict) -> Tuple[Dict, bool]:
        """
        Send action to NS-3 and receive next environment.

        Args:
            action_dict: TWT scheduling action containing:
                - num_sta: Number of STAs
                - num_active_twt_groups: Number of active TWT groups
                - twt_group_configs: List of group configurations
                - sta_group_assignments: List of STA-to-group assignments

        Returns:
            Tuple of (env_dict, done):
                - env_dict: Next environment observation
                - done: True if simulation has ended
        """
        if not self.initialized:
            raise RuntimeError("Wrapper not initialized. Call reset() first.")

        self.step_count += 1

        try:
            # Get current sim time for logging (before sending action)
            # We'll update this after receiving new env

            # Send action to NS-3
            self._dict_to_action_struct(action_dict)

            # Log action
            # Use step_count as proxy for time until we get new env
            self._log_action(action_dict, self.step_count)

            # Check if finished
            if self.msgInterface.PyGetFinished():
                self._log("Simulation finished signal received", "green")
                return {}, True

            # Receive next environment
            env_dict, done = self._receive_env()

            if done:
                self._log("Simulation ended", "green")
                return {}, True

            # Log environment with actual sim time
            sim_time = env_dict.get("simulation_time_sec", 0)
            self._log_env(env_dict, sim_time)

            if self.verbose and self.step_count % 10 == 0:
                self._log(f"[Step {self.step_count}] t={sim_time:.2f}s", "cyan")

            return env_dict, False

        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["finished", "experiment", "connection"]):
                self._log("Simulation completed", "green")
                return {}, True
            else:
                self._log(f"Error in step: {e}", "red")
                traceback.print_exc()
                return {}, True

    def is_done(self) -> bool:
        """Check if simulation has finished."""
        if not self.initialized or not self.msgInterface:
            return True
        return self.msgInterface.PyGetFinished()

    def close(self):
        """Cleanup resources properly, including killing NS-3 subprocess."""
        self._log("\nClosing wrapper...", "yellow")

        self._close_logging()

        # Kill the NS-3 subprocess and cleanup shared memory.
        # Without this, subsequent reset() calls will fail with:
        # "boost::interprocess_exception::library_error"
        if self.exp is not None:
            try:
                self._log("Killing NS-3 subprocess...", "yellow")
                self.exp.kill()
            except Exception as e:
                self._log(f"Warning: Error killing experiment: {e}", "yellow")

            # Delete the Experiment object to release shared memory
            try:
                del self.exp
            except Exception as e:
                self._log(f"Warning: Error deleting experiment: {e}", "yellow")

            # Reset the singleton flag so a new Experiment can be created
            try:
                from ns3ai_utils import Experiment

                Experiment._created = False
            except Exception as e:
                self._log(
                    f"Warning: Could not reset Experiment singleton: {e}", "yellow"
                )

        # exp.kill() kills the ./ns3-run launcher but can leave the ns3 BINARY grandchild orphaned (it spins / oversubscribes cores and blocks SHM reuse).
        # SIGKILL any ns3 binary tagged with OUR unique segment name — safe (never touches another worker's sim) and stops the orphan leak that piled up across parallel runs.
        try:
            import glob as _glob
            import signal as _sig

            _seg = getattr(self, "_segmentName", None)
            if _seg:
                for _d in _glob.glob("/proc/[0-9]*"):
                    try:
                        _c = (
                            open(_d + "/cmdline", "rb")
                            .read()
                            .decode("utf-8", "replace")
                        )
                    except Exception:
                        continue
                    if "ns3.44-twt" in _c and _seg in _c:
                        try:
                            os.kill(int(_d.split("/")[-1]), _sig.SIGKILL)
                        except Exception:
                            pass
        except Exception:
            pass

        # Explicitly remove shared memory segments by name
        # This is the nuclear option to ensure clean state
        try:
            import subprocess

            # Clean both old-style fixed names AND unique names from this session
            shm_files = [
                "/dev/shm/My Seg",
                "/dev/shm/My Cpp to Python Msg",
                "/dev/shm/My Python to Cpp Msg",
                "/dev/shm/My Lockable",
            ]
            # Also clean unique segment names if they exist
            if hasattr(self, "_segmentName") and self._segmentName:
                shm_files.extend(
                    [
                        f"/dev/shm/{self._segmentName}",
                        f"/dev/shm/{self._cpp2pyMsgName}",
                        f"/dev/shm/{self._py2cppMsgName}",
                        f"/dev/shm/{self._lockableName}",
                    ]
                )
            subprocess.run(["rm", "-f"] + shm_files, capture_output=True, timeout=5)
        except Exception as e:
            self._log(f"Warning: Could not clean shared memory: {e}", "yellow")

        # Give time for cleanup
        import time

        time.sleep(1.0)

        # Reset state
        self.initialized = False
        self.msgInterface = None
        self.exp = None

        self._log(f"Total steps: {self.step_count}", "green")
        if self.log_timestamp:
            self._log(f"Logs saved with timestamp: {self.log_timestamp}", "green")

    def get_num_sta(self) -> int:
        """Get maximum number of STAs supported."""
        self._import_bindings()
        return self.MAX_NUM_STA

    def get_num_twt_groups(self) -> int:
        """Get maximum number of TWT groups supported."""
        self._import_bindings()
        return self.MAX_NUM_TWT_GROUPS


# --- Convenience function for quick testing ---


def create_default_action(num_sta: int, num_groups: int = 1) -> Dict:
    """
    Create a default action dict for testing.

    Args:
        num_sta: Number of STAs
        num_groups: Number of TWT groups

    Returns:
        Action dict with default TWT parameters
    """
    # Create groups
    twt_group_configs = []
    stas_per_group = num_sta // num_groups

    for g in range(num_groups):
        twt_group_configs.append(
            {
                "group_id": g,
                "twt_wake_interval_ms": 102.4,
                "twt_wake_duration_ms": 5.0,
                "twt_sp_offset_ms": 2.0 + g * 10.0,  # Stagger offsets
                "num_stas_assigned": stas_per_group,
            }
        )

    # Assign STAs to groups (round-robin)
    sta_group_assignments = []
    for i in range(num_sta):
        sta_group_assignments.append(
            {"sta_id": i, "assigned_twt_group": i % num_groups, "enable_twt": 1}
        )

    return {
        "num_sta": num_sta,
        "num_active_twt_groups": num_groups,
        "action_timestamp_ms": 0,
        "twt_group_configs": twt_group_configs,
        "sta_group_assignments": sta_group_assignments,
    }


# --- Demo / Test ---

if __name__ == "__main__":
    """Demo showing wrapper usage."""
    import argparse

    parser = argparse.ArgumentParser(description="TWT Wrapper Demo")
    parser.add_argument("--seed", type=int, default=9000, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps")
    args = parser.parse_args()

    print("=" * 60)
    print("TWT Python Wrapper Demo")
    print("=" * 60)

    wrapper = TWTWrapper(verbose=True)

    try:
        # Reset (starts NS-3)
        env = wrapper.reset(seed=args.seed)
        print(
            f"\nInitial env received: {env['num_sta']} STAs, t={env['simulation_time_sec']:.2f}s"
        )

        # Run for max_steps
        for step in range(args.max_steps):
            # Create action (cycle through 1-4 groups)
            num_groups = (step % 4) + 1
            action = create_default_action(env["num_sta"], num_groups)

            # Step
            env, done = wrapper.step(action)

            if done:
                print(f"\nSimulation ended at step {step}")
                break

            # Print periodic status
            if step % 20 == 0:
                print(
                    f"Step {step}: t={env['simulation_time_sec']:.2f}s, "
                    f"groups={num_groups}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        wrapper.close()

    print("\n" + "=" * 60)
    print("Demo complete. Check data-log/ for CSV files.")
    print("=" * 60)
