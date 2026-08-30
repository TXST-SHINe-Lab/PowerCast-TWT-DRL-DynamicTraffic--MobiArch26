#ifndef PB_TWT_CORE_H
#define PB_TWT_CORE_H

#include "twt-constants.h"

#include <cstdint>

// Forward declaration for ns3-ai message interface
namespace ns3
{
template <typename EnvType, typename ActType>
class Ns3AiMsgInterfaceImpl;
}

// --- PER-STA PROTOCOL-COMPLIANT OBSERVATIONS (802.11ax/k) ---
/**
 * StaRealisticMetrics: Per-STA metrics observable via 802.11ax/k protocols
 *
 * These metrics can be obtained by a REAL ACCESS POINT using standard
 * 802.11ax and 802.11k protocol mechanisms. Use this struct for training
 * RL agents that will deploy on real hardware.
 *
 * Data sources:
 * - 802.11ax: BSR (Buffer Status Report) per Access Category
 * - 802.11k: STA Statistics Report (Section 7.3.2.22)
 * - 802.11k: Link Measurement Report
 * - MAC headers: MCS, NSS observed from received frames
 * - 802.11e TSPEC: Device characteristics declared during association
 */
struct StaRealisticMetrics
{
    // --- IDENTIFICATION ---

    uint32_t sta_id;               // STA identifier (0 to NUM_STA-1)
    uint8_t sta_mac[MAC_ADDR_LEN]; // MAC address (6 bytes)
    uint8_t is_active;             // Is this STA currently active (0/1)

    // --- 802.11ax BSR (Buffer Status Report) - Per Access Category ---
    // IEEE 802.11ax-2021 Section 9.2.4.6.6
    // AP receives BSR in Trigger Frame responses or QoS Data frames
    uint8_t bsr_queue_ac_be;     // Best Effort queue (AC_BE, TID 0,3) - quantized 0-255
    uint8_t bsr_queue_ac_bk;     // Background queue (AC_BK, TID 1,2) - quantized 0-255
    uint8_t bsr_queue_ac_vi;     // Video queue (AC_VI, TID 4,5) - quantized 0-255
    uint8_t bsr_queue_ac_vo;     // Voice queue (AC_VO, TID 6,7) - quantized 0-255
    uint16_t bsr_scaling_factor; // Bytes per BSR unit (typically 256)

    // --- AP-OBSERVABLE RX COUNTERS (Truly realistic - AP directly observes) ---
    uint64_t rx_fragment_count; // AP counts frames it receives from STA
    uint64_t fcs_error_count;   // AP counts RX decode failures (FCS errors)

    // --- 802.11k LINK MEASUREMENT - Raw + converted values ---
    // IEEE 802.11k-2008 Section 7.3.2.18
    // AP requests via Link Measurement Request frame
    // INT8_MIN (-128) is used as sentinel for "no valid measurement" since int8_t cannot hold NAN.
    // Check for this before using values.
    int8_t rcpi;           // Raw RCPI (0.5 dBm units, range 0-220), INT8_MIN = no data
    int8_t rsni;           // Raw RSNI (0.5 dB units, range 0-255), INT8_MIN = no data
    int8_t rssi_dbm;       // Converted from RCPI: (rcpi/2) - 110, INT8_MIN = no data
    int8_t snr_db;         // Converted from RSNI: rsni/2, INT8_MIN = no data
    int8_t link_margin_db; // dB above RX sensitivity threshold, INT8_MIN = no data
    // tx_power_dbm MOVED to StaOracleMetrics (2026-06-07): a STA's TX power is only learnable via an 802.11h/k TPC Report, which this sim does not model (read from config) -> oracle.
    // (link_margin_db stays: it is set = SNR, AP-measured.)

    // --- MAC LAYER OBSERVATIONS - AP observes from received frame headers ---
    // NOTE: 0 indicates no observation yet for uint8_t fields
    uint8_t last_rx_frame_type;    // Frame type: 0=mgmt, 1=ctrl, 2=data
    uint8_t last_rx_frame_subtype; // Frame subtype (0-15)
    uint8_t last_rx_mcs;           // MCS index from last received frame (0-11 HE)
    uint8_t last_rx_nss;           // NSS from last received frame
    uint8_t channel_width_mhz;     // Channel width (20/40/80/160)
    uint16_t guard_interval_ns;    // Guard interval: 800, 1600, or 3200 ns
    uint8_t power_mgmt_bit;        // Power Management bit from Frame Control
    uint64_t last_rx_timestamp_us; // TSF timestamp of last received frame

    // --- AP-DERIVED METRICS - Computed from received frames at AP ---
    uint64_t bytes_received_at_ap;   // Total bytes AP received from this STA
    uint64_t packets_received_at_ap; // Total packets AP received from this STA

    // Per-AC transmission tracking (from QoS Control field TID)
    uint64_t bytes_received_ac_vo;   // Voice (TID 6,7)
    uint64_t bytes_received_ac_vi;   // Video (TID 4,5)
    uint64_t bytes_received_ac_be;   // Best Effort (TID 0,3)
    uint64_t bytes_received_ac_bk;   // Background (TID 1,2)
    uint64_t packets_received_ac_vo; // Voice packet count
    uint64_t packets_received_ac_vi; // Video packet count
    uint64_t packets_received_ac_be; // Best Effort packet count
    uint64_t packets_received_ac_bk; // Background packet count

    // Airtime tracking (from frame duration field)
    uint64_t airtime_used_us; // Total airtime used by this STA (microseconds)

    // --- QoS USER PRIORITY — AP reads the TID/UP directly from received QoS frames. ---
    // (device_class / nominal_msdu_size / mean_data_rate_kbps / delay_bound_ms were MOVED to StaOracleMetrics 2026-06-07.
    //  The 3 TSPEC descriptors are realistic-*eligible* via 802.11e ADDTS and tx_power via 802.11h/k TPC, but this sim models none of that signaling (reads config); device_class has no standard signal at all and rule #25 keeps the AP blind to class — so all are oracle.)
    uint8_t user_priority; // QoS UP (0-7) — AP reads it from received QoS frames

    // --- QoEH-REALISTIC INPUTS (AP-observable, for Quality of Energy Harvesting) ---
    // All cumulative — Python computes deltas across calls.

    // Per-SP gated UL accounting: τ_{i,k} = bytes/packets AP received from STA i within STA i's currently-assigned TWT SP window, summed across all completed SPs since simulation start.
    // Used to detect SP underutilization.
    uint64_t bytes_rx_at_ap_in_sp;     // Cumulative τ_{i,k} sum
    uint64_t packets_rx_at_ap_in_sp;   // Cumulative packet count counterpart
    uint32_t sp_completed_count;       // # SPs whose end boundary this STA crossed
    uint32_t sp_with_demand_count;     // # SPs where BSR > B_min at SP start
    uint32_t sp_with_starvation_count; // # SPs flagged as starved (demand ∧ under-served)

    // AP DL accounting (AP knows what it sent — fully realistic).
    uint64_t dl_bytes_to_sta;         // Cumulative DL unicast bytes AP → STA i
    uint64_t dl_duration_us_to_sta;   // Cumulative DL airtime AP → STA i (µs)
    uint32_t dl_unicast_count_to_sta; // # DL unicast frames AP → STA i

    // The "last UL RX timestamp" lives in last_rx_timestamp_us (above), which IS the last-UL-frame TSF (populated from lastRxTimestampUsForSta) and already backs the obs silence_norm feature.
    // A separate last_ul_rx_ts_us would duplicate it, so it was removed.

    // --- OBSERVATION METADATA ---
    double observation_time_ms;        // Simulation time of this observation
    uint32_t observation_sequence_num; // Monotonic sequence number for ordering
};

// --- PER-STA ORACLE/SIMULATION METRICS (NOT Protocol Compliant) ---
/**
 * StaOracleMetrics: Per-STA simulation-only metrics for validation
 *
 * These metrics are ONLY available in simulation. A real AP cannot observe
 * these without STA cooperation or firmware modifications. Use for:
 * - Upper bound comparison (oracle agent vs realistic agent)
 * - Debugging and validation
 * - Ground truth for reward computation (during training only)
 *
 * DO NOT use these for RL state in deployment-ready agents!
 */
struct StaOracleMetrics
{
    uint32_t sta_id; // STA identifier (must match realistic)

    // --- STA TX COUNTERS (Would need 802.11k STA Statistics Request in real deployment) ---
    // These are STA-side local counters, NOT directly observable by AP
    uint64_t tx_fragment_count; // dot11TransmittedFragmentCount (STA-local)
    uint64_t tx_failed_count;   // dot11FailedCount (STA-local)
    uint64_t tx_retry_count;    // dot11RetryCount (STA-local)
    uint64_t ack_failure_count; // dot11ACKFailureCount (STA-local)

    // --- ENERGY (STA-private, not reportable via 802.11) ---
    double total_energy_consumed_mj; // Cumulative energy consumption
    double current_times_time_ma_ms; // Integral of current*time (for avg power)
    double awake_time_ms;            // Time in active states (TX/RX/IDLE/CCA)
    double sleep_time_ms;            // Time in SLEEP state
    double duty_cycle;               // awake_time / total_time

    // --- APPLICATION LAYER (above MAC, not visible to AP) ---
    uint64_t packets_generated; // Packets created by application
    uint64_t packets_enqueued;  // Packets successfully enqueued at MAC
    uint64_t bytes_generated;   // Bytes created by application

    // --- STA QUEUE DROPS (internal to STA, not reportable) ---
    uint64_t mpdu_drops_expired;     // Drops due to lifetime expiry
    uint64_t mpdu_drops_queue_full;  // Drops due to queue overflow
    uint64_t psdu_response_timeouts; // PSDU timeouts (no ACK received)

    // --- TX SIDE METRICS (STA-local, not reported to AP) ---
    uint64_t packets_transmitted; // Packets sent by PHY
    uint64_t bytes_transmitted;   // Bytes sent by PHY
    uint64_t ampdu_count;         // Number of A-MPDU transmissions
    uint64_t ampdu_mpdus_total;   // Total MPDUs in A-MPDUs
    uint64_t ampdu_bytes_total;   // Total bytes in A-MPDUs

    // --- QUEUE STATE (STA-internal snapshot) ---
    uint32_t queue_size_packets; // Current queue size (packets)
    uint32_t queue_size_bytes;   // Current queue size (bytes)
    uint32_t queue_max_size;     // Maximum queue capacity

    // --- LATENCY (requires app-layer timestamp, not in 802.11) ---
    double avg_latency_ms;      // Average packet latency
    double queue_delay_sum_ms;  // Sum of queue delays
    uint64_t queue_delay_count; // Number of delay samples

    // --- STA POSITION (simulation-only, not reportable via 802.11) ---
    double position_x_m; // STA x-coordinate in meters
    double position_y_m; // STA y-coordinate in meters
    double position_z_m; // STA z-coordinate in meters (height/altitude)

    // --- POWERCAST ENERGY HARVESTING — ORACLE (STA-internal capacitor state + sim-side η-curve estimates) ---
    // Ground truth from PowercastEnergyHarvester / BasicEnergySource.
    // Not visible to the AP in any real deployment.
    // Phase 2 of the QoEH plan.

    // AP-SIDE ESTIMATE of RF energy delivered to STA i (joules, cumulative).
    // Computed by integrating PowerCast η(P_rx@STA) over each DL TX event (unicast + broadcast) using RSSI-derived pathloss + assumed η curve.
    // CLASSIFIED ORACLE because it requires the AP to know the STA's hardware-class η curve — that's only available via vendor-specific advertisement (Vendor IE at association) which is a stretch for "standard protocol".
    // Use this paired with harvested_total_j below for gap analysis (pathloss-estimation + η-modeling error), not as an obs/reward input.
    double est_rf_energy_delivered_j;

    double vcap_v;      // Current capacitor voltage (V)
    double vcap_max_v;  // Hardware-class max voltage (1.2 / 0.9 / 0.7 V)
    double e_nominal_j; // ½·C·V² — total stored electrostatic energy
    double e_sunk_j;  // ½·C·(DEEP_RATIO·V_min)² (0.8·V_min) — unusable reserve below the hard floor
    double e_avail_j; // e_nominal − e_sunk — truly usable budget
    double harvested_total_j; // Cumulative harvested energy (J)
    double consumed_total_j;  // Cumulative consumed energy (J)
    uint64_t t_active_us;     // Cumulative time (µs) where vcap_v > vcap_min — for S_r computation
    uint64_t unpowered_tx_events; // Cumulative # TX where CanSustainTransmission() = false
    uint8_t output_enabled;       // Boost converter currently delivering output (0/1)
    uint8_t harvester_phase;      // 0=charging, 1=active, 2=discharging, 3=protection

    // --- DEVICE CHARACTERISTICS — moved from realistic 2026-06-07 (signal audit) ---

    // device_class: no standard signal (AP can only infer); rule #25 keeps AP blind.
    // The 3 TSPEC descriptors + tx_power are realistic-*eligible* (802.11e ADDTS / 802.11h-k TPC) but this sim reads them from config — so treated as oracle, ground-truth/reward-and-analysis only, NEVER in the observation.
    uint8_t device_class;       // 0=IoT,1=Camera,2=Voice,3=Video,4=REHD
    double nominal_msdu_size;   // Nominal MSDU size (802.11e TSPEC; config here)
    double mean_data_rate_kbps; // Mean data rate (802.11e TSPEC; config here)
    double delay_bound_ms;      // Delay bound / deadline (802.11e TSPEC; config here)
    uint8_t tx_power_dbm;       // STA TX power (802.11h/k TPC Report; config here)
};

// --- COMBINED PER-STA STRUCTURE ---
/**
 * StaEnvStruct: Complete per-STA observations (realistic + oracle)
 *
 * Contains both protocol-compliant and oracle metrics in separate sub-structs.
 * Python code should use ONLY `realistic` for training deployable agents.
 */
struct StaEnvStruct
{
    StaRealisticMetrics realistic; // Protocol-compliant (802.11ax/k)
    StaOracleMetrics oracle;       // Simulation-only (for validation)
};

// --- ENVIRONMENT STRUCTURE ---
/**
 * EnvStruct: Complete System Observations
 *
 * Contains metadata and per-STA observations.
 * Sent from NS-3 C++ to Python controller.
 *
 * Note: Aggregate metrics (TWT group assignments, channel utilization, etc.)
 * are NOT included because:
 * - TWT group info: Controller already knows this from the action it sent
 * - Channel/collision stats: Can be derived from per-STA metrics in Python
 * All aggregations should be computed in Python from per-STA data.
 */
struct EnvStruct
{
    // --- METADATA ---

    uint32_t num_sta;                  // Number of active STAs
    double simulation_time_sec;        // Current simulation time
    uint64_t observation_timestamp_ms; // When observation was captured
    uint32_t observation_count;        // Sequential observation counter
    double beacon_interval_ms;         // Beacon interval (TWT timing reference)

    // --- GLOBAL AP TX STATE (NEW for QoEH realistic) ---

    // AP knows its own TX behavior — bookkeeping/visibility for broadcast traffic and AP TX power.
    // Per-STA broadcast energy attribution is folded into each STA's est_rf_energy_delivered_j (in StaRealisticMetrics), so these globals are not load-bearing for the QoEH reward — they exist for diagnostics and future power-beacon work.
    double ap_tx_power_dbm;            // Current AP TX power setting (dBm)
    uint64_t dl_broadcast_bytes;       // Cumulative broadcast/multicast DL bytes
    uint64_t dl_broadcast_duration_us; // Cumulative broadcast/multicast DL airtime (µs)
    uint32_t dl_broadcast_count;       // Cumulative # broadcast/multicast DL frames

    // --- PER-STA OBSERVATIONS ---

    StaEnvStruct sta_observations[MAX_NUM_STA];
};

// --- TWT GROUP CONFIGURATION STRUCTURE ---
/**
 * TwtGroupConfig: TWT Group Parameters
 *
 * Defines TWT timing parameters for an entire TWT group.
 * All STAs assigned to this group will share these parameters.
 */
struct TwtGroupConfig
{
    uint8_t group_id; // TWT group identifier (0 to MAX_NUM_TWT_GROUPS-1)

    // --- TWT TIMING PARAMETERS ---

    double twt_wake_interval_ms; // Wake interval (how often STAs wake up)
    double twt_wake_duration_ms; // Wake duration (how long STAs stay awake)
    double twt_sp_offset_ms;     // Service Period offset from beacon

    // --- GROUP METADATA ---

    uint8_t num_stas_assigned; // Number of STAs currently assigned to this group
};

// --- STA GROUP ASSIGNMENT STRUCTURE ---
/**
 * StaGroupAssignment: Assigns a STA to a TWT Group
 *
 * Maps a single STA to a TWT group. The STA will inherit all
 * TWT parameters from the assigned group.
 */
struct StaGroupAssignment
{
    uint32_t sta_id;            // STA identifier (must match observation)
    uint8_t assigned_twt_group; // TWT group assignment (0 to MAX_NUM_TWT_GROUPS-1)
    uint8_t enable_twt;         // Enable TWT for this STA (0=disable, 1=enable)
};

// --- AGGREGATE ACTION STRUCTURE ---
/**
 * ActionStruct: Complete TWT Scheduling Decision
 *
 * Contains TWT group configurations and STA-to-group assignments.
 * This is sent from Python controller to NS-3 C++.
 *
 * Design: Group-centric approach where:
 * 1. TWT groups define timing parameters (wake interval, duration, offset)
 * 2. STAs are assigned to groups and inherit group parameters
 * 3. Multiple STAs can share the same TWT group (contention-based access)
 */
struct ActionStruct
{
    // --- METADATA ---

    uint32_t num_sta;               // Number of STAs being controlled
    uint32_t num_active_twt_groups; // Number of active TWT groups (0 to MAX_NUM_TWT_GROUPS)
    uint64_t action_timestamp_ms;   // When action was generated

    // --- POWER DELIVERY WINDOW (PDW) — agent's 3rd action head (Phase 2) ---

    // Per-BI window [0, pdw_duration_ms] in which the AP broadcasts dedicated power frames so sleeping REHDs harvest.
    // The Python decoder has ALREADY scale-shifted twt_group_configs into the comms region [D,95] (front-guard = D), so ApplyTWTSchedule applies the group offsets directly (no +2 ms baseline) and (re)installs the front-aligned broadcast window for this D.
    // 0.0 => PDW off (backward-compatible with the pre-PDW schedule path).
    double pdw_duration_ms;

    // --- TWT GROUP CONFIGURATIONS ---

    TwtGroupConfig twt_group_configs[MAX_NUM_TWT_GROUPS]; // Array of TWT group parameters

    // --- STA-TO-GROUP ASSIGNMENTS ---

    StaGroupAssignment sta_group_assignments[MAX_NUM_STA]; // Array of STA group assignments
};

// --- C++ INTERFACE FUNCTIONS ---

/**
 * GetNs3AiInterface: Initialize NS3-AI message interface for TWT controller
 * Creates and configures the message interface for EnvStruct/ActionStruct communication
 *
 * Returns:
 *   msgInterface: Configured message interface
 */
ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>* GetNs3AiInterface();

/**
 * Send2Python: Send environment observations to Python and receive TWT schedule
 *
 * @param msgInterface: NS-3 AI message interface
 * @param env_struct: Environment observations (all per-STA metrics)
 * @return action_struct: TWT scheduling decisions from Python controller
 */
ActionStruct Send2Python(ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>* msgInterface,
                         const EnvStruct& env_struct);

#endif // PB_TWT_CORE_H