// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file twt-simulation-config.h
 * @brief Network topology, device and REHD classes, and TWT/PDW schedule application
 */

#ifndef TWT_SIMULATION_CONFIG_H
#define TWT_SIMULATION_CONFIG_H

#include "pb-twt-core.h"
#include "ph-harvester-hardware.h" // PowerCast RF energy harvester (per-STA)
#include "twt-constants.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/energy-module.h" // BasicEnergySource for the harvesters to attach to
#include "ns3/internet-module.h"
#include "ns3/mobility-helper.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/wifi-module.h"

#include <unordered_map>

namespace ns3
{

// --- DEVICE CLASS DEFINITIONS FOR HETEROGENEOUS NETWORK ---

/**
 * DeviceClass: Enumeration for device types in heterogeneous network
 * Each class has distinct traffic patterns, energy constraints, and QoS requirements
 */
enum DeviceClass
{
    DEVICE_IOT_SENSOR = 0,      // Low rate, periodic, battery-powered (AC_BK after refactor)
    DEVICE_VIDEO_CAMERA = 1,    // Medium rate, constant, plugged-in (AC_VI)
    DEVICE_VOICE_ASSISTANT = 2, // Low rate, bursty, latency-critical (AC_VO)
    DEVICE_VIDEO_STREAMING = 3, // High rate, elastic, interactive (AC_VI)
    DEVICE_REHD = 4 // RF energy-harvesting sensor; static, very light intermittent (AC_BK)
};

/**
 * RehdType: REHD hardware sub-types. Each REHD STA is randomly assigned one of
 * these four templates at episode start. Sub-type fixes (capacitor, voltage)
 * hardware classes; traffic params are then sampled from per-template ranges.
 */
enum RehdType
{
    REHD_T1 = 0, // Tiny telemetry heartbeat   (CapA / Volt1 / 500±100 bps,  32±4B,  30-60s)
    REHD_T2 = 1, // Motion / occupancy ping    (CapB / Volt2 / 2500±500 bps, 80±8B,  5-15s)
    REHD_T3 = 2, // Asset tracker location     (CapC / Volt1 / 1000±200 bps, 200±20B,30-90s)
    REHD_T4 = 3, // Vibration sampler          (CapB / Volt3 / 4000±800 bps, 100±10B,3-10s)
    REHD_TYPE_COUNT
};

/**
 * RehdTypeTemplate: traffic + hardware ranges for one REHD sub-type. At episode
 * start, each REHD samples its (rate, packet_size, interval) uniformly from
 * these ranges and then is fixed for the remainder of the episode. The
 * capacitor + voltage classes come straight from the template (no jitter —
 * hardware is hardware).
 */
struct RehdTypeTemplate
{
    RehdType type;
    const char* name;
    // PowerCast hardware class (consumed by SetupEnergyHarvesting per-REHD)
    energy::PowercastEnergyHarvester::CapacitorClass cap_class;
    energy::PowercastEnergyHarvester::VoltageClass volt_class;
    // Traffic param ranges (drawn once at init, frozen for the episode)
    uint32_t rate_min_bps;
    uint32_t rate_max_bps;
    uint32_t pkt_size_min_bytes;
    uint32_t pkt_size_max_bytes;
    double interval_min_s;
    double interval_max_s;
    // Constant fields for the type
    double
        latency_bound_s; // not used: every REHD gets the 200 ms deadline set in InitializeHeterogeneousNetwork()
    double priority_level;
};

// Traffic ramped 2026-05-28 so each REHD generates ~1 data packet every few beacon intervals (BI = 102.4 ms) — enough to build a backlog that the energy-gated TX must drain.
// The OnOff app fires a fixed 50 ms burst every `interval` s; the peak rate is sized so that 50 ms ≈ one packet, and the interval is set to a few BI.
// So `interval` is the burst period and `peak` sets packets-per-burst.
// Resulting data-packet spacing ≈ interval ≈ 2-6 BI.
static const RehdTypeTemplate REHD_T1_TEMPLATE = {
    REHD_T1,
    "REHD-T1-Telemetry",
    energy::PowercastEnergyHarvester::CLASS_A,
    energy::PowercastEnergyHarvester::CLASS_1,
    12000,
    18000, // 15k ± 3k bps → ~1 pkt per 50 ms burst (60 B = 480 b)
    48,
    72, // 60 ± 12 B
    0.4,
    0.6,  // burst every 0.4-0.6 s (~4-6 BI)
    30.0, // latency_bound_s (unused)
    0.2   // lowest priority (AC_BK)
};

static const RehdTypeTemplate REHD_T2_TEMPLATE = {REHD_T2,
                                                  "REHD-T2-Motion",
                                                  energy::PowercastEnergyHarvester::CLASS_B,
                                                  energy::PowercastEnergyHarvester::CLASS_2,
                                                  24000,
                                                  34000, // ~1 pkt per 50 ms burst (140 B = 1120 b)
                                                  120,
                                                  160, // 140 ± 20 B
                                                  0.25,
                                                  0.4, // burst every 0.25-0.4 s (~3 BI)
                                                  10.0,
                                                  0.2};

static const RehdTypeTemplate REHD_T3_TEMPLATE = {REHD_T3,
                                                  "REHD-T3-AssetTracker",
                                                  energy::PowercastEnergyHarvester::CLASS_C,
                                                  energy::PowercastEnergyHarvester::CLASS_1,
                                                  60000,
                                                  80000, // ~1 pkt per 50 ms burst (350 B = 2800 b)
                                                  300,
                                                  400, // 350 ± 50 B
                                                  0.3,
                                                  0.5, // burst every 0.3-0.5 s (~3-5 BI)
                                                  30.0,
                                                  0.2};

static const RehdTypeTemplate REHD_T4_TEMPLATE = {
    REHD_T4,
    "REHD-T4-Vibration",
    energy::PowercastEnergyHarvester::CLASS_B,
    energy::PowercastEnergyHarvester::CLASS_3,
    32000,
    46000, // ~1 pkt per 50 ms burst (185 B = 1480 b)
    150,
    220, // 185 ± 35 B
    0.2,
    0.3, // burst every 0.2-0.3 s (~2-3 BI) — most demanding
    5.0,
    0.3 // slightly higher priority — more demanding sampler
};

static const RehdTypeTemplate REHD_TYPE_TEMPLATES[REHD_TYPE_COUNT] = {
    REHD_T1_TEMPLATE, REHD_T2_TEMPLATE, REHD_T3_TEMPLATE, REHD_T4_TEMPLATE};

/**
 * StaApplicationConfig: Per-STA application configuration structure
 * Contains all parameters needed to configure traffic generation and QoS for one STA
 */
struct StaApplicationConfig
{
    uint32_t sta_id;          // STA identifier
    DeviceClass device_class; // Device type

    // Traffic characteristics
    uint32_t traffic_rate_bps;  // Bits per second
    Time packet_interval;       // Interval between packets (for periodic)
    uint32_t packet_size_bytes; // Packet size

    // Temporal characteristics
    double burstiness;  // Coefficient of variation (0=CBR, >1=bursty)
    Time on_time_mean;  // Mean ON time (for ON/OFF)
    Time off_time_mean; // Mean OFF time (for ON/OFF)

    // Energy characteristics
    double battery_capacity_mj; // Battery capacity in millijoules
    bool is_battery_powered;    // True if battery, false if plugged-in

    // QoS characteristics
    Time latency_requirement; // Maximum acceptable latency
    double priority_level;    // 0.0 (low) to 1.0 (critical)

    // Descriptive
    std::string device_name; // Descriptive name for logging

    // REHD-specific fields (only meaningful when device_class == DEVICE_REHD)
    RehdType rehd_type = REHD_T1; // Which sub-type this REHD is (drawn at init)
    energy::PowercastEnergyHarvester::CapacitorClass rehd_cap_class =
        energy::PowercastEnergyHarvester::CLASS_A;
    energy::PowercastEnergyHarvester::VoltageClass rehd_volt_class =
        energy::PowercastEnergyHarvester::CLASS_1;
};

// --- PREDEFINED DEVICE CONFIGURATIONS ---

// HETEROGENEITY WIDENING (2026-06-07): make the network LOPSIDED so scheduling/assignment is decisive.
// Spread widened to ~40 kbps (IoT mouse) .. 12 Mbps (Video elephant) ~= 300x, and the class-draw skewed toward mice (few elephants, many mice) — see the weighted draw in InitializeHeterogeneousNetwork.
// trafficScale is re-calibrated after this change (the old 1.9 anchor was for the balanced 64k-5M mix).
static const StaApplicationConfig IOT_SENSOR_CONFIG = {
    0,                 // sta_id (will be overwritten)
    DEVICE_IOT_SENSOR, // device_class
    40000,             // traffic_rate_bps: 40 Kbps (tiny telemetry mouse)
    MilliSeconds(100), // packet_interval: 100 ms
    100,               // packet_size_bytes: 100 bytes
    0.1,               // burstiness: Almost perfect CBR
    MilliSeconds(20),  // on_time_mean
    MilliSeconds(80),  // off_time_mean
    10.0,              // battery_capacity_mj: Small battery
    true,              // is_battery_powered: Yes
    MilliSeconds(500), // latency_requirement: 500 ms (IoT — moderate; swapped with REHD 2026-06-11
                       // so REHD is tightest)
    0.3,               // priority_level: Low priority
    "IoT-Sensor"       // device_name
};

static const StaApplicationConfig VIDEO_CAMERA_CONFIG = {
    0,                   // sta_id
    DEVICE_VIDEO_CAMERA, // device_class
    4000000,             // traffic_rate_bps: 4 Mbps (medium-heavy)
    MilliSeconds(3),     // packet_interval: 3 ms
    1500,                // packet_size_bytes: 1500 bytes
    0.3,                 // burstiness: Low variation (CBR-like)
    MilliSeconds(100),   // on_time_mean: Continuous
    MilliSeconds(1),     // off_time_mean: Minimal sleep
    1000.0,              // battery_capacity_mj: Large (AC-powered)
    false,               // is_battery_powered: No (AC)
    MilliSeconds(2000),  // latency_requirement: 2 s (Camera — elastic/buffered, NOT time-sensitive)
    0.6,                 // priority_level: Medium
    "Video-Camera"       // device_name
};

static const StaApplicationConfig VOICE_ASSISTANT_CONFIG = {
    0,                      // sta_id
    DEVICE_VOICE_ASSISTANT, // device_class
    64000,                  // traffic_rate_bps: 64 Kbps (G.711)
    MilliSeconds(20),       // packet_interval: 20 ms
    160,                    // packet_size_bytes: 160 bytes
    0.2,                    // burstiness: Low variation
    Seconds(2),             // on_time_mean: 2 second turns
    Seconds(3),             // off_time_mean: 3 second silence
    1000.0,                 // battery_capacity_mj: Large (AC)
    false,                  // is_battery_powered: No (AC)
    MilliSeconds(1000),     // latency_requirement: 1 s (Voice — moderate)
    0.9,                    // priority_level: High (real-time)
    "Voice-Assistant"       // device_name
};

static const StaApplicationConfig VIDEO_STREAMING_CONFIG = {
    0,                      // sta_id
    DEVICE_VIDEO_STREAMING, // device_class
    12000000,               // traffic_rate_bps: 12 Mbps (the ELEPHANT — hogs a shared SP)
    MilliSeconds(2),        // packet_interval: 2 ms
    1400,                   // packet_size_bytes: 1400 bytes
    2.0,                    // burstiness: High variation (bursty)
    Seconds(5),             // on_time_mean: 5 second watching
    Seconds(1),             // off_time_mean: 1 second pause
    1000.0,                 // battery_capacity_mj: Large (AC)
    false,                  // is_battery_powered: No (AC)
    MilliSeconds(3000), // latency_requirement: 3 s (Video — elastic streaming, NOT time-sensitive)
    0.7,                // priority_level: Medium-high
    "Video-Streaming"   // device_name
};

// Configuration structure to hold all simulation parameters
struct TwtSimulationConfig
{
    // Simulation parameters
    uint32_t simId = 10001;
    uint32_t randSeed = 9000;
    bool parallelSim = false;
    std::string scenario = "ns3UnilateralTwt";
    std::string currentsimId_string;

    // Timing
    // Default value; overwritten in constructor to (METRICS_START_TIME_BI * beaconInterval_s)
    double keepTrackOfMetricsFrom_ms = 0.0;
    double delayBinWidth_ms = 0.1; // Histogram bin width in milliseconds for FlowMonitor

    // Network
    // nStations = number of NON-REHD STAs (split uniform-random across IoT / Camera / Voice / Video classes at init).
    // nRehd    = number of REHD STAs (assigned class=REHD at init, then each REHD's sub-type drawn uniformly from REHD_TYPE_TEMPLATES).
    // Total active = nStations + nRehd, exposed via nTotal(). Must be <= MAX_NUM_STA.
    std::size_t nStations;
    std::size_t nRehd = 0; // default 0 = behave as the legacy 4-class network

    std::size_t nTotal() const
    {
        return nStations + nRehd;
    }

    // Multiplier on NON-REHD offered traffic rate = the SATURATION ANCHOR S_sat.
    // RE-CALIBRATED to 1.9 (load sweep 2026-06-04 at max-airtime action (2,0,0)): the saturation knee is ~2.0 (served ~1.0/expiry ~0 up to 1.75, served 0.64/expiry 0.42 by 2.25; queue-full rejection 0 everywhere -> deadline-expiry is the captured cost).
    // The segmented-coin regime swings each segment to 0.85*S_sat (under, served ~1.12) or 1.15*S_sat (over, served ~0.66) — see twt-constants.h N_TRAFFIC_SEGMENTS / SEG_*_MULT.
    // Overridable via --trafficScale / wrapper traffic_scale; dsweep --scale.
    double trafficScale = 1.9;

    double simulationTime_ms;
    double roomLength = DEFAULT_ROOM_LENGTH;
    uint32_t p2pLinkDelay_ms = 0;
    // Use Seconds() instead of MilliSeconds() to preserve fractional precision (102.4ms)
    // MilliSeconds(102.4) truncates to 102ms due to uint64_t conversion
    Time beaconInterval_s = Seconds(BEACON_INTERVAL_MS / 1000.0);
    std::string dlAckSeqType;
    uint32_t bsrLife_ms = 10.0; // BSR validity lifetime in milliseconds (10ms)
    uint32_t ampduLimitBytes = 20000;

    // Transport
    uint32_t payloadSize = 1500;

    // Logging
    bool enablePcap = false;
    bool enableStateLogs = false;
    bool enableFlowMon = false;
    bool recordApPhyState = false;
    bool linkStatusLogging = false;

    // TWT Configuration
    // Implicit TWT Operation:
    // - twtTriggerBased = false: STAs autonomously know wake times without explicit trigger frames
    // - All STAs in same TWT group wake simultaneously at twtNominalWakeDuration intervals
    // - Within each window: AP sends DL first, then STAs send UL (dynamic UL/DL split)
    // - Collision Behavior: With AP downlink present, first collision is nearly certain on UL (both STAs see clear channel simultaneously after AP finishes), resolved via exponential backoff
    // - maxMuSta = 1: No multi-user aggregation (sequential transmission per STA, not parallel MU)
    double twtSetupTimeBeaconIntervals = TWT_SETUP_TIME_BI;
    Time firstTwtSpStart;
    Time firstTwtSpOffsetFromBeacon = MilliSeconds(2.0); // 2 milliseconds
    bool twtTriggerBased = false;
    uint64_t maxMuSta = 1;
    Time twtWakeInterval;
    // Initial TWT wake duration: deliberately permissive (95 ms of a 102.4 ms BI ≈ 93% duty cycle).
    // Combined with all STAs at the same SP offset (see SetupTwtSchedule), this makes the pre-agent TWT effectively "no TWT" — EDCA contention with the entire BI minus a small guard.
    // The agent's first action overrides this at TWT_UPDATE_START_BI; until then the network runs unrestricted so apps and queues reach steady state before any decision is recorded.
    Time twtNominalWakeDuration = MilliSeconds(95.0);

    // Standalone "grouped sleep" test schedule (diagnostic; not used in dynamic mode).
    // When twtSleepGroups > 0, SetupTwtSchedule round-robins STAs into N groups with staggered SP offsets (BI/N apart) and a SHORT wake of twtGroupWakeMs each — so STAs sleep ~(1 - wake/BI) of every BI and REHDs actually harvest.
    // 0 = the permissive default above.
    std::size_t twtSleepGroups = 0;
    double twtGroupWakeMs = 10.0;

    // Phase-1 Power Delivery Window (PDW): when > 0, the AP broadcasts back-to-back dummy frames over [0, pdwDurationMs] ms of each beacon interval at its normal TX power (26 dBm EIRP) so REHDs asleep during the window harvest the RF.
    // TWT SP offsets are shifted to start after the PDW so REHDs doze through it.
    // Pure power delivery — broadcast frames are NOT tracked (they hit g_dlBroadcast* only, never the per-STA DL counters).
    // 0 = disabled (legacy behavior).
    double pdwDurationMs = 0.0;

    // App timing
    Time AppStartTimeMin;
    Time AppStartTimeMax;

    // Energy model
    std::unordered_map<std::string, double> TI_currentModel_mA = {
        {"IDLE", PHY_STATE_IDLE_MA},
        {"CCA_BUSY", PHY_STATE_CCA_BUSY_MA},
        {"RX", PHY_STATE_RX_MA},
        {"TX", PHY_STATE_TX_MA},
        {"SLEEP", PHY_STATE_SLEEP_MA}};

    // Constructor to initialize derived values
    TwtSimulationConfig()

    {
        // Initialize number of stations with default value
        nStations = ACTIVE_NUM_STA;

        twtWakeInterval = beaconInterval_s;
        firstTwtSpStart = twtSetupTimeBeaconIntervals * beaconInterval_s;
        keepTrackOfMetricsFrom_ms = (METRICS_START_TIME_BI * beaconInterval_s).GetMilliSeconds();
        // Total simulation duration in milliseconds
        simulationTime_ms = (SIMULATION_DURATION_BI * beaconInterval_s).GetMilliSeconds();
        // 3ms offset is to avoid collision with beacon.
        AppStartTimeMin = (APP_START_TIME_MIN_BI * beaconInterval_s) + MilliSeconds(3.0);
        AppStartTimeMax = (APP_START_TIME_MAX_BI * beaconInterval_s) + MilliSeconds(3.0);
    }

    void GenerateSimIdString();
    void WriteDeviceClassAssignments();

    // --- HETEROGENEOUS NETWORK CONFIGURATION ---

    // Device class assignments for each STA (size = nStations)
    std::vector<StaApplicationConfig> sta_app_configs;

    // Enable heterogeneous traffic generation
    bool enable_heterogeneous_traffic = true;

    /**
     * Initialize heterogeneous network with 4-class distribution
     * Divides STAs into 4 equal groups: IoT, Camera, Voice, Video
     */
    void InitializeHeterogeneousNetwork();

    /**
     * Get device class name for logging
     */
    std::string GetDeviceClassName(DeviceClass dc) const;
};

// Helper class to encapsulate network setup
class TwtNetworkSetup
{
  public:
    TwtNetworkSetup(const TwtSimulationConfig& config);

    void CreateNodes();
    void ConfigureWifi();
    void SetupMobility();
    void ConfigureInternet();
    void SetupApplications();
    void SetupTwtSchedule();
    void EnablePcap();

    // Phase 1: install BasicEnergySource + PowercastEnergyHarvester on each STA and connect PHY RX/TX Begin/End callbacks so RF power is harvested and TX power is consumed in real time.
    // Call AFTER ConfigureWifi() (needs PHY) and BEFORE SetupApplications() (so harvesters are tracking from the first TX).
    // Energy state is NOT yet plumbed into EnvStruct or the reward — that's phase 2.
    void SetupEnergyHarvesting(const std::string& configFile = std::string());

    // Print a per-STA energy summary line at the end of the run.
    // Useful as a smoke-test signal that the harvesters are actually charging/discharging.
    void LogFinalEnergyStats() const;

    // Per-STA harvester accessor (returns nullptr if SetupEnergyHarvesting was never called, or for staId out of range).
    // Used by phase-2 metrics code to read Vcap / available / harvested / consumed for the EnvStruct.
    Ptr<energy::PowercastEnergyHarvester> GetHarvester(uint32_t staId) const;

    // Cumulative count of TX events the harvester refused to sustain (rate-limited by CanSustainTransmission).
    // Backed by the file-scope g_phUnpoweredTxEvents vector in twt-simulation-config.cc; returns 0 for non-REHD STA indices.
    uint64_t GetUnpoweredTxEvents(uint32_t staId) const;

    // Current MAC backlog (packets queued across all ACs) for STA index staId.
    // Used by the diagnostic Vcap/queue logger to watch REHD queues build/drain.
    uint32_t GetStaQueueDepth(uint32_t staId) const;

    // ORACLE: cumulative UL packets that expired (MaxDelay) in the REHD's queue before being sent.
    // AP-invisible; for reward shaping / evaluation only.
    uint64_t GetExpiredPackets(uint32_t staId) const;

    // Release all harvester Ptr<> references (member vector + file-scope state used by the PHY callbacks).
    // MUST be called BEFORE Simulator::Destroy() — otherwise the file-scope statics are destroyed after NS-3 globals at process exit, and the Ptr destructors hit a UAF on Time/Object state.
    void TeardownEnergyHarvesting();

    // Dynamic-env hooks (no-op when ENABLE_DYNAMIC_* = 0).
    // Call ScheduleDynamicMobility() AFTER SetupMobility() and ScheduleDynamicTraffic() AFTER SetupApplications() — both populate recurring per-STA events and depend on the prior init step having run.
    void ScheduleDynamicMobility();
    void ScheduleDynamicTraffic();

    // Getters
    NodeContainer GetStaNodes() const
    {
        return wifiStaNodes;
    }

    NodeContainer GetApNodes() const
    {
        return wifiApNodes;
    }

    Ptr<Node> GetServerNode() const
    {
        return p2pServerNode;
    }

    ApplicationContainer GetServerApps() const
    {
        return serverApp;
    }

    Ipv4InterfaceContainer GetStaInterfaces() const
    {
        return staNodeInterfaces;
    }

    Ptr<Node> GetApNode() const
    {
        return ApNode;
    }

    // Dynamic TWT reconfiguration
    void ApplyTWTSchedule(const ActionStruct& action);
    Ptr<WifiMac> GetStaMac(uint32_t staId) const;
    Ptr<WifiMac> GetApMac() const;

    // Get last applied action info
    uint32_t GetLastNumActiveTwtGroups() const
    {
        return m_lastNumActiveTwtGroups;
    }

    // Get STA's assigned TWT group from last applied action
    uint8_t GetStaTwtGroup(uint32_t staId) const
    {
        if (staId < MAX_NUM_STA)
        {
            return m_lastStaGroupAssignments[staId];
        }
        return 0;
    }

    // Get TWT group config from last applied action
    const TwtGroupConfig& GetTwtGroupConfig(uint8_t groupId) const
    {
        return m_lastTwtGroupConfigs[groupId < MAX_NUM_TWT_GROUPS ? groupId : 0];
    }

    // Check if STA has TWT enabled
    bool IsStaTwtEnabled(uint32_t staId) const
    {
        if (staId < MAX_NUM_STA)
        {
            return m_lastStaTwtEnabled[staId];
        }
        return false;
    }

  private:
    const TwtSimulationConfig& m_config;

    // Network containers
    NodeContainer wifiApNodes;
    NodeContainer wifiStaNodes;
    NodeContainer p2pServerNodes;
    Ptr<Node> p2pServerNode;
    Ptr<Node> ApNode;

    // Device containers
    NetDeviceContainer apDevice;
    NetDeviceContainer staDevices;
    NetDeviceContainer p2pdevices;

    // Interface containers
    Ipv4InterfaceContainer staNodeInterfaces;
    Ipv4InterfaceContainer apNodeInterface;
    Ipv4InterfaceContainer p2pNodeInterfaces;

    // Application container
    ApplicationContainer serverApp;

    // Per-STA PowerCast harvester pointers (size = nStations, indexed by STA index 0..nStations-1).
    // Empty until SetupEnergyHarvesting() is called.
    std::vector<Ptr<energy::PowercastEnergyHarvester>> m_harvesters;
    // The BasicEnergySource container is intentionally NOT stored as a member.
    // EnergySourceContainer derives from ns3::Object; holding it by value and destroying it after Simulator::Destroy() (in ~TwtNetworkSetup) walked a freed m_aggregates buffer → SIGSEGV.
    // The sources live on the nodes via aggregation for the whole run; the harvesters are tracked above.

    // Track last applied TWT action
    uint32_t m_lastNumActiveTwtGroups = 0;
    uint8_t m_lastStaGroupAssignments[MAX_NUM_STA] = {0};          // STA -> Group mapping
    bool m_lastStaTwtEnabled[MAX_NUM_STA] = {false};               // STA TWT enabled flag
    TwtGroupConfig m_lastTwtGroupConfigs[MAX_NUM_TWT_GROUPS] = {}; // Group configs

    // Helpers
    SpectrumWifiPhyHelper phy;
    WifiMacHelper mac;
    WifiHelper wifi;

    // Random variable for app start times (heterogeneous traffic)
    Ptr<UniformRandomVariable> appStartTimeRand;

    // Per-STA client-app handles, populated by the Create*Application helpers.
    // Used by the dynamic-traffic scheduler to mutate DataRate/OnTime/OffTime mid-episode without tearing down and reinstalling apps.
    std::vector<Ptr<Application>> m_perStaClientApps;

    // Per-STA RNG handles for the dynamic-env updaters (independent NS-3 streams via AssignStreams so episodes stay reproducible per seed).
    std::vector<Ptr<ExponentialRandomVariable>> m_trafficUpdateRand; // inter-arrival
    std::vector<Ptr<UniformRandomVariable>> m_trafficRateRand;
    std::vector<Ptr<UniformRandomVariable>> m_trafficOnRand;
    std::vector<Ptr<UniformRandomVariable>> m_trafficOffRand;
    std::vector<Ptr<UniformRandomVariable>> m_mobilitySpeedRand;
    std::vector<Ptr<UniformRandomVariable>> m_mobilityDirRand;

    // Segmented-coin dynamic traffic (2026-06-04): m_segCoinRand tosses over/under each segment; m_segMult is the resulting multiplier (SEG_OVER_MULT / SEG_UNDER_MULT) applied on top of trafficScale in UpdateTrafficForSta.
    // Default 1.0 = saturation anchor before any segment fires.
    Ptr<UniformRandomVariable> m_segCoinRand;
    double m_segMult = 1.0;

    void RandomWifiStart(Ptr<WifiPhy> phy);
    void initiateUnicastTwtAtAp(Ptr<WifiMac> apMac,
                                Mac48Address staMacAddress,
                                uint8_t flowId,
                                Time twtWakeInterval,
                                Time twtNominalWakeDuration,
                                Time nextTwtOffsetFromNextBeacon);
    void initiateTwtAtSta(Ptr<WifiMac> staMac,
                          Ptr<WifiMac> apMac,
                          uint8_t flowId,
                          Time twtWakeInterval,
                          Time twtNominalWakeDuration,
                          Time nextTwtOffsetFromNextBeacon);

    // --- HETEROGENEOUS APPLICATION CREATION HELPERS ---
    void CreateIoTSensorApplication(uint32_t sta_index,
                                    const StaApplicationConfig& config,
                                    const Address& server_addr);
    void CreateVideoCameraApplication(uint32_t sta_index,
                                      const StaApplicationConfig& config,
                                      const Address& server_addr);
    void CreateVoiceAssistantApplication(uint32_t sta_index,
                                         const StaApplicationConfig& config,
                                         const Address& server_addr);
    void CreateVideoStreamingApplication(uint32_t sta_index,
                                         const StaApplicationConfig& config,
                                         const Address& server_addr);
    void CreateRehdSensorApplication(uint32_t sta_index,
                                     const StaApplicationConfig& config,
                                     const Address& server_addr);

    // Power Delivery Window (PDW): the AP acts as a dedicated RF power beacon over [PDW_BEACON_LEAD, pdw_end] of every BI so sleeping REHDs harvest.
    // Delivered DIRECTLY (no packetized frames, no WiFi MAC) — each beacon arms the window via the g_onApBeaconTx hook, which schedules DeliverPdwPower at PDW_BEACON_LEAD after the real beacon TX.
    // DeliverPdwPower credits every currently-asleep REHD the AP's RF energy for the full window duration (pdw_end - PDW_BEACON_LEAD), via the same analytical Friis path as the comms harvest.
    // This is the continuous-RF ideal: 100% window occupancy, no CSMA gaps, and zero leak (an energy credit cannot spill onto the channel or collide with the next SP).
    // pdw_end (= m_pdwDurationMs) is set per agent action by ApplyTWTSchedule (dynamic) or --pdwDurationMs (static).
    void SetupPowerDeliveryWindow(); // register the beacon hook that arms per-BI power delivery
    void DeliverPdwPower();          // credit one BI's RF energy to every asleep REHD (analytical)

    double m_pdwDurationMs = 0.0; // current PDW end time pdw_end (ms); <= PDW_BEACON_LEAD = no PDW

    // Dynamic-env helpers — see ScheduleDynamicMobility/Traffic above.
    void UpdateStaDirection(uint32_t staId);
    void EnforceBuffer();
    void UpdateTrafficForSta(uint32_t staId);
    void UpdateTrafficSegment(uint32_t segIdx);
};

} // namespace ns3

#endif // TWT_SIMULATION_CONFIG_H