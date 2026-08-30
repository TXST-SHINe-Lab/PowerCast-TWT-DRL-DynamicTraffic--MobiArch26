// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

#include "twt-simulation-config.h"

#include "ph-deployment-helper.h" // PowercastEnergyHarvesterHelper (phase 1)
#include "twt-trace-callbacks.h"  // for g_quietMode

#include "ns3/config.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/constant-velocity-mobility-model.h"
#include "ns3/data-rate.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/node-list.h" // NodeList::GetNode (transmitter mobility for harvest)
#include "ns3/on-off-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/propagation-delay-model.h" // ConstantSpeedPropagationDelayModel
#include "ns3/propagation-loss-model.h"  // FriisPropagationLossModel
#include "ns3/rng-seed-manager.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/udp-client-server-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace
{
// --- PHASE 1 — PowerCast energy harvesting state + PHY callbacks ---
//
// Free callbacks (required by NS-3 Config::Connect signatures) need access to per-STA harvester pointers.
// We keep them in a file-scope vector populated by TwtNetworkSetup::SetupEnergyHarvesting().
// When the simulation ends and the next worker starts in a *different* process, this state is fresh — no cross-episode contamination.
std::vector<ns3::Ptr<ns3::energy::PowercastEnergyHarvester>> g_phHarvesters;
std::vector<uint64_t> g_phHarvestEvents; // per-REHD count of harvest accruals (sleep windows)
std::vector<uint64_t> g_phTxEvents;      // per-REHD count of OnPhyTxEnd hits (own TX)

// --- Time-switching harvest model (2026-05-28) ---
// A single-antenna REHD is an RF switch: transceiver (RX/TX/sense) XOR rectenna (harvest).
// So it harvests ONLY while its PHY is asleep (TWT doze).
// We do NOT read the REHD's PHY for RX power (it's off) — instead we compute the antenna RX power analytically from each transmitter using the SAME propagation-loss model the channel uses: P_rx = CalcRxPower(txConducted + txGain) + rectennaGain.
// On every network TX, each REHD currently in PHY SLEEP harvests that frame's ambient RF for its duration.
//
// HARVEST-SOURCE SEAM: the desired future model measures RX power intrinsically at the REHD antenna (a co-located always-listening power-meter SpectrumPhy) so a changed channel model / other RF sources / nearby APs are captured automatically.
// When that lands, only the P_rx computation in OnPhyTxEnd's harvest loop changes; the sleep-gating + accrual stay the same.
ns3::Ptr<ns3::PropagationLossModel> g_propLoss; // shared Friis model
std::vector<ns3::Ptr<ns3::WifiPhy>>
    g_rehdPhy; // per-staIdx REHD PHY (sleep query); nullptr otherwise
std::vector<ns3::Ptr<ns3::MobilityModel>>
    g_rehdMob; // per-staIdx REHD mobility (CalcRxPower); nullptr otherwise

// Energy-gated TX (Phase 2): a REHD whose capacitor hit Vmin is "locked" — the hysteresis keeps TX disabled until a full recharge to Vmax.
// While locked we BLOCK its uplink MAC queue (frames to the AP) so packets stay queued instead of going out — we do NOT touch PHY sleep (TWT still owns that, so the REHD keeps dozing/harvesting on its schedule, and there's no frame-exchange or dual-controller hazard).
// On recovery to Vmax we unblock and the backlog drains during the next SPs.
// Blocking holds packets in place; it does not drop them.
std::vector<bool> g_rehdEnergyLocked;          // per-staIdx; true while depleted/recharging
std::vector<ns3::Ptr<ns3::WifiMac>> g_rehdMac; // per-staIdx REHD MAC (queue block/unblock)
ns3::Mac48Address g_apMacAddr;                 // AP MAC — REHD uplink RA to block

// ORACLE metric: per-REHD count of UL packets that aged out of the MAC queue (MaxDelay expiry) before they could be sent.
// The AP cannot observe this (it's STA-internal); it's recorded only for reward shaping / evaluation.
// A high count = the REHD is energy-starved (blocked too long to drain its backlog in time).
std::vector<uint64_t> g_rehdExpiredPkts;

void
OnRehdExpired(uint32_t staIdx, ns3::Ptr<const ns3::WifiMpdu>)
{
    if (staIdx < g_rehdExpiredPkts.size())
    {
        g_rehdExpiredPkts[staIdx]++;
    }
}

struct PhyEvent
{
    ns3::Time startTime;
    double powerDbm;
    uint32_t packetSize;
};

// Ongoing TX keyed by transmitter nodeId — tracked for ALL nodes (the AP and every STA) because any of them is an ambient RF source for sleeping REHDs.
std::map<uint32_t, PhyEvent> g_phOngoingTxEvents;

// Helper: parse NodeList path "/NodeList/<id>/..." → node id.
uint32_t
ParseNodeIdFromContext(const std::string& context)
{
    std::string s = context.substr(10); // skip "/NodeList/"
    size_t pos = s.find('/');
    return (pos == std::string::npos) ? std::stoul(s) : std::stoul(s.substr(0, pos));
}

// In TWT setup, node 0 = AP, nodes [1..nStations] = STAs.
// Harvesters live on STAs only, so staIdx = nodeId - 1 (valid iff in [1..nStations]).
inline bool
ResolveStaIdx(uint32_t nodeId, uint32_t& staIdx)
{
    if (nodeId == 0 || nodeId > g_phHarvesters.size())
    {
        return false;
    }
    staIdx = nodeId - 1;
    return g_phHarvesters[staIdx] != nullptr;
}

// Reconcile a REHD's uplink queue block with its energy audit.
// Call after any capacitor update (consume / idle drain / harvest).
// No hysteresis: the gate is simply "can the cap pay for one more packet (staying >= 0.8*Vmin)?":
//   - can't afford & not locked  → lock + BLOCK uplink queue (packets wait)
//   - can afford   & locked       → unlock + UNBLOCK (drain as many as energy allows)
// So a depleted REHD unblocks the instant it harvests one packet's worth — and the MAC then drains greedily until the next packet is unaffordable (re-block in the following OnPhyTxEnd).
// Net effect = the "energy audit" send-what-you-can.
// Audit uses a representative REHD packet airtime (~400 us at REHD TX power).
void
ReconcileRehdLock(uint32_t staIdx)
{
    if (staIdx >= g_phHarvesters.size() || !g_phHarvesters[staIdx] || !g_rehdMac[staIdx])
    {
        return;
    }
    const bool txOk =
        g_phHarvesters[staIdx]->CanSustainTransmission(REHD_TX_POWER_DBM, ns3::MicroSeconds(400));
    if (!txOk && !g_rehdEnergyLocked[staIdx])
    {
        g_rehdEnergyLocked[staIdx] = true;
        g_rehdMac[staIdx]->BlockUnicastTxOnLinks(
            ns3::WifiQueueBlockedReason::POWER_SAVE_MODE, g_apMacAddr, {0});
    }
    else if (txOk && g_rehdEnergyLocked[staIdx])
    {
        g_rehdEnergyLocked[staIdx] = false;
        g_rehdMac[staIdx]->UnblockUnicastTxOnLinks(
            ns3::WifiQueueBlockedReason::POWER_SAVE_MODE, g_apMacAddr, {0});
    }
}

// Harvest is NO LONGER driven by the REHD's own RX (PhyRxBegin/End).
// A time-switching harvester cannot harvest while its transceiver is active — see the harvest-source comment above.
// Harvest now accrues in OnPhyTxEnd for every REHD that is asleep when an ambient transmission occurs.

void
OnPhyTxBegin(std::string context, ns3::Ptr<const ns3::Packet> packet, double txPowerW)
{
    // Record TX for ANY transmitter (AP or any STA) — all are ambient RF sources that a sleeping REHD can harvest.
    // Self-consumption (REHD transmitting) is resolved at TX end.
    uint32_t nodeId = ParseNodeIdFromContext(context);
    if (txPowerW <= 0.0)
    {
        return;
    }
    double txPowerDbm = 10.0 * std::log10(txPowerW * 1000.0);
    g_phOngoingTxEvents[nodeId] = PhyEvent{ns3::Simulator::Now(), txPowerDbm, packet->GetSize()};
}

// Per-STA count of TX events the harvester could not sustain.
// The energy-gate (ReconcileRehdLock blocks a depleted REHD's uplink queue to the AP) normally stops the MAC from dequeuing before this fires, so it should rarely increment — it only catches a boundary TX that drains the capacitor to its floor between gate re-evaluations.
// Such a TX skips its capacitor debit (the audit clamps).
std::vector<uint64_t> g_phUnpoweredTxEvents;

void
OnPhyTxEnd(std::string context, ns3::Ptr<const ns3::Packet> packet)
{
    uint32_t nodeId = ParseNodeIdFromContext(context);
    auto it = g_phOngoingTxEvents.find(nodeId);
    if (it == g_phOngoingTxEvents.end())
    {
        return;
    }
    const ns3::Time dur = ns3::Simulator::Now() - it->second.startTime;
    const double txPowerDbm = it->second.powerDbm; // conducted TX power of this frame
    g_phOngoingTxEvents.erase(it);

    // (1) Self-consumption: if the transmitter is itself a REHD, it spends its own capacitor energy to radiate this frame (gated by CanSustain).
    uint32_t txStaIdx;
    if (ResolveStaIdx(nodeId, txStaIdx))
    {
        if (g_phHarvesters[txStaIdx]->CanSustainTransmission(txPowerDbm, dur))
        {
            g_phHarvesters[txStaIdx]->SetConsumedEnergy(txPowerDbm, dur);
        }
        else if (txStaIdx < g_phUnpoweredTxEvents.size())
        {
            g_phUnpoweredTxEvents[txStaIdx]++;
        }
        g_phTxEvents[txStaIdx]++;
        ReconcileRehdLock(txStaIdx); // this TX may have hit Vmin → lock + sleep
    }

    // (2) Time-switching harvest: every REHD whose PHY is asleep RIGHT NOW has its antenna on the rectenna and harvests this frame's ambient RF.
    //     The RX power is computed analytically (the sleeping PHY can't report it) via the shared propagation model + EIRP/rectenna gains, then fed to the harvester's existing efficiency-curve accumulator.
    //     A transmitting REHD is in TX state (not SLEEP) so it never harvests its own frame.
    if (g_propLoss)
    {
        const double txGainDbi = (nodeId == 0) ? AP_TX_GAIN_DBI : 0.0; // only the AP has TX gain
        ns3::Ptr<ns3::Node> txNode = ns3::NodeList::GetNode(nodeId);
        ns3::Ptr<ns3::MobilityModel> txMob =
            txNode ? txNode->GetObject<ns3::MobilityModel>() : nullptr;
        if (txMob)
        {
            for (uint32_t i = 0; i < g_phHarvesters.size(); ++i)
            {
                if (!g_phHarvesters[i] || !g_rehdPhy[i] || !g_rehdMob[i] ||
                    !g_rehdPhy[i]->IsStateSleep())
                {
                    continue;
                }
                const double rxDbm =
                    g_propLoss->CalcRxPower(txPowerDbm + txGainDbi, txMob, g_rehdMob[i]) +
                    REHD_RX_GAIN_DBI;
                g_phHarvesters[i]->SetHarvestedEnergy(rxDbm, dur);
                g_phHarvestEvents[i]++;
                ReconcileRehdLock(i); // may have recharged to Vmax → release lock
            }
        }
    }
    // PDW power delivery is NOT driven from here — it's a direct per-BI energy credit (TwtNetworkSetup::DeliverPdwPower), not packetized AP frames.
    // This hook only harvests INCIDENTAL ambient RF (real comms TX) while a REHD sleeps.
}

// Per-state REHD power draw.
// The PHY "State" trace reports each completed state interval (start, duration, state).
// REHD device-power budget (LP-radio class, twt-constants.h): RX/decode 10 mW, IDLE/CCA-busy 1 µW, SLEEP 1 µW deep-sleep leakage.
// (TX is handled in OnPhyTxEnd: radiated RF / PA_EFFICIENCY.)
// SLEEP also HARVESTS via the rectenna (credited in OnPhyTxEnd / DeliverPdwPower); the 1 µW leakage here runs in parallel, so net sleep energy = harvest − leakage.
void
OnRehdPhyState(std::string context, ns3::Time, ns3::Time duration, ns3::WifiPhyState state)
{
    uint32_t nodeId = ParseNodeIdFromContext(context);
    uint32_t staIdx;
    if (!ResolveStaIdx(nodeId, staIdx))
    {
        return;
    }
    // Awake-state housekeeping drain for the completed interval (IDLE/CCA/RX).
    double powerW = 0.0;
    if (state == ns3::WifiPhyState::IDLE || state == ns3::WifiPhyState::CCA_BUSY)
    {
        powerW = REHD_IDLE_POWER_W;
    }
    else if (state == ns3::WifiPhyState::RX)
    {
        powerW = REHD_RX_POWER_W;
    }
    else if (state == ns3::WifiPhyState::SLEEP)
    {
        powerW = REHD_SLEEP_POWER_W; // deep-sleep leakage (rectenna harvest credited separately)
    }
    if (powerW > 0.0)
    {
        g_phHarvesters[staIdx]->ConsumeDcEnergy(powerW * duration.GetSeconds());
        ReconcileRehdLock(staIdx); // drain may have hit Vmin → lock (block uplink queue)
    }
}

// Parse the harvester config file.
// Two passes:
//   pass 1: scan header comments for "# CAP_A <µF>" / "# CAP_B <µF>" / "# CAP_C <µF>" and call PowercastEnergyHarvester::SetCapacitorValue (mandatory — the harvester has no hardcoded defaults).
//   pass 2: parse "<nodeId> <capLetter> <voltDigit>" lines into the helper.
// Returns the number of node-config lines parsed (so caller can warn if 0).
uint32_t
LoadPhConfigFromFile(const std::string& path, ns3::energy::PowercastEnergyHarvesterHelper& helper)
{
    using HW = ns3::energy::PowercastEnergyHarvester;
    std::ifstream f(path);
    if (!f.is_open())
    {
        return 0;
    }
    std::string line;
    while (std::getline(f, line))
    {
        if (line.find("# CAP_A ") != std::string::npos)
        {
            std::istringstream iss(line.substr(line.find("CAP_A")));
            std::string tag;
            double uF;
            if (iss >> tag >> uF)
            {
                HW::SetCapacitorValue(HW::CLASS_A, uF / 1e6);
            }
        }
        else if (line.find("# CAP_B ") != std::string::npos)
        {
            std::istringstream iss(line.substr(line.find("CAP_B")));
            std::string tag;
            double uF;
            if (iss >> tag >> uF)
            {
                HW::SetCapacitorValue(HW::CLASS_B, uF / 1e6);
            }
        }
        else if (line.find("# CAP_C ") != std::string::npos)
        {
            std::istringstream iss(line.substr(line.find("CAP_C")));
            std::string tag;
            double uF;
            if (iss >> tag >> uF)
            {
                HW::SetCapacitorValue(HW::CLASS_C, uF / 1e6);
            }
        }
        if (!line.empty() && line[0] != '#')
        {
            f.clear();
            f.seekg(0);
            break;
        }
    }
    uint32_t parsed = 0;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream iss(line);
        uint32_t nodeId;
        char cap, volt;
        if (!(iss >> nodeId >> cap >> volt))
        {
            continue;
        }
        HW::CapacitorClass cc;
        switch (cap)
        {
        case 'A':
            cc = HW::CLASS_A;
            break;
        case 'B':
            cc = HW::CLASS_B;
            break;
        case 'C':
            cc = HW::CLASS_C;
            break;
        default:
            continue;
        }
        HW::VoltageClass vc;
        switch (volt)
        {
        case '1':
            vc = HW::CLASS_1;
            break;
        case '2':
            vc = HW::CLASS_2;
            break;
        case '3':
            vc = HW::CLASS_3;
            break;
        default:
            continue;
        }
        helper.ConfigureNode(nodeId, cc, vc);
        parsed++;
    }
    return parsed;
}

} // anonymous namespace

namespace ns3
{

// Generate simulation ID string with timestamp
void
TwtSimulationConfig::GenerateSimIdString()
{
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_time_t);

    // Format: YYYYMMDD_HHMMSS (matching Python controller format)
    std::stringstream timestampStream;
    timestampStream << std::setfill('0') << std::setw(4) << (now_tm.tm_year + 1900) << std::setw(2)
                    << (now_tm.tm_mon + 1) << std::setw(2) << now_tm.tm_mday << "_" << std::setw(2)
                    << now_tm.tm_hour << std::setw(2) << now_tm.tm_min << std::setw(2)
                    << now_tm.tm_sec;

    currentsimId_string = timestampStream.str();
}

// Write device class assignments to file
void
TwtSimulationConfig::WriteDeviceClassAssignments()
{
    // Create filename with simulation ID
    std::string filename =
        TwtResultsPath("data-log/device_class_assignments_") + currentsimId_string + ".txt";
    std::ofstream outFile(filename);

    if (!outFile.is_open())
    {
        std::cerr << "Warning: Could not open file " << filename << " for writing" << std::endl;
        return;
    }

    outFile << "Device Class Assignments - Simulation: " << currentsimId_string << std::endl;
    outFile << "======================================" << std::endl;
    outFile << std::endl;

    for (const auto& config : sta_app_configs)
    {
        outFile << "STA " << config.sta_id << ": " << config.device_name << std::endl;
        outFile << "  Class: " << GetDeviceClassName(config.device_class) << std::endl;
        outFile << "  Rate: " << (config.traffic_rate_bps / 1e6) << " Mbps" << std::endl;
        outFile << "  Packet Size: " << config.packet_size_bytes << " bytes" << std::endl;
        outFile << "  Battery Capacity: " << config.battery_capacity_mj << " mJ" << std::endl;
        outFile << "  Battery Powered: " << (config.is_battery_powered ? "Yes" : "No") << std::endl;
        outFile << "  Priority Level: " << config.priority_level << std::endl;
        outFile << std::endl;
    }

    outFile.close();
    if (!g_quietMode)
    {
        std::cout << "Device class assignments written to " << filename << std::endl;
    }
}

void
TwtSimulationConfig::InitializeHeterogeneousNetwork()
{
    sta_app_configs.clear();
    const std::size_t total = nTotal();
    sta_app_configs.reserve(total);

    // RNG streams (offset well clear of WiFi assignment block and the mobility / traffic-update streams in twt-simulation-config.cc).
    Ptr<UniformRandomVariable> classRand = CreateObject<UniformRandomVariable>();
    classRand->SetAttribute("Min", DoubleValue(0.0));
    classRand->SetAttribute("Max", DoubleValue(0.999999)); // weighted draw in [0,1)
    classRand->SetStream(RNG_INITIAL_STREAM_ID + 3000);

    Ptr<UniformRandomVariable> rehdTypeRand = CreateObject<UniformRandomVariable>();
    rehdTypeRand->SetAttribute("Min", DoubleValue(0.0));
    rehdTypeRand->SetAttribute("Max", DoubleValue(static_cast<double>(REHD_TYPE_COUNT) - 0.001));
    rehdTypeRand->SetStream(RNG_INITIAL_STREAM_ID + 3001);

    Ptr<UniformRandomVariable> jitterRand = CreateObject<UniformRandomVariable>();
    jitterRand->SetStream(RNG_INITIAL_STREAM_ID + 3002);

    // Track count per class for naming.
    uint32_t classCount[5] = {0, 0, 0, 0, 0}; // IoT, Camera, Voice, Video, REHD

    // First nStations slots: uniform-random from the 4 non-REHD classes.
    for (uint32_t i = 0; i < nStations; i++)
    {
        StaApplicationConfig config;
        // HETEROGENEITY (2026-06-07): skew the mix toward MICE so the network is lopsided — IoT 40% / Voice 25% / Camera 20% / Video 15% (few elephants, many mice => isolating the elephant / slotting the mice is decisive).
        // classIndex maps to the switch below: 0=IoT,1=Camera,2=Voice,3=Video.
        double cr = classRand->GetValue();
        uint32_t classIndex;
        if (cr < 0.40)
        {
            classIndex = 0; // IoT    (mouse)        40%
        }
        else if (cr < 0.65)
        {
            classIndex = 2; // Voice  (urgent mouse) 25%
        }
        else if (cr < 0.85)
        {
            classIndex = 1; // Camera (medium)       20%
        }
        else
        {
            classIndex = 3; // Video  (elephant)     15%
        }
        switch (classIndex)
        {
        case 0:
            config = IOT_SENSOR_CONFIG;
            config.device_name = "Sensor-" + std::to_string(classCount[0]++);
            break;
        case 1:
            config = VIDEO_CAMERA_CONFIG;
            config.device_name = "Camera-" + std::to_string(classCount[1]++);
            break;
        case 2:
            config = VOICE_ASSISTANT_CONFIG;
            config.device_name = "Voice-" + std::to_string(classCount[2]++);
            break;
        case 3:
        default:
            config = VIDEO_STREAMING_CONFIG;
            config.device_name = "Video-" + std::to_string(classCount[3]++);
            break;
        }
        // Scale NON-REHD offered load by trafficScale (1.0 = default).
        // OnOff classes (IoT/Voice/Video) use traffic_rate_bps (DataRate); CBR (Camera) uses packet_interval — scale both so every class's effective rate is multiplied.
        // REHDs (the next loop) are deliberately untouched.
        if (trafficScale > 0.0 && trafficScale != 1.0)
        {
            config.traffic_rate_bps = static_cast<uint32_t>(config.traffic_rate_bps * trafficScale);
            config.packet_interval = Seconds(config.packet_interval.GetSeconds() / trafficScale);
        }
        config.sta_id = i;
        sta_app_configs.push_back(config);
    }

    // Remaining nRehd slots: REHD class + uniform-random sub-type + per-instance traffic-param sampling within the sub-type's range.
    // Hardware (cap, voltage) class comes straight from the template — no jitter.
    for (uint32_t k = 0; k < nRehd; k++)
    {
        const uint32_t i = static_cast<uint32_t>(nStations) + k;
        uint32_t typeIdx = static_cast<uint32_t>(rehdTypeRand->GetValue());
        if (typeIdx >= REHD_TYPE_COUNT)
        {
            typeIdx = REHD_TYPE_COUNT - 1; // defensive
        }
        const RehdTypeTemplate& tpl = REHD_TYPE_TEMPLATES[typeIdx];

        // Draw traffic params uniformly within the template range — once, frozen for the episode.
        const uint32_t rate_bps =
            static_cast<uint32_t>(jitterRand->GetValue(tpl.rate_min_bps, tpl.rate_max_bps));
        const uint32_t pkt_size = static_cast<uint32_t>(
            jitterRand->GetValue(tpl.pkt_size_min_bytes, tpl.pkt_size_max_bytes));
        const double interval_s = jitterRand->GetValue(tpl.interval_min_s, tpl.interval_max_s);

        StaApplicationConfig config{};
        config.sta_id = i;
        config.device_class = DEVICE_REHD;
        config.traffic_rate_bps = rate_bps;
        config.packet_interval = Seconds(interval_s);
        config.packet_size_bytes = pkt_size;
        config.burstiness = 0.1;                // near-CBR within ON period
        config.on_time_mean = MilliSeconds(50); // brief burst
        config.off_time_mean = Seconds(std::max(0.05, interval_s - 0.05));
        config.battery_capacity_mj = 0.0; // harvester-only
        config.is_battery_powered = true;
        // REHD: 200 ms — TIGHTEST (swapped w/ IoT 2026-06-11): energy-harvesting sensor data, but energy-constrained -> tight-but-achievable ONLY with timely PDW energy + prompt airtime (THE protection challenge); else it expires (data loss).
        config.latency_requirement = MilliSeconds(200);
        config.priority_level = tpl.priority_level;
        config.device_name = std::string(tpl.name) + "-" + std::to_string(classCount[4]++);
        config.rehd_type = static_cast<RehdType>(typeIdx);
        config.rehd_cap_class = tpl.cap_class;
        config.rehd_volt_class = tpl.volt_class;
        sta_app_configs.push_back(config);
    }

    if (!g_quietMode)
    {
        std::cout << "Heterogeneous network initialized: " << total << " STAs (" << nStations
                  << " non-REHD + " << nRehd << " REHD)" << std::endl;
        std::cout << "  IoT Sensors: " << classCount[0] << ", Video Cameras: " << classCount[1]
                  << ", Voice Assistants: " << classCount[2]
                  << ", Video Streaming: " << classCount[3] << ", REHDs: " << classCount[4]
                  << std::endl;
    }
}

// Get device class name for logging
std::string
TwtSimulationConfig::GetDeviceClassName(DeviceClass dc) const
{
    switch (dc)
    {
    case DEVICE_IOT_SENSOR:
        return "IoT-Sensor";
    case DEVICE_VIDEO_CAMERA:
        return "Video-Camera";
    case DEVICE_VOICE_ASSISTANT:
        return "Voice-Assistant";
    case DEVICE_VIDEO_STREAMING:
        return "Video-Streaming";
    case DEVICE_REHD:
        return "REHD-Sensor";
    default:
        return "Unknown";
    }
}

// TwtNetworkSetup Constructor
TwtNetworkSetup::TwtNetworkSetup(const TwtSimulationConfig& config)
      : m_config(config)
{
}

// Helper function to start WiFi PHY at random time
void
TwtNetworkSetup::RandomWifiStart(Ptr<WifiPhy> phyPtr)
{
    phyPtr->ResumeFromOff();
}

// Create network nodes
void
TwtNetworkSetup::CreateNodes()
{
    wifiApNodes.Create(1);
    ApNode = wifiApNodes.Get(0);

    wifiStaNodes.Create(m_config.nTotal());

    p2pServerNodes.Create(1);
    p2pServerNode = p2pServerNodes.Get(0);

    // Initialize heterogeneous network configuration
    if (m_config.enable_heterogeneous_traffic)
    {
        const_cast<TwtSimulationConfig&>(m_config).InitializeHeterogeneousNetwork();

        if (!g_quietMode)
        {
            std::cout << "\n================================================================"
                      << std::endl;
            std::cout << "HETEROGENEOUS NETWORK CONFIGURATION" << std::endl;
            std::cout << "================================================================"
                      << std::endl;

            for (uint32_t i = 0; i < m_config.nTotal(); i++)
            {
                const auto& cfg = m_config.sta_app_configs[i];
                std::cout << "STA " << i << ": " << cfg.device_name << " ("
                          << m_config.GetDeviceClassName(cfg.device_class) << ") "
                          << "Rate=" << (cfg.traffic_rate_bps / 1e6) << " Mbps "
                          << "Battery=" << cfg.battery_capacity_mj << " mJ "
                          << "Priority=" << cfg.priority_level << std::endl;
            }
            std::cout << "================================================================\n"
                      << std::endl;
        }
    }
}

// Configure WiFi network
void
TwtNetworkSetup::ConfigureWifi()
{
    // P2P setup: Wired backbone link between AP and Server.
    // This link simulates the backhaul connection where AP forwards STA traffic to/from backend.
    // Topology: [WiFi STAs] <--WiFi--> [AP] <--P2P (1Gbps)--> [Server]
    NodeContainer p2pConnectedNodes;
    p2pConnectedNodes.Add(ApNode);
    p2pConnectedNodes.Add(p2pServerNode);

    PointToPointHelper pointToPoint;
    std::stringstream delayString;
    delayString << m_config.p2pLinkDelay_ms << "ms";
    pointToPoint.SetDeviceAttribute("DataRate", StringValue("1000Mbps"));
    pointToPoint.SetChannelAttribute("Delay", StringValue(delayString.str()));
    p2pdevices = pointToPoint.Install(p2pConnectedNodes);

    // WiFi configuration (constants defined in twt-simulation-config.h)
    int channelWidth = DEFAULT_CHANNEL_WIDTH_MHZ;
    int mcs = DEFAULT_MCS;

    std::ostringstream ossDataMode;
    ossDataMode << "HeMcs" << mcs;

    // 2.4 GHz ISM band — required for PowerCast P21XXCSR Band 6 harvesting coherence (the rectenna only harvests 2.4 GHz RF) and to match the Friis model frequency (WIFI_CENTER_FREQ_HZ) + harvester η-curve (2450 MHz).
    // Channel 0 = auto-select (resolves to channel 1, 2412 MHz, at 20 MHz).
    std::string channelStr = "{0, " + std::to_string(channelWidth) + ", BAND_2_4GHZ, 0}";

    wifi.SetStandard(WIFI_STANDARD_80211ax);
    std::ostringstream ossControlMode;
    auto nonHtRefRateMbps = HePhy::GetNonHtReferenceRate(mcs) / 1e6;
    ossControlMode << "OfdmRate" << nonHtRefRateMbps << "Mbps";

    Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold",
                       UintegerValue(RTS_CTS_THRESHOLD));

    // Disable beacon jitter so the AP's first beacon is at t=0 and beacons land on a clean k*BI grid.
    // The PDW burst tick fires on the same k*BI grid, so the burst (which starts PDW_BEACON_LEAD into each BI) is correctly anchored AFTER the beacon.
    // With jitter ON the first beacon is randomly offset in [0,BI), which would slide the burst back onto the beacon.
    // (Single-AP sim: jitter, meant to de-correlate multiple APs, has no other purpose here.)
    Config::SetDefault("ns3::ApWifiMac::EnableBeaconJitter", BooleanValue(false));

    wifi.SetRemoteStationManager(
        "ns3::ConstantRateWifiManager",
        "DataMode",
        StringValue(ossDataMode.str()),
        "ControlMode",
        StringValue("HeMcs0"),
        // Pin the rate for group-addressed frames (beacons + the PDW power broadcast).
        // Default is the lowest basic rate (~1 Mbps), which makes a 1400 B PDW frame ~12 ms — too coarse for the burst.
        // HeMcs0 (~8.6 Mbps) keeps PDW frames ~1.5 ms so D is a graded knob.
        // See PDW_BCAST_WIFI_MODE in twt-constants.h.
        "NonUnicastMode",
        StringValue(PDW_BCAST_WIFI_MODE));

    Ssid ssid = Ssid("unilateral-twt-network");

    Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();

    // --- PROPAGATION MODEL: Friis free-space (n=2.0, ref 40.05 dB @ 1m @ 2.4 GHz). ---
    // Most-lossless defensible indoor model — physical floor for path loss.
    // Combined with the AP-side 26 dBm EIRP and the +6 dBi REHD rectenna gain (set below), in-room harvest is the same order as REHD TX drain.
    Ptr<FriisPropagationLossModel> propLoss = CreateObject<FriisPropagationLossModel>();
    propLoss->SetAttribute("Frequency", DoubleValue(WIFI_CENTER_FREQ_HZ));
    spectrumChannel->AddPropagationLossModel(propLoss);
    // Share this exact model with the harvest path so a sleeping REHD's antenna RX power is computed with the same path loss the channel applies (see the time-switching harvest comment near the top of this file).
    g_propLoss = propLoss;
    spectrumChannel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    phy.SetChannel(spectrumChannel);

    // Configure STA MAC
    // MaxMissedBeacons: STA never disconnects from AP (UINT32_MAX)
    // BE_BlockAckThreshold: Enable Block Ack after 1 packet for efficient ACK aggregation
    //   - Individual ACKs: Each frame gets its own ACK (high overhead)
    //   - Block ACK: Aggregates ACKs into single bitmap frame (efficient)
    //   - Threshold=1: Enables Block Ack immediately, critical for TWT Service Periods where STAs wake for short windows and may send only 1-2 packets
    mac.SetType("ns3::StaWifiMac",
                "MaxMissedBeacons",
                UintegerValue(MAX_MISSED_BEACONS),
                "BE_BlockAckThreshold",
                UintegerValue(BLOCK_ACK_THRESHOLD),
                "Ssid",
                SsidValue(ssid));

    phy.Set("ChannelSettings", StringValue(channelStr));
    // STAs install with NS-3 default TX power (16 dBm conducted, 0 dB gains) — realistic for typical mobile / IoT devices.
    // Per-REHD overrides below.
    staDevices = wifi.Install(phy, mac, wifiStaNodes);

    // --- Per-device-class PHY overrides ---
    // REHD STAs are ultra-low-power sensors with a directional rectenna:
    //   TX power 10 dBm (vs 16 dBm default) — they can barely afford to TX
    //   RxGain  +6 dBi (small directional rectenna pointed at AP)
    // Non-REHD STAs keep the NS-3 defaults (16 dBm, 0 dB gains).
    for (uint32_t i = 0; i < wifiStaNodes.GetN(); ++i)
    {
        const bool isRehd = (i < m_config.sta_app_configs.size() &&
                             m_config.sta_app_configs[i].device_class == DEVICE_REHD);
        if (!isRehd)
        {
            continue;
        }
        Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(wifiStaNodes.Get(i)->GetDevice(0));
        Ptr<WifiPhy> staPhy = dev->GetPhy();
        staPhy->SetAttribute("TxPowerStart", DoubleValue(REHD_TX_POWER_DBM));
        staPhy->SetAttribute("TxPowerEnd", DoubleValue(REHD_TX_POWER_DBM));
        staPhy->SetAttribute("RxGain", DoubleValue(REHD_RX_GAIN_DBI));
    }

    // Configure AP MAC
    // EnableBeaconJitter: Disabled (false) for precise, predictable beacon timing
    //   - Critical for TWT: STAs rely on beacon intervals to anchor wake schedules
    //   - Consistent timing ensures reliable Service Period synchronization
    // BE_BlockAckThreshold: Same aggressive threshold (BLOCK_ACK_THRESHOLD) as STA for symmetry
    //   - Reduces per-packet ACK overhead on downlink during TWT SPs
    // BE_MaxAmpduSize: A-MPDU aggregation limit (ampduLimitBytes = 20KB)
    //   - Balances throughput gains vs retry latency on lossy channels
    // BsrLifetime: Buffer Status Report validity window (bsrLife_ms = 10ms)
    //   - AP considers BSR valid for duration before requesting fresh reports
    //   - Allows AP to maintain accurate knowledge of STA queue states
    mac.SetType("ns3::ApWifiMac",
                "EnableBeaconJitter",
                BooleanValue(false),
                "BE_BlockAckThreshold",
                UintegerValue(BLOCK_ACK_THRESHOLD),
                "BE_MaxAmpduSize",
                UintegerValue(m_config.ampduLimitBytes),
                "BsrLifetime",
                TimeValue(MilliSeconds(m_config.bsrLife_ms)),
                "Ssid",
                SsidValue(ssid));

    // AP-side: enterprise AP.
    // 20 dBm conducted + 6 dBi gain → 26 dBm EIRP (lowered from 36 EIRP so harvest no longer dwarfs REHD TX drain — see twt-constants.h AP_TX_POWER_DBM).
    phy.Set("TxPowerStart", DoubleValue(AP_TX_POWER_DBM));
    phy.Set("TxPowerEnd", DoubleValue(AP_TX_POWER_DBM));
    phy.Set("TxGain", DoubleValue(AP_TX_GAIN_DBI));
    apDevice = wifi.Install(phy, mac, wifiApNodes);

    // Random Number Generator (RNG) Stream Assignment
    // Sets up deterministic randomness for reproducible simulations:
    //   - SetSeed(randSeed): Controls global RNG seed (from config, overridable via cmdline)
    //   - SetRun(N): Marks this as run N of a multi-run parameter study
    //   - AssignStreams(): Assigns independent RNG streams to each device to avoid correlation
    //
    // For single-run reproducibility:
    //   Set same seed → get identical random positions, traffic patterns, backoffs
    //
    // For multi-run statistical analysis (recommended for TWT controller evaluation):
    //   Run with increasing seeds (seed+0, seed+1, seed+2, ...) to generate different topologies while maintaining reproducibility per-run.
    //   Analyze aggregate statistics across runs to get confidence intervals on controller performance metrics.
    //
    // Stream ID allocation:
    //   - Start at RNG_INITIAL_STREAM_ID (100)
    //   - AP device: streams 100-119
    //   - STA devices: streams 120+ (auto-incremented by AssignStreams)
    RngSeedManager::SetSeed(m_config.randSeed);
    RngSeedManager::SetRun(RNG_DEFAULT_RUN_NUMBER);
    int64_t streamNumber = RNG_INITIAL_STREAM_ID;
    streamNumber += wifi.AssignStreams(apDevice, streamNumber);
    streamNumber += wifi.AssignStreams(staDevices, streamNumber);

    // WiFi 6 Guard Interval Configuration (802.11ax)
    // NOTE: This is SINGLE-USER mode (NOT OFDMA):
    //   - ConstantRateWifiManager: Fixed MCS per device, one at a time
    //   - TWT implicit: Non-overlapping Service Periods ensure no collisions
    //   - Result: Only ONE STA transmits during its SP window
    //
    // Guard Interval (GI) in single-user WiFi 6:
    //   - Not for multicarrier separation (that's OFDMA's job)
    //   - Purpose: Protect against multipath reflections from walls/objects
    //   - 800ns GI: Safe margin for indoor room (~67ns/meter × 12m = ~800ns max delay)
    //   - Allows receiver to complete channel estimation before symbol sampling
    //
    // Why not use 0.4µs GI here?
    //   - Saves time per symbol (3.2µs total symbol instead of 4.0µs)
    //   - But requires very clean channel with minimal reflections
    //   - Indoor room with furniture has significant multipath → need 0.8µs margin
    //
    Config::Set(
        "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HeConfiguration/GuardInterval",
        TimeValue(NanoSeconds(DEFAULT_GUARD_INTERVAL_NS))); // Convert microseconds to nanoseconds

    // Randomized WiFi start times
    // Stagger WiFi PHY startup across STAs to avoid synchronized boot behavior:
    //   - Each STA randomly delays before turning on WiFi (uniform 0-2.5s)
    //   - Prevents "thundering herd" of simultaneous association attempts
    //   - Helps AP handle association requests more gracefully
    //   - More realistic boot scenario (devices power on at different times in practice)
    for (uint32_t i = 0; i < wifiStaNodes.GetN(); i++)
    {
        Ptr<Node> n = wifiStaNodes.Get(i);
        Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(n->GetDevice(0));
        Ptr<WifiPhy> phyPtr = wifi_dev->GetPhy();
        phyPtr->SetOffMode();

        Ptr<UniformRandomVariable> random = CreateObject<UniformRandomVariable>();
        uint64_t start_time_bi = random->GetInteger(0, WIFI_STARTUP_WINDOW_BI);
        uint64_t start_time_ms = static_cast<uint64_t>(start_time_bi * BEACON_INTERVAL_MS);

        Simulator::Schedule(
            MilliSeconds(start_time_ms), &TwtNetworkSetup::RandomWifiStart, this, phyPtr);
    }
}

// Setup mobility
// AP is stationary at the origin.
// STAs are placed via rejection sampling so every pair starts at least MOBILITY_BUFFER_M apart.
// With ENABLE_DYNAMIC_MOBILITY they get ConstantVelocityMobilityModel (velocity is updated by per-STA UpdateStaDirection events scheduled in ScheduleDynamicMobility); otherwise they get ConstantPositionMobilityModel (legacy static-env behavior).
void
TwtNetworkSetup::SetupMobility()
{
    const double half = m_config.roomLength / 2.0;
    const double minSep = MOBILITY_BUFFER_M;
    const double minSep2 = minSep * minSep;
    const int maxAttempts = 1000;

    Ptr<UniformRandomVariable> xCoordinateRand = CreateObject<UniformRandomVariable>();
    Ptr<UniformRandomVariable> yCoordinateRand = CreateObject<UniformRandomVariable>();
    xCoordinateRand->SetAttribute("Min", DoubleValue(-half));
    xCoordinateRand->SetAttribute("Max", DoubleValue(half));
    yCoordinateRand->SetAttribute("Min", DoubleValue(-half));
    yCoordinateRand->SetAttribute("Max", DoubleValue(half));

    // REHDs spawn inside a harvest ANNULUS around the AP: [REHD_SPAWN_MIN_RADIUS_M, REHD_SPAWN_RADIUS_M] = [1.5 m, 4.75 m].
    // The inner exclusion is a deployment choice (no point-blank REHDs), distinct from the REHD_BUFFER_M collision bubble.
    // Sample uniform-by-area in polar coords: r = sqrt(uniform(rMin², rMax²)), theta = uniform(0, 2π).
    Ptr<UniformRandomVariable> rehdRSqRand = CreateObject<UniformRandomVariable>();
    rehdRSqRand->SetAttribute("Min",
                              DoubleValue(REHD_SPAWN_MIN_RADIUS_M * REHD_SPAWN_MIN_RADIUS_M));
    rehdRSqRand->SetAttribute("Max", DoubleValue(REHD_SPAWN_RADIUS_M * REHD_SPAWN_RADIUS_M));
    Ptr<UniformRandomVariable> rehdThetaRand = CreateObject<UniformRandomVariable>();
    rehdThetaRand->SetAttribute("Min", DoubleValue(0.0));
    rehdThetaRand->SetAttribute("Max", DoubleValue(2.0 * M_PI));

    // AP at origin in its own helper (always stationary).
    {
        MobilityHelper apMobility;
        Ptr<ListPositionAllocator> apAlloc = CreateObject<ListPositionAllocator>();
        apAlloc->Add(Vector(0.0, 0.0, 0.0));
        apMobility.SetPositionAllocator(apAlloc);
        apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        apMobility.Install(wifiApNodes);
    }

    // Rejection sampling for STA positions: every pair >= minSep apart.
    // Also keep STAs minSep away from the AP at origin so we don't model unrealistically-close clients.
    std::vector<Vector> staPositions;
    staPositions.reserve(m_config.nTotal());
    uint32_t totalAttempts = 0;
    for (uint32_t ii = 0; ii < m_config.nTotal(); ii++)
    {
        const bool isRehd = (ii < m_config.sta_app_configs.size() &&
                             m_config.sta_app_configs[ii].device_class == DEVICE_REHD);
        Vector chosen(0.0, 0.0, 0.0);
        bool placed = false;
        for (int a = 0; a < maxAttempts; a++)
        {
            totalAttempts++;
            Vector cand;
            if (isRehd)
            {
                // REHDs: polar sample inside harvest disk, outside AP buffer band.
                const double r = std::sqrt(rehdRSqRand->GetValue());
                const double theta = rehdThetaRand->GetValue();
                cand = Vector(r * std::cos(theta), r * std::sin(theta), 0.0);
                // (No need to re-check AP buffer or room bounds — the polar sample is constructed within the [REHD_SPAWN_MIN_RADIUS_M, REHD_SPAWN_RADIUS_M] annulus, which (4.75 m max) is fully inside the 12 m room.)
            }
            else
            {
                // Non-REHDs: uniform across full room, reject if too close to AP (non-REHDs keep the full STA buffer from the AP at origin).
                cand = Vector(xCoordinateRand->GetValue(), yCoordinateRand->GetValue(), 0.0);
                if (cand.x * cand.x + cand.y * cand.y < minSep2)
                {
                    continue;
                }
            }
            // Pairwise collision-avoidance buffer (asymmetric by class):
            //   REHD_BUFFER_M (0.25) if either entity is a REHD, else MOBILITY_BUFFER_M (0.5).
            // staPositions is index-aligned with sta_app_configs (both filled in STA-index order), so staPositions[k] belongs to STA index k.
            bool ok = true;
            for (uint32_t k = 0; k < staPositions.size(); k++)
            {
                const bool kIsRehd = (k < m_config.sta_app_configs.size() &&
                                      m_config.sta_app_configs[k].device_class == DEVICE_REHD);
                const double reqBuf = (isRehd || kIsRehd) ? REHD_BUFFER_M : MOBILITY_BUFFER_M;
                double dx = staPositions[k].x - cand.x;
                double dy = staPositions[k].y - cand.y;
                if (dx * dx + dy * dy < reqBuf * reqBuf)
                {
                    ok = false;
                    break;
                }
            }
            if (ok)
            {
                chosen = cand;
                placed = true;
                break;
            }
        }
        if (!placed)
        {
            std::cerr << "WARNING: TwtNetworkSetup::SetupMobility could not place STA " << ii
                      << " (class=" << (isRehd ? "REHD" : "non-REHD") << ") with " << minSep
                      << " m buffer after " << maxAttempts
                      << " attempts; falling back to last candidate (room may be too small)."
                      << std::endl;
            // Fallback: last sample, same distribution as the loop tried
            if (isRehd)
            {
                const double r = std::sqrt(rehdRSqRand->GetValue());
                const double theta = rehdThetaRand->GetValue();
                chosen = Vector(r * std::cos(theta), r * std::sin(theta), 0.0);
            }
            else
            {
                chosen = Vector(xCoordinateRand->GetValue(), yCoordinateRand->GetValue(), 0.0);
            }
        }
        staPositions.push_back(chosen);
        if (!g_quietMode)
        {
            std::cout << "STA " << ii << " position: [" << chosen.x << ", " << chosen.y << ", 0.0]"
                      << std::endl;
        }
    }

    Ptr<ListPositionAllocator> staAlloc = CreateObject<ListPositionAllocator>();
    for (const auto& p : staPositions)
    {
        staAlloc->Add(p);
    }

    MobilityHelper staMobility;
    staMobility.SetPositionAllocator(staAlloc);
#if ENABLE_DYNAMIC_MOBILITY
    staMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
#else
    staMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
#endif
    staMobility.Install(wifiStaNodes);

    if (!g_quietMode)
    {
#if ENABLE_DYNAMIC_MOBILITY
        std::cout << "[Mobility] Dynamic random walk ENABLED ("
                  << "speed=" << MOBILITY_SPEED_MIN_MPS << "-" << MOBILITY_SPEED_MAX_MPS
                  << " m/s, dir-change=" << MOBILITY_DIR_CHANGE_S
                  << " s, buffer=" << MOBILITY_BUFFER_M << " m). Rejection sampling used "
                  << totalAttempts << " attempts." << std::endl;
#else
        std::cout << "[Mobility] Dynamic mobility DISABLED (static positions). "
                  << "Rejection sampling used " << totalAttempts << " attempts." << std::endl;
#endif
    }
}

// Configure Internet stack
void
TwtNetworkSetup::ConfigureInternet()
{
    InternetStackHelper stack;
    stack.Install(wifiApNodes);
    stack.Install(wifiStaNodes);
    stack.Install(p2pServerNodes);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    apNodeInterface = address.Assign(apDevice);
    staNodeInterfaces = address.Assign(staDevices);
    address.SetBase("192.168.2.0", "255.255.255.0");
    p2pNodeInterfaces = address.Assign(p2pdevices);

    // Print IP to MAC mapping
    std::map<Ipv4Address, Mac48Address> ipToMac;
    for (uint32_t i = 0; i < wifiStaNodes.GetN(); i++)
    {
        Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(wifiStaNodes.Get(i)->GetDevice(0));
        Ptr<WifiMac> wifi_mac = wifi_dev->GetMac();
        Ptr<StaWifiMac> sta_mac = DynamicCast<StaWifiMac>(wifi_mac);
        ipToMac[staNodeInterfaces.GetAddress(i)] = sta_mac->GetAddress();
    }

    if (!g_quietMode)
    {
        std::cout << "\n\nIP to MAC mapping:\n";
        for (auto it = ipToMac.begin(); it != ipToMac.end(); it++)
        {
            std::cout << it->first << " => " << it->second << std::endl;
        }
    }

    Simulator::Schedule(Seconds(0), &Ipv4GlobalRoutingHelper::PopulateRoutingTables);
}

// Setup applications - HETEROGENEOUS NETWORK VERSION
void
TwtNetworkSetup::SetupApplications()
{
    if (!g_quietMode)
    {
        std::cout << "twt-simulation-config-SetupApplications Starting heterogeneous app setup"
                  << std::endl;
    }

    uint16_t port = 50000;

    // PER-CLASS PACKET LIFETIME (2026-06-07): set each STA's UL queue MaxDelay to its class deadline (= latency_requirement, also exposed as the oracle delay_bound).
    // Distinct per class — Voice 200 / Video 400 / Camera 600 ms, IoT 2 s, REHD 5 s — sized achievable-but-tight under TWT (>=~1.5 BI), so a GOOD schedule meets them while a BAD one expires the tight ones.
    // Overrides the global 1000 ms default.
    // A tight-deadline STA served late -> its packets EXPIRE -> the class-blind expiry term guides the agent to serve it promptly (latency-as-expiry proxy).
    // REHDs expire only on a long energy lockout.
    for (std::size_t i = 0; i < m_config.nTotal(); i++)
    {
        Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(wifiStaNodes.Get(i)->GetDevice(0));
        if (!dev)
        {
            continue;
        }
        Ptr<WifiMac> mac = dev->GetMac();
        if (!mac)
        {
            continue;
        }
        Time dl = m_config.sta_app_configs[i].latency_requirement;
        for (AcIndex ac : {AC_BE, AC_BK, AC_VI, AC_VO})
        {
            Ptr<WifiMacQueue> q = mac->GetTxopQueue(ac);
            if (q)
            {
                q->SetMaxDelay(dl);
            }
        }
    }

    // Create random variable for app start times
    appStartTimeRand = CreateObject<UniformRandomVariable>();
    appStartTimeRand->SetAttribute("Min", DoubleValue(m_config.AppStartTimeMin.GetMilliSeconds()));
    appStartTimeRand->SetAttribute("Max", DoubleValue(m_config.AppStartTimeMax.GetMilliSeconds()));

    // Create sink applications at server (one per STA)
    for (std::size_t i = 0; i < m_config.nTotal(); i++)
    {
        Address sinkLocalAddress(InetSocketAddress(Ipv4Address::GetAny(), port + i));
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", sinkLocalAddress);
        ApplicationContainer tempServerApp = sinkHelper.Install(p2pServerNode);
        tempServerApp.Start(Seconds(0.0));
        tempServerApp.Stop(MilliSeconds(m_config.simulationTime_ms + 1));
        serverApp.Add(tempServerApp);
    }

    // Create heterogeneous client applications based on device class
    if (!m_config.enable_heterogeneous_traffic || m_config.sta_app_configs.empty())
    {
        std::cerr << "\n[ERROR] Heterogeneous traffic disabled or no STA application configs found!"
                  << std::endl;
        std::cerr << "  enable_heterogeneous_traffic: " << m_config.enable_heterogeneous_traffic
                  << std::endl;
        std::cerr << "  sta_app_configs size: " << m_config.sta_app_configs.size() << std::endl;
        throw std::runtime_error("Application setup failed: heterogeneous traffic must be enabled "
                                 "with valid STA configs");
    }

    if (!g_quietMode)
    {
        std::cout << "\n[HETEROGENEOUS TRAFFIC SETUP]" << std::endl;
    }

    for (uint32_t i = 0; i < m_config.nTotal(); i++)
    {
        if (i >= m_config.sta_app_configs.size())
        {
            std::cerr << "Error: STA " << i << " has no app config" << std::endl;
            throw std::runtime_error("Missing application config for STA " + std::to_string(i));
        }

        const auto& config = m_config.sta_app_configs[i];
        Address serverAddr(InetSocketAddress(p2pNodeInterfaces.GetAddress(1), port + i));

        switch (config.device_class)
        {
        case DEVICE_IOT_SENSOR:
            CreateIoTSensorApplication(i, config, serverAddr);
            break;
        case DEVICE_VIDEO_CAMERA:
            CreateVideoCameraApplication(i, config, serverAddr);
            break;
        case DEVICE_VOICE_ASSISTANT:
            CreateVoiceAssistantApplication(i, config, serverAddr);
            break;
        case DEVICE_VIDEO_STREAMING:
            CreateVideoStreamingApplication(i, config, serverAddr);
            break;
        case DEVICE_REHD:
            CreateRehdSensorApplication(i, config, serverAddr);
            break;
        default:
            std::cerr << "Error: Unknown device class for STA " << i << std::endl;
            throw std::runtime_error("Unknown device class for STA " + std::to_string(i));
        }
    }

    // PDW: arm the AP-side broadcast burst engine (per-BI tick; silent while D==0).
    SetupPowerDeliveryWindow();

    if (!g_quietMode)
    {
        std::cout << std::endl;
    }
}

// IoT Sensor: Periodic traffic, very low rate, battery-powered
void
TwtNetworkSetup::CreateIoTSensorApplication(uint32_t sta_index,
                                            const StaApplicationConfig& config,
                                            const Address& server_addr)
{
    OnOffHelper onoff("ns3::UdpSocketFactory", server_addr);

    // ON time: use config value for burst duration (expects random variable string in seconds)
    double onTime_s = config.on_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OnTime",
        StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(onTime_s) + "]"));

    // OFF time: sleep between bursts (expects random variable string in seconds)
    double offTime_s = config.off_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OffTime",
        StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(offTime_s) + "]"));

    // Data rate during ON period
    std::stringstream rateStr;
    rateStr << config.traffic_rate_bps << "bps";
    onoff.SetAttribute("DataRate", StringValue(rateStr.str()));
    onoff.SetAttribute("PacketSize", UintegerValue(config.packet_size_bytes));

    double appStartTimeMs = appStartTimeRand->GetValue();
    ApplicationContainer clientApp = onoff.Install(wifiStaNodes.Get(sta_index));
    clientApp.Start(MilliSeconds(appStartTimeMs));
    clientApp.Stop(MilliSeconds(m_config.simulationTime_ms - 1));
    m_perStaClientApps.push_back(clientApp.Get(0));

    double dutyCycle = onTime_s / (onTime_s + offTime_s) * 100.0;
    double effectiveRate_kbps =
        config.traffic_rate_bps * onTime_s / (onTime_s + offTime_s) / 1000.0;
    if (!g_quietMode)
    {
        std::cout << "  IoT Sensor STA " << sta_index << ": ON=" << (onTime_s * 1000)
                  << "ms, OFF=" << (offTime_s * 1000)
                  << "ms, Rate=" << (config.traffic_rate_bps / 1000.0) << "Kbps"
                  << ", Effective=" << effectiveRate_kbps << "Kbps (" << dutyCycle << "% duty)"
                  << std::endl;
    }
}

// Video Camera: Constant bitrate, medium rate, always-on
void
TwtNetworkSetup::CreateVideoCameraApplication(uint32_t sta_index,
                                              const StaApplicationConfig& config,
                                              const Address& server_addr)
{
    UdpClientHelper client(server_addr);

    client.SetAttribute("Interval", TimeValue(config.packet_interval));
    client.SetAttribute("PacketSize", UintegerValue(config.packet_size_bytes));
    client.SetAttribute("MaxPackets", UintegerValue(1000000)); // Continuous streaming

    double appStartTimeMs = appStartTimeRand->GetValue();
    ApplicationContainer clientApp = client.Install(wifiStaNodes.Get(sta_index));
    clientApp.Start(MilliSeconds(appStartTimeMs));
    clientApp.Stop(MilliSeconds(m_config.simulationTime_ms - 1));
    m_perStaClientApps.push_back(clientApp.Get(0));

    if (!g_quietMode)
    {
        std::cout << "  Video Camera STA " << sta_index
                  << ": CBR=" << (config.traffic_rate_bps / 1e6)
                  << "Mbps, Interval=" << config.packet_interval.GetMilliSeconds() << "ms"
                  << std::endl;
    }
}

// Voice Assistant: ON/OFF pattern, low rate, latency-critical
void
TwtNetworkSetup::CreateVoiceAssistantApplication(uint32_t sta_index,
                                                 const StaApplicationConfig& config,
                                                 const Address& server_addr)
{
    OnOffHelper onoff("ns3::UdpSocketFactory", server_addr);

    // ON time: exponential distribution for realistic conversation turns (expects seconds)
    double onMean_s = config.on_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OnTime",
        StringValue("ns3::ExponentialRandomVariable[Mean=" + std::to_string(onMean_s) + "]"));

    // OFF time: exponential silence periods (expects seconds)
    double offMean_s = config.off_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OffTime",
        StringValue("ns3::ExponentialRandomVariable[Mean=" + std::to_string(offMean_s) + "]"));

    std::stringstream rateStr;
    rateStr << config.traffic_rate_bps << "bps";
    onoff.SetAttribute("DataRate", StringValue(rateStr.str()));
    onoff.SetAttribute("PacketSize", UintegerValue(config.packet_size_bytes));

    double appStartTimeMs = appStartTimeRand->GetValue();
    ApplicationContainer clientApp = onoff.Install(wifiStaNodes.Get(sta_index));
    clientApp.Start(MilliSeconds(appStartTimeMs));
    clientApp.Stop(MilliSeconds(m_config.simulationTime_ms - 1));
    m_perStaClientApps.push_back(clientApp.Get(0));

    if (!g_quietMode)
    {
        std::cout << "  Voice Assistant STA " << sta_index << ": OnTime=" << onMean_s
                  << "s, OffTime=" << offMean_s
                  << "s, Latency=" << config.latency_requirement.GetMilliSeconds() << "ms"
                  << std::endl;
    }
}

// Video Streaming: Bursty VBR, high rate, interactive
void
TwtNetworkSetup::CreateVideoStreamingApplication(uint32_t sta_index,
                                                 const StaApplicationConfig& config,
                                                 const Address& server_addr)
{
    OnOffHelper onoff("ns3::UdpSocketFactory", server_addr);

    // ON time: exponential for realistic video watching/seeking (expects seconds)
    double onMean_s = config.on_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OnTime",
        StringValue("ns3::ExponentialRandomVariable[Mean=" + std::to_string(onMean_s) + "]"));

    // OFF time: exponential for pause/seek/buffer (expects seconds)
    double offMean_s = config.off_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OffTime",
        StringValue("ns3::ExponentialRandomVariable[Mean=" + std::to_string(offMean_s) + "]"));

    std::stringstream rateStr;
    rateStr << config.traffic_rate_bps << "bps";
    onoff.SetAttribute("DataRate", StringValue(rateStr.str()));
    onoff.SetAttribute("PacketSize", UintegerValue(config.packet_size_bytes));

    double appStartTimeMs = appStartTimeRand->GetValue();
    ApplicationContainer clientApp = onoff.Install(wifiStaNodes.Get(sta_index));
    clientApp.Start(MilliSeconds(appStartTimeMs));
    clientApp.Stop(MilliSeconds(m_config.simulationTime_ms - 1));
    m_perStaClientApps.push_back(clientApp.Get(0));

    if (!g_quietMode)
    {
        std::cout << "  Video Streaming STA " << sta_index
                  << ": Rate=" << (config.traffic_rate_bps / 1e6)
                  << "Mbps, Burstiness=" << config.burstiness << std::endl;
    }
}

// REHD Sensor: harvester-only powered; brief ON-burst at the start of each interval, then long OFF.
// Traffic params (rate / packet size / interval) were already sampled per-STA inside InitializeHeterogeneousNetwork from the sub-type template — here we just install the OnOff app with those frozen values.
void
TwtNetworkSetup::CreateRehdSensorApplication(uint32_t sta_index,
                                             const StaApplicationConfig& config,
                                             const Address& server_addr)
{
    OnOffHelper onoff("ns3::UdpSocketFactory", server_addr);

    const double onTime_s = config.on_time_mean.GetSeconds();
    const double offTime_s = config.off_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OnTime",
        StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(onTime_s) + "]"));
    onoff.SetAttribute(
        "OffTime",
        StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(offTime_s) + "]"));

    std::stringstream rateStr;
    rateStr << config.traffic_rate_bps << "bps";
    onoff.SetAttribute("DataRate", StringValue(rateStr.str()));
    onoff.SetAttribute("PacketSize", UintegerValue(config.packet_size_bytes));

    double appStartTimeMs = appStartTimeRand->GetValue();
    ApplicationContainer clientApp = onoff.Install(wifiStaNodes.Get(sta_index));
    clientApp.Start(MilliSeconds(appStartTimeMs));
    clientApp.Stop(MilliSeconds(m_config.simulationTime_ms - 1));
    m_perStaClientApps.push_back(clientApp.Get(0));

    if (!g_quietMode)
    {
        const double effective_bps = config.traffic_rate_bps * onTime_s / (onTime_s + offTime_s);
        std::cout << "  REHD Sensor STA " << sta_index << " (" << config.device_name
                  << "): peak=" << config.traffic_rate_bps
                  << " bps, pkt=" << config.packet_size_bytes
                  << " B, interval=" << (onTime_s + offTime_s) << " s, effective=" << effective_bps
                  << " bps" << std::endl;
    }
}

// Power Delivery Window (PDW) — per-BI DIRECT RF power delivery, anchored to beacon.
//
// The AP is a dedicated power beacon over [PDW_BEACON_LEAD, pdw_end] of every BI (pdw_end = m_pdwDurationMs, set per agent action by ApplyTWTSchedule or the CLI --pdwDurationMs).
// We DO NOT transmit packetized WiFi frames: routing power through UDP/IP/WiFi-MAC drags in CSMA contention, which leaves IFS/backoff gaps between frames (wasted airtime) and — worse — lets a momentarily-busy channel defer the back-to-back chain late, spilling power frames deep into the comms region (measured p99 end ~83-98 ms).
// A real Powercast RF power transmitter emits a continuous waveform, not WiFi packets.
// So we deliver the harvest DIRECTLY: DeliverPdwPower credits every asleep REHD the AP's RF energy for the full window duration, using the SAME analytical Friis path as the comms harvest (OnPhyTxEnd).
// Result: 100% window occupancy, zero gaps, and ZERO leak — an energy credit cannot spill onto the channel or collide with the next SP (and a REHD awake in its SP isn't credited, since the gate is IsStateSleep).
// The channel stays idle during the window, so comms STAs (asleep [0,pdw_end], waking at pdw_end) get a pristine medium.
//
// Anchored to the REAL beacon TX (g_onApBeaconTx, fired from ApPhyTxPsduBeginTrace) so the delivery lands PDW_BEACON_LEAD after the actual beacon — the AP's GetTimeTillNextBeacon is a dummy, so a computed BI grid would be offset from the true TBTT.
// (The lead also lets any REHD that briefly woke to RX the beacon get back asleep.)
void
TwtNetworkSetup::SetupPowerDeliveryWindow()
{
    // Static fixed-D sweep (enableDynamicTWT=false) drives delivery from the CLI D.
    // In dynamic mode ApplyTWTSchedule overrides m_pdwDurationMs each action (CLI D=0).
    m_pdwDurationMs = m_config.pdwDurationMs;
    const double marginMs = PDW_BEACON_LEAD_US / 1000.0;

    // Beacon hook (real beacon TX): schedule this BI's power delivery PDW_BEACON_LEAD after the beacon.
    // No-op if pdw_end <= lead (window empty = no PDW).
    g_onApBeaconTx = [this, marginMs]()
    {
        if (m_pdwDurationMs > marginMs)
        {
            Simulator::Schedule(
                MicroSeconds(PDW_BEACON_LEAD_US), &TwtNetworkSetup::DeliverPdwPower, this);
        }
    };

    if (!g_quietMode)
    {
        std::cout << "[PDW] direct power delivery armed (beacon-anchored, continuous-RF): lead="
                  << marginMs << " ms, initial pdw_end=" << m_pdwDurationMs << " ms" << std::endl;
    }
}

void
TwtNetworkSetup::DeliverPdwPower()
{
    // Credit one BI's RF energy to every REHD asleep RIGHT NOW, for the full power window [PDW_BEACON_LEAD, pdw_end].
    // SetHarvestedEnergy is linear in duration and the AP->REHD RX power is fixed by geometry, so one call over the whole window == a continuous beam over it (the packetized burst only ever reached ~90% occupancy).
    // Bounded to the window by construction: no channel TX, so nothing can leak past it.
    const double windowMs = m_pdwDurationMs - (PDW_BEACON_LEAD_US / 1000.0);
    if (windowMs <= 0.0 || !g_propLoss)
    {
        return;
    }
    const Time windowDur = MilliSeconds(windowMs);
    Ptr<MobilityModel> apMob = ApNode->GetObject<MobilityModel>();
    if (!apMob)
    {
        return;
    }
    // AP EIRP = conducted + TX gain, matching OnPhyTxEnd's nodeId==0 harvest treatment.
    const double apEirpDbm = AP_TX_POWER_DBM + AP_TX_GAIN_DBI;
    for (uint32_t i = 0; i < g_phHarvesters.size(); ++i)
    {
        if (!g_phHarvesters[i] || !g_rehdPhy[i] || !g_rehdMob[i] || !g_rehdPhy[i]->IsStateSleep())
        {
            continue;
        }
        const double rxDbm =
            g_propLoss->CalcRxPower(apEirpDbm, apMob, g_rehdMob[i]) + REHD_RX_GAIN_DBI;
        g_phHarvesters[i]->SetHarvestedEnergy(rxDbm, windowDur);
        g_phHarvestEvents[i]++;
        ReconcileRehdLock(i); // may have recharged enough to release the uplink-queue lock
    }
}

// UNILATERAL TWT - AP assigns individual schedules
void
TwtNetworkSetup::initiateUnicastTwtAtAp(Ptr<WifiMac> apMac,
                                        Mac48Address staMacAddress,
                                        uint8_t flowId,
                                        Time twtWakeInterval,
                                        Time twtNominalWakeDuration,
                                        Time nextTwtOffsetFromNextBeacon)
{
    apMac->SetTwtSchedule(flowId,
                          staMacAddress,
                          false, // isRequestingNode = false (AP is responder)
                          true,  // isImplicitAgreement = true
                          true,  // flowType = true (unannounced)
                          false, // isTriggerBasedAgreement = false
                          true,  // isIndividualAgreement = true (per-STA)
                          0, // twtChannel - specifies WiFi channel (0 = primary/default channel)
                          twtWakeInterval,
                          twtNominalWakeDuration,
                          nextTwtOffsetFromNextBeacon);
}

void
TwtNetworkSetup::initiateTwtAtSta(Ptr<WifiMac> staMac,
                                  Ptr<WifiMac> apMac,
                                  uint8_t flowId,
                                  Time twtWakeInterval,
                                  Time twtNominalWakeDuration,
                                  Time nextTwtOffsetFromNextBeacon)
{
    Mac48Address apMacAddress = apMac->GetAddress();

    staMac->SetTwtSchedule(flowId,
                           apMacAddress,
                           true,  // isRequestingNode = true at STA
                           true,  // isImplicitAgreement = true
                           true,  // flowType = true (unannounced)
                           false, // isTriggerBasedAgreement = false
                           true,  // isIndividualAgreement = true
                           0, // twtChannel - specifies WiFi channel (0 = primary/default channel)
                           twtWakeInterval,
                           twtNominalWakeDuration,
                           nextTwtOffsetFromNextBeacon);
}

// Setup TWT schedule
void
TwtNetworkSetup::SetupTwtSchedule()
{
    if (!g_quietMode)
    {
        std::cout << "\n===== UNILATERAL TWT ASSIGNMENT =====" << std::endl;
        std::cout << "AP assigns individual TWT slots to each STA" << std::endl;
        std::cout << "STAs automatically adopt the announced schedule" << std::endl;
        std::cout << "No negotiation required\n" << std::endl;
    }

    for (uint32_t i = 0; i < wifiStaNodes.GetN(); i++)
    {
        Ptr<WifiNetDevice> apDevicePtr = DynamicCast<WifiNetDevice>(ApNode->GetDevice(1));
        Ptr<WifiMac> apMac = apDevicePtr->GetMac();

        Ptr<WifiNetDevice> staDevice =
            DynamicCast<WifiNetDevice>(wifiStaNodes.Get(i)->GetDevice(0));
        Ptr<WifiMac> staMac = staDevice->GetMac();
        Ptr<StaWifiMac> sta_mac = DynamicCast<StaWifiMac>(staMac);
        Mac48Address staMacAddress = sta_mac->GetAddress();

        // Default (twtSleepGroups==0): all STAs share the SAME SP offset and a 95 ms wake -> ~93% duty, effectively "no TWT" (EDCA).
        // The agent overrides this at TWT_UPDATE_START_BI in dynamic mode.
        // Grouped sleep test (twtSleepGroups>0): round-robin STAs into N groups, each with a staggered SP offset (BI/N apart) and a SHORT wake -> STAs sleep most of every BI so REHDs harvest.
        // Diagnostic only (standalone).
        Time nextTwtOffsetFromNextBeacon;
        Time wakeDur;
        if (m_config.pdwDurationMs > 0.0)
        {
            // Phase-1 PDW schedule: [0,D] = AP power-delivery window (every STA asleep and harvesting), [D,100] = ONE TWT block containing every STA (all share this offset+wake), [100,102.4] = margin.
            // REHDs doze through [0,D] so they harvest the AP broadcast (harvest gate = IsStateSleep).
            nextTwtOffsetFromNextBeacon = MilliSeconds(m_config.pdwDurationMs);
            wakeDur = MilliSeconds(100.0 - m_config.pdwDurationMs);
        }
        else if (m_config.twtSleepGroups > 0)
        {
            const uint32_t G = static_cast<uint32_t>(m_config.twtSleepGroups);
            const uint32_t g = i % G;
            const double biMs = m_config.beaconInterval_s.GetSeconds() * 1000.0;
            const double spacingMs = biMs / G;
            const double wakeMs = std::min(m_config.twtGroupWakeMs, spacingMs - 1.0);
            nextTwtOffsetFromNextBeacon = MilliSeconds(1.0 + g * spacingMs);
            wakeDur = MilliSeconds(wakeMs);
        }
        else
        {
            nextTwtOffsetFromNextBeacon = m_config.firstTwtSpOffsetFromBeacon;
            wakeDur = m_config.twtNominalWakeDuration;
        }

        if (nextTwtOffsetFromNextBeacon + wakeDur >= m_config.beaconInterval_s)
        {
            std::cerr << "ERROR: TWT SP for STA " << i << " (offset+duration) exceeds beacon "
                      << "interval! Offset: " << nextTwtOffsetFromNextBeacon.As(Time::MS)
                      << ", Duration: " << wakeDur.As(Time::MS)
                      << ", Beacon Interval: " << m_config.beaconInterval_s.As(Time::MS)
                      << std::endl;
            throw std::runtime_error("TWT schedule exceeds beacon interval - no wrapping allowed");
        }

        // Flow ID = 0 for all STAs (individual TWT uses single Flow ID per STA).
        // TWT group assignment is managed separately via wake interval/duration/offset.
        uint8_t flowId = 0;

        Simulator::Schedule(m_config.firstTwtSpStart,
                            &TwtNetworkSetup::initiateUnicastTwtAtAp,
                            this,
                            apMac,
                            staMacAddress,
                            flowId,
                            m_config.twtWakeInterval,
                            wakeDur,
                            nextTwtOffsetFromNextBeacon);

        Simulator::Schedule(m_config.firstTwtSpStart,
                            &TwtNetworkSetup::initiateTwtAtSta,
                            this,
                            staMac,
                            apMac,
                            flowId,
                            m_config.twtWakeInterval,
                            wakeDur,
                            nextTwtOffsetFromNextBeacon);

        if (!g_quietMode)
        {
            std::cout << "STA " << i << " (" << staMacAddress << "):" << std::endl;
            std::cout << "  TWT announced at t=" << m_config.firstTwtSpStart.As(Time::S)
                      << std::endl;
            std::cout << "  SP offset: +" << nextTwtOffsetFromNextBeacon.As(Time::MS)
                      << " from beacon" << std::endl;
            std::cout << "  Wake interval: " << m_config.twtWakeInterval.As(Time::MS) << std::endl;
            std::cout << "  Wake duration: " << m_config.twtNominalWakeDuration.As(Time::MS)
                      << std::endl;
            std::cout << std::endl;
        }
    }

    if (!g_quietMode)
    {
        std::cout << "=====================================\n" << std::endl;
    }
}

// Get STA MAC by index
Ptr<WifiMac>
TwtNetworkSetup::GetStaMac(uint32_t staId) const
{
    if (staId >= wifiStaNodes.GetN())
    {
        return nullptr;
    }
    Ptr<WifiNetDevice> staDevice =
        DynamicCast<WifiNetDevice>(wifiStaNodes.Get(staId)->GetDevice(0));
    return staDevice->GetMac();
}

// Get AP MAC
Ptr<WifiMac>
TwtNetworkSetup::GetApMac() const
{
    Ptr<WifiNetDevice> apDevicePtr = DynamicCast<WifiNetDevice>(ApNode->GetDevice(1));
    return apDevicePtr->GetMac();
}

// Apply TWT schedule from Python controller (group-based)
// We will be using ms for time as it is related to python
void
TwtNetworkSetup::ApplyTWTSchedule(const ActionStruct& action)
{
    // PDW 3rd action head (Phase 2).
    // The Python decoder has already scale-shifted every group's twt_sp_offset_ms / twt_wake_duration_ms into the comms region [D, 95], so when D>0 we apply those offsets DIRECTLY (the front-guard is D, not the legacy 2 ms firstTwtSpOffsetFromBeacon).
    // D==0 keeps the legacy +2 ms baseline for exact backward-compatibility with the pre-PDW path.
    const double pdwD = action.pdw_duration_ms;

    if (!g_quietMode)
    {
        std::cout << "\n[TWT Update] Applying group-based TWT schedule at t="
                  << Simulator::Now().GetSeconds() << "s" << std::endl;
        std::cout << "  Active TWT Groups: " << action.num_active_twt_groups << ", PDW: " << pdwD
                  << " ms" << std::endl;
    }

    // Store last action for later retrieval by metrics
    m_lastNumActiveTwtGroups = action.num_active_twt_groups;
    for (uint32_t g = 0; g < MAX_NUM_TWT_GROUPS; g++)
    {
        m_lastTwtGroupConfigs[g] = action.twt_group_configs[g];
    }
    for (uint32_t i = 0; i < MAX_NUM_STA; i++)
    {
        if (i < action.num_sta)
        {
            m_lastStaGroupAssignments[i] = action.sta_group_assignments[i].assigned_twt_group;
            m_lastStaTwtEnabled[i] = (action.sta_group_assignments[i].enable_twt != 0);
        }
        else
        {
            m_lastStaGroupAssignments[i] = 0;
            m_lastStaTwtEnabled[i] = false;
        }
    }

    Ptr<WifiMac> apMac = GetApMac();

    // Process each STA assignment
    for (uint32_t i = 0; i < action.num_sta && i < wifiStaNodes.GetN(); i++)
    {
        const StaGroupAssignment& sta_assign = action.sta_group_assignments[i];

        if (!sta_assign.enable_twt)
        {
            if (!g_quietMode)
            {
                std::cout << "  STA " << i << ": TWT disabled by controller" << std::endl;
            }
            continue;
        }

        // Get the TWT group configuration for this STA
        uint8_t groupId = sta_assign.assigned_twt_group;
        if (groupId >= action.num_active_twt_groups)
        {
            if (!g_quietMode)
            {
                std::cout << "  STA " << i << ": Invalid group ID " << (int)groupId << std::endl;
            }
            continue;
        }

        const TwtGroupConfig& group_cfg = action.twt_group_configs[groupId];

        Ptr<WifiMac> staMac = GetStaMac(i);
        if (!staMac)
        {
            continue;
        }

        Ptr<StaWifiMac> sta_mac = DynamicCast<StaWifiMac>(staMac);
        Mac48Address staMacAddress = sta_mac->GetAddress();
        Mac48Address apMacAddress = apMac->GetAddress();

        // Use group timing parameters.
        // MilliSeconds() takes an integer, so MilliSeconds(102.4) TRUNCATES to 102 ms (same gotcha noted at the beaconInterval_s definition).
        // That truncation made the TWT wake interval 102 ms vs the 102.4 ms beacon interval → the SP drifted ~0.4 ms earlier every BI (measured), so the front group crept into the PDW power window.
        // Convert via Seconds(ms/1000.0) to keep the 0.4 ms (and sub-ms SP offsets like 57.6 ms) exact and phase-lock the SP to the beacon.
        Time newWakeInterval = Seconds(group_cfg.twt_wake_interval_ms / 1000.0);
        Time newWakeDuration = Seconds(group_cfg.twt_wake_duration_ms / 1000.0);
        Time groupOffset = Seconds(group_cfg.twt_sp_offset_ms / 1000.0);
        // PDW mode (D>0): decoder already front-guarded the SP at D, so use the offset as-is.
        // Legacy (D==0): keep the +2 ms firstTwtSpOffsetFromBeacon.
        Time effectiveGroupOffset =
            (pdwD > 0.0) ? groupOffset : (m_config.firstTwtSpOffsetFromBeacon + groupOffset);

        uint8_t flowId = 0; // Flow ID must be 0-7 per IEEE 802.11ax

        // Update AP side
        apMac->SetTwtSchedule(
            flowId,
            staMacAddress,
            false, // isRequestingNode = false (AP is responder)
            true,  // isImplicitAgreement
            true,  // flowType (unannounced)
            false, // isTriggerBasedAgreement
            true,  // isIndividualAgreement
            0,     // twtChannel - specifies WiFi channel (0 = primary/default channel)
            newWakeInterval,
            newWakeDuration,
            effectiveGroupOffset);

        // Update STA side
        staMac->SetTwtSchedule(
            flowId,
            apMacAddress,
            true,  // isRequestingNode = true at STA
            true,  // isImplicitAgreement
            true,  // flowType (unannounced)
            false, // isTriggerBasedAgreement
            true,  // isIndividualAgreement
            0,     // twtChannel - specifies WiFi channel (0 = primary/default channel)
            newWakeInterval,
            newWakeDuration,
            effectiveGroupOffset);

        if (!g_quietMode)
        {
            std::cout << "  STA " << i << " → Group " << (int)groupId
                      << ": Interval=" << group_cfg.twt_wake_interval_ms << "ms"
                      << ", Duration=" << group_cfg.twt_wake_duration_ms << "ms"
                      << ", Offset=" << group_cfg.twt_sp_offset_ms << "ms" << std::endl;
        }
    }

    // Set the PDW end time for this action; the per-BI beacon hook (armed in SetupPowerDeliveryWindow) reads it and delivers power [lead, pdw_end] at the next beacon.
    // pdw_end <= PDW_BEACON_LEAD (empty window) = no PDW.
    m_pdwDurationMs = pdwD;

    if (!g_quietMode)
    {
        std::cout << "[TWT Update] Group-based schedule applied successfully\n" << std::endl;
    }
}

// Enable PCAP
void
TwtNetworkSetup::EnablePcap()
{
    if (m_config.enablePcap)
    {
        std::stringstream ss1;
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        ss1 << TwtResultsPath("pcap_AP_unilateral");
        phy.EnablePcap(ss1.str(), apDevice);
    }
}

// DYNAMIC MOBILITY: per-STA random walk on top of ConstantVelocityMobilityModel.
//
// SetupMobility() installs the ConstantVelocityMobilityModel and seeds STA positions with rejection-sampled non-overlapping placements.
// This routine then schedules:
//   - per-STA UpdateStaDirection events that re-pick (speed, direction) every MOBILITY_DIR_CHANGE_S seconds
//   - a periodic EnforceBuffer sweep that keeps every pair >= MOBILITY_BUFFER_M apart by pushing them apart and redirecting their velocities outward
//
// All RNG variables get explicit stream IDs so reproducibility per seed is preserved (different seed -> different walk; same seed -> identical walk).
void
TwtNetworkSetup::ScheduleDynamicMobility()
{
#if !ENABLE_DYNAMIC_MOBILITY
    return; // toggle off: legacy static positions
#else
    const uint32_t n = wifiStaNodes.GetN();
    m_mobilitySpeedRand.resize(n);
    m_mobilityDirRand.resize(n);

    // Stream IDs are spaced well clear of the WiFi assignment block (which starts at RNG_INITIAL_STREAM_ID = 100 and consumes a few hundred IDs for AP + STA devices).
    int64_t streamId = RNG_INITIAL_STREAM_ID + 1000;

    for (uint32_t i = 0; i < n; i++)
    {
        m_mobilitySpeedRand[i] = CreateObject<UniformRandomVariable>();
        m_mobilitySpeedRand[i]->SetAttribute("Min", DoubleValue(MOBILITY_SPEED_MIN_MPS));
        m_mobilitySpeedRand[i]->SetAttribute("Max", DoubleValue(MOBILITY_SPEED_MAX_MPS));
        m_mobilitySpeedRand[i]->SetStream(streamId++);

        m_mobilityDirRand[i] = CreateObject<UniformRandomVariable>();
        m_mobilityDirRand[i]->SetAttribute("Min", DoubleValue(0.0));
        m_mobilityDirRand[i]->SetAttribute("Max", DoubleValue(2.0 * M_PI));
        m_mobilityDirRand[i]->SetStream(streamId++);

        // Stagger first direction-set across STAs so they don't all fire at t=0.
        Time t0 = MilliSeconds(static_cast<uint64_t>(i * 50));
        Simulator::Schedule(t0, &TwtNetworkSetup::UpdateStaDirection, this, i);
    }

    // First buffer-correction sweep one interval in.
    Simulator::Schedule(
        MilliSeconds(MOBILITY_CORRECTION_INTERVAL_MS), &TwtNetworkSetup::EnforceBuffer, this);

    if (!g_quietMode)
    {
        std::cout << "[Dynamic Mobility] Scheduled per-STA random walk for " << n
                  << " STAs (dir-change every " << MOBILITY_DIR_CHANGE_S
                  << " s, buffer correction every " << MOBILITY_CORRECTION_INTERVAL_MS << " ms)."
                  << std::endl;
    }
#endif
}

void
TwtNetworkSetup::UpdateStaDirection(uint32_t staId)
{
    if (staId >= wifiStaNodes.GetN())
    {
        return;
    }
    // REHDs are static-random: drawn-once spawn position, no random walk.
    if (staId < m_config.sta_app_configs.size() &&
        m_config.sta_app_configs[staId].device_class == DEVICE_REHD)
    {
        return; // do NOT reschedule — REHDs stay at zero velocity for the episode
    }
    Ptr<ConstantVelocityMobilityModel> mob =
        wifiStaNodes.Get(staId)->GetObject<ConstantVelocityMobilityModel>();
    if (!mob)
    {
        return;
    }

    const double half = m_config.roomLength / 2.0;
    Vector pos = mob->GetPosition();

    // Snap back into bounds in case buffer correction or a previous reflection pushed us beyond the wall (should be rare but defensive).
    pos.x = std::max(-half, std::min(half, pos.x));
    pos.y = std::max(-half, std::min(half, pos.y));
    mob->SetPosition(pos);

    double speed = m_mobilitySpeedRand[staId]->GetValue();
    double angle = m_mobilityDirRand[staId]->GetValue();
    double vx = speed * std::cos(angle);
    double vy = speed * std::sin(angle);

    // Wall reflection: if the new velocity would push us out of bounds before the next direction change, flip the appropriate component.
    double dt = MOBILITY_DIR_CHANGE_S;
    if (pos.x + vx * dt > half)
    {
        vx = -std::abs(vx);
    }
    else if (pos.x + vx * dt < -half)
    {
        vx = std::abs(vx);
    }
    if (pos.y + vy * dt > half)
    {
        vy = -std::abs(vy);
    }
    else if (pos.y + vy * dt < -half)
    {
        vy = std::abs(vy);
    }

    mob->SetVelocity(Vector(vx, vy, 0.0));

    // Reschedule self with same period (no jitter — staggering at episode start is enough; further randomization happens via direction draws).
    Simulator::Schedule(
        Seconds(MOBILITY_DIR_CHANGE_S), &TwtNetworkSetup::UpdateStaDirection, this, staId);
}

void
TwtNetworkSetup::EnforceBuffer()
{
#if !ENABLE_DYNAMIC_MOBILITY
    return;
#else
    const double half = m_config.roomLength / 2.0;
    const uint32_t n = wifiStaNodes.GetN();

    for (uint32_t i = 0; i < n; i++)
    {
        Ptr<ConstantVelocityMobilityModel> mi =
            wifiStaNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
        if (!mi)
        {
            continue;
        }
        Vector pi = mi->GetPosition();
        for (uint32_t j = i + 1; j < n; j++)
        {
            Ptr<ConstantVelocityMobilityModel> mj =
                wifiStaNodes.Get(j)->GetObject<ConstantVelocityMobilityModel>();
            if (!mj)
            {
                continue;
            }
            Vector pj = mj->GetPosition();
            double dx = pj.x - pi.x;
            double dy = pj.y - pi.y;
            double dist = std::sqrt(dx * dx + dy * dy);

            // Pairwise buffer (asymmetric by class): REHD_BUFFER_M if either entity is a REHD, else MOBILITY_BUFFER_M.
            // REHDs are immovable — when a non-REHD wanders too close, push only the non-REHD the FULL distance; REHDs stay put.
            // Two-REHD pairs are both immovable so we skip them (spawn rejection sampling already separated them).
            const bool i_is_rehd = (i < m_config.sta_app_configs.size() &&
                                    m_config.sta_app_configs[i].device_class == DEVICE_REHD);
            const bool j_is_rehd = (j < m_config.sta_app_configs.size() &&
                                    m_config.sta_app_configs[j].device_class == DEVICE_REHD);
            const double reqBuf = (i_is_rehd || j_is_rehd) ? REHD_BUFFER_M : MOBILITY_BUFFER_M;

            if (dist >= reqBuf || dist < 1e-6)
            {
                continue; // OK or coincident (nothing useful we can do)
            }
            if (i_is_rehd && j_is_rehd)
            {
                continue;
            }

            double ux = dx / dist;
            double uy = dy / dist;
            const double fullPush = (reqBuf - dist) + 0.05;
            const double halfPush = (reqBuf - dist) * 0.5 + 0.05;
            const double push_i = i_is_rehd ? 0.0 : (j_is_rehd ? fullPush : halfPush);
            const double push_j = j_is_rehd ? 0.0 : (i_is_rehd ? fullPush : halfPush);

            if (push_i > 0.0)
            {
                Vector npi(std::max(-half, std::min(half, pi.x - push_i * ux)),
                           std::max(-half, std::min(half, pi.y - push_i * uy)),
                           0.0);
                mi->SetPosition(npi);
                Vector vi = mi->GetVelocity();
                double si = std::sqrt(vi.x * vi.x + vi.y * vi.y);
                if (si > 0)
                {
                    mi->SetVelocity(Vector(-ux * si, -uy * si, 0.0));
                }
            }
            if (push_j > 0.0)
            {
                Vector npj(std::max(-half, std::min(half, pj.x + push_j * ux)),
                           std::max(-half, std::min(half, pj.y + push_j * uy)),
                           0.0);
                mj->SetPosition(npj);
                Vector vj = mj->GetVelocity();
                double sj = std::sqrt(vj.x * vj.x + vj.y * vj.y);
                if (sj > 0)
                {
                    mj->SetVelocity(Vector(ux * sj, uy * sj, 0.0));
                }
            }
        }
        // Re-fetch pi after possible SetPosition above (defensive).
        pi = mi->GetPosition();
    }

    Simulator::Schedule(
        MilliSeconds(MOBILITY_CORRECTION_INTERVAL_MS), &TwtNetworkSetup::EnforceBuffer, this);
#endif
}

// DYNAMIC TRAFFIC: per-STA Poisson schedule of (rate, on, off) updates.
//
// The applications themselves stay installed for the whole episode — we just SetAttribute() new param values mid-flight, which the OnOff/UdpClient generators pick up on the next ON-period / next packet (no teardown).
//
// Class is fixed per episode (not redrawn here); only the parameters drift within the per-class min/max windows defined in twt-constants.h.
void
TwtNetworkSetup::ScheduleDynamicTraffic()
{
#if !ENABLE_DYNAMIC_TRAFFIC
    return;
#else
    if (m_perStaClientApps.empty())
    {
        std::cerr << "WARNING: ScheduleDynamicTraffic called before app installation; skipping."
                  << std::endl;
        return;
    }
    const uint32_t n = static_cast<uint32_t>(m_perStaClientApps.size());

    m_trafficUpdateRand.resize(n);
    m_trafficRateRand.resize(n);
    m_trafficOnRand.resize(n);
    m_trafficOffRand.resize(n);

    int64_t streamId = RNG_INITIAL_STREAM_ID + 2000;

    // Per-STA draw RNGs (rate/on/off).
    // Created for ALL STAs to keep stream IDs index-aligned and deterministic; REHD indices are simply never re-drawn (skipped in UpdateTrafficSegment).
    for (uint32_t i = 0; i < n; i++)
    {
        m_trafficRateRand[i] = CreateObject<UniformRandomVariable>();
        m_trafficRateRand[i]->SetStream(streamId++);
        m_trafficOnRand[i] = CreateObject<UniformRandomVariable>();
        m_trafficOnRand[i]->SetStream(streamId++);
        m_trafficOffRand[i] = CreateObject<UniformRandomVariable>();
        m_trafficOffRand[i]->SetStream(streamId++);
    }
    // (The over/under coin is hash-based per (randSeed, segIdx) in UpdateTrafficSegment — no RNG stream needed; the ns3 substream coin was seed-correlated at fixed segment positions.)

    // Schedule N_TRAFFIC_SEGMENTS segment boundaries across the decision horizon.
    // Segment s spans agent-updates [s*segUpdates, (s+1)*segUpdates); set its traffic just before that span begins.
    const uint32_t segUpdates = DURATION_IN_UPDATE / N_TRAFFIC_SEGMENTS;
    for (uint32_t s = 0; s < N_TRAFFIC_SEGMENTS; s++)
    {
        double startBi =
            TWT_UPDATE_START_BI + static_cast<double>(s) * segUpdates * TWT_UPDATE_INTERVAL_BI;
        // seg0 set well before the first agent decision; later segments just before their boundary.
        // Use double-ms -> Seconds(ms/1000) (MilliSeconds() truncates non-integer ms).
        double leadBi = (s == 0) ? 20.0 : 1.0;
        double t_s = ((startBi - leadBi) * BEACON_INTERVAL_MS) / 1000.0;
        Simulator::Schedule(Seconds(t_s), &TwtNetworkSetup::UpdateTrafficSegment, this, s);
    }
    if (!g_quietMode)
    {
        std::cout << "[Dynamic Traffic] Segmented-coin regime: " << N_TRAFFIC_SEGMENTS
                  << " segments x " << segUpdates << " updates; S_sat=" << m_config.trafficScale
                  << " (over x" << SEG_OVER_MULT << ", under x" << SEG_UNDER_MULT << ")."
                  << std::endl;
    }
#endif
}

// SEGMENTED-COIN traffic: at each segment boundary toss a 50/50 coin to put this segment OVER (SEG_OVER_MULT) or UNDER (SEG_UNDER_MULT) saturation, then re-draw EVERY non-REHD STA's params at the resulting effective scale (= trafficScale * coin).
// REHDs are skipped (their pattern is fixed).
// m_segMult is read by UpdateTrafficForSta when it draws each rate.
void
TwtNetworkSetup::UpdateTrafficSegment(uint32_t segIdx)
{
#if ENABLE_DYNAMIC_TRAFFIC
    // Hash-based over/under coin: deterministic per (randSeed, segIdx) -> reproducible per scenario (CRN-safe), but ~50/50 and decorrelated across BOTH seed and segment.
    // Replaces the ns3 RNG coin, whose substream had persistent cross-seed correlation at fixed draw positions (the varying-SEED/fixed-RUN trap) -> some segment positions were always OVER/UNDER.
    // splitmix64 mix (verified ~0.50 over-rate per segment over thousands of seeds).
    uint64_t h = (uint64_t)m_config.randSeed * 0x9E3779B97F4A7C15ULL +
                 (uint64_t)(segIdx + 1) * 0xBF58476D1CE4E5B9ULL;
    h ^= (h >> 30);
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= (h >> 27);
    h *= 0x94D049BB133111EBULL;
    h ^= (h >> 31);
    const bool over = (h & 1ULL) == 0ULL;
    m_segMult = over ? SEG_OVER_MULT : SEG_UNDER_MULT;
    uint32_t redrawn = 0;
    for (uint32_t i = 0; i < m_perStaClientApps.size(); i++)
    {
        const bool isRehd = (i < m_config.sta_app_configs.size() &&
                             m_config.sta_app_configs[i].device_class == DEVICE_REHD);
        if (isRehd)
        {
            continue; // REHDs hold their episode-start pattern
        }
        UpdateTrafficForSta(i);
        redrawn++;
    }
    if (!g_quietMode)
    {
        std::cout << "[Dynamic Traffic] segment " << segIdx << " @ "
                  << Simulator::Now().GetSeconds() << "s -> " << (over ? "OVER" : "UNDER")
                  << " (effScale=" << (m_config.trafficScale * m_segMult) << "), redrew " << redrawn
                  << " non-REHD STAs." << std::endl;
    }
#endif
}

void
TwtNetworkSetup::UpdateTrafficForSta(uint32_t staId)
{
    if (staId >= m_perStaClientApps.size() || staId >= m_config.sta_app_configs.size())
    {
        return;
    }
    Ptr<Application> app = m_perStaClientApps[staId];
    if (!app)
    {
        return;
    }
    DeviceClass cls = m_config.sta_app_configs[staId].device_class;

    // Effective scale = saturation anchor (trafficScale) * this segment's coin multiplier (SEG_OVER_MULT / SEG_UNDER_MULT, set by UpdateTrafficSegment).
    // m_segMult defaults to 1.0.
    const double effScale = m_config.trafficScale * m_segMult;

    auto fmtRv = [](const std::string& kind, double param_s)
    {
        std::ostringstream s;
        s << "ns3::" << kind << "RandomVariable[" << (kind == "Constant" ? "Constant=" : "Mean=")
          << param_s << "]";
        return s.str();
    };

    switch (cls)
    {
    case DEVICE_IOT_SENSOR:
    {
        uint32_t rate = static_cast<uint32_t>(
            effScale * m_trafficRateRand[staId]->GetValue(IOT_RATE_MIN_BPS, IOT_RATE_MAX_BPS));
        double on_s = m_trafficOnRand[staId]->GetValue(IOT_ON_MIN_MS, IOT_ON_MAX_MS) / 1000.0;
        double off_s = m_trafficOffRand[staId]->GetValue(IOT_OFF_MIN_MS, IOT_OFF_MAX_MS) / 1000.0;
        app->SetAttribute("DataRate", DataRateValue(DataRate(rate)));
        app->SetAttribute("OnTime", StringValue(fmtRv("Constant", on_s)));
        app->SetAttribute("OffTime", StringValue(fmtRv("Constant", off_s)));
        break;
    }
    case DEVICE_VIDEO_CAMERA:
    {
        uint32_t rate =
            static_cast<uint32_t>(effScale * m_trafficRateRand[staId]->GetValue(
                                                 CAMERA_RATE_MIN_BPS, CAMERA_RATE_MAX_BPS));
        // CBR: interval = packet_size_bits / rate.
        // PacketSize is a fixed config value, not in the random ranges — so we read it from the per-STA config.
        uint32_t pkt_bits = m_config.sta_app_configs[staId].packet_size_bytes * 8;
        if (rate == 0)
        {
            return;
        }
        double interval_s = static_cast<double>(pkt_bits) / rate;
        app->SetAttribute("Interval", TimeValue(Seconds(interval_s)));
        break;
    }
    case DEVICE_VOICE_ASSISTANT:
    {
        uint32_t rate = static_cast<uint32_t>(
            effScale * m_trafficRateRand[staId]->GetValue(VOICE_RATE_MIN_BPS, VOICE_RATE_MAX_BPS));
        double on_s =
            m_trafficOnRand[staId]->GetValue(VOICE_ON_MEAN_MIN_MS, VOICE_ON_MEAN_MAX_MS) / 1000.0;
        double off_s =
            m_trafficOffRand[staId]->GetValue(VOICE_OFF_MEAN_MIN_MS, VOICE_OFF_MEAN_MAX_MS) /
            1000.0;
        app->SetAttribute("DataRate", DataRateValue(DataRate(rate)));
        app->SetAttribute("OnTime", StringValue(fmtRv("Exponential", on_s)));
        app->SetAttribute("OffTime", StringValue(fmtRv("Exponential", off_s)));
        break;
    }
    case DEVICE_VIDEO_STREAMING:
    {
        uint32_t rate = static_cast<uint32_t>(
            effScale * m_trafficRateRand[staId]->GetValue(VIDEO_STREAM_RATE_MIN_BPS,
                                                          VIDEO_STREAM_RATE_MAX_BPS));
        double on_s = m_trafficOnRand[staId]->GetValue(VIDEO_STREAM_ON_MEAN_MIN_MS,
                                                       VIDEO_STREAM_ON_MEAN_MAX_MS) /
                      1000.0;
        double off_s = m_trafficOffRand[staId]->GetValue(VIDEO_STREAM_OFF_MEAN_MIN_MS,
                                                         VIDEO_STREAM_OFF_MEAN_MAX_MS) /
                       1000.0;
        app->SetAttribute("DataRate", DataRateValue(DataRate(rate)));
        app->SetAttribute("OnTime", StringValue(fmtRv("Exponential", on_s)));
        app->SetAttribute("OffTime", StringValue(fmtRv("Exponential", off_s)));
        break;
    }
    default:
        break;
    }
    // No self-reschedule: the segmented-coin handler (UpdateTrafficSegment) drives all re-draws.
}

// SetupEnergyHarvesting — install BasicEnergySource + PowercastEnergyHarvester on every STA, then connect PHY RX/TX Begin/End callbacks to drive realtime harvesting and consumption.
// Phase-1 scope only: no EnvStruct / reward integration yet — the harvesters just charge/discharge silently and we read totals at the end (LogFinalEnergyStats).
void
TwtNetworkSetup::SetupEnergyHarvesting(const std::string& configFile)
{
    using namespace energy;

    const uint32_t totalSTAs = static_cast<uint32_t>(m_config.nTotal());
    if (totalSTAs == 0)
    {
        return;
    }

    // Pre-size all the per-sta_idx accounting vectors to total — non-REHD slots stay nullptr/zero and the PHY callbacks self-filter via ResolveStaIdx().
    m_harvesters.assign(totalSTAs, nullptr);
    g_phHarvesters.assign(totalSTAs, nullptr);
    g_phHarvestEvents.assign(totalSTAs, 0);
    g_phTxEvents.assign(totalSTAs, 0);
    g_phUnpoweredTxEvents.assign(totalSTAs, 0);
    g_rehdPhy.assign(totalSTAs, nullptr);
    g_rehdMob.assign(totalSTAs, nullptr);
    g_rehdMac.assign(totalSTAs, nullptr);
    g_rehdEnergyLocked.assign(totalSTAs, false);
    g_rehdExpiredPkts.assign(totalSTAs, 0);
    g_phOngoingTxEvents.clear();
    // AP MAC address: REHD uplink RA to block/unblock for energy-gated TX.
    {
        Ptr<WifiNetDevice> apDev = DynamicCast<WifiNetDevice>(ApNode->GetDevice(1));
        if (apDev)
        {
            g_apMacAddr = apDev->GetMac()->GetAddress();
        }
    }

    // --- Collect REHD nodes; early-return if none ---

    NodeContainer rehdNodes;
    std::vector<uint32_t>
        rehdStaIdx; // parallel to rehdNodes: rehdNodes.Get(j) lives at STA index rehdStaIdx[j]
    for (uint32_t i = 0; i < totalSTAs; ++i)
    {
        if (i < m_config.sta_app_configs.size() &&
            m_config.sta_app_configs[i].device_class == DEVICE_REHD)
        {
            rehdNodes.Add(wifiStaNodes.Get(i));
            rehdStaIdx.push_back(i);
        }
    }
    if (rehdNodes.GetN() == 0)
    {
        if (!g_quietMode)
        {
            std::cout << "[twt-powercast] No REHD-class STAs configured; "
                      << "skipping harvester setup." << std::endl;
        }
        return;
    }

    // --- Load hardware-class catalog (CAP_A/B/C µF) from config file ---

    // Per-node assignment lines in the file are IGNORED — the REHD type catalog (REHD_TYPE_TEMPLATES) drives (cap, voltage) per REHD instance instead.
    std::vector<std::string> candidates;
    if (!configFile.empty())
    {
        candidates.push_back(configFile);
    }
    candidates.push_back(TwtExamplePath("ph-harvester-config.txt"));
    candidates.push_back("ph-harvester-config.txt"); // cmake-cache/build CWD

    PowercastEnergyHarvesterHelper harvesterHelper;
    std::string usedPath;
    for (const auto& p : candidates)
    {
        if (LoadPhConfigFromFile(p, harvesterHelper) > 0)
        {
            usedPath = p;
            break;
        }
    }
    if (usedPath.empty())
    {
        NS_FATAL_ERROR("SetupEnergyHarvesting: no harvester config file found; tried "
                       << candidates.size() << " paths. Place ph-harvester-config.txt at "
                       << TwtExamplePath("."));
    }

    // --- Override per-REHD hardware classes from the type catalog ---

    // Helper keys configurations by absolute Node ID (set via Node::GetId() in ph-deployment-helper.cc::DoInstall).
    // Node 0 is the AP, STAs are 1..N.
    for (uint32_t j = 0; j < rehdNodes.GetN(); ++j)
    {
        const uint32_t sta_idx = rehdStaIdx[j];
        const auto& cfg = m_config.sta_app_configs[sta_idx];
        harvesterHelper.ConfigureNode(
            rehdNodes.Get(j)->GetId(), cfg.rehd_cap_class, cfg.rehd_volt_class);
    }

    // --- Install BasicEnergySource on REHD nodes only ---

    BasicEnergySourceHelper esHelper;
    esHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(100.0));
    // Copy-INITIALIZE (not assign) into a local.
    // EnergySourceContainer derives from ns3::Object: copy-construction allocates a fresh m_aggregates buffer, but copy-ASSIGNMENT memberwise-copies the temporary's raw m_aggregates pointer (then the temporary frees it) → dangling pointer → SIGSEGV in ~Object later.
    // The sources stay alive for the whole run via node aggregation, so the local can go out of scope here safely.
    energy::EnergySourceContainer staEnergySources = esHelper.Install(rehdNodes);

    // --- Install one harvester per REHD source ---

    EnergyHarvesterContainer harvesters = harvesterHelper.Install(staEnergySources);

    for (uint32_t j = 0; j < rehdNodes.GetN(); ++j)
    {
        Ptr<PowercastEnergyHarvester> h = DynamicCast<PowercastEnergyHarvester>(harvesters.Get(j));
        const uint32_t sta_idx = rehdStaIdx[j];
        m_harvesters[sta_idx] = h;
        g_phHarvesters[sta_idx] = h;
        // Cache the REHD's PHY (for the SLEEP query) and mobility (for the analytical RX-power calc) used by the time-switching harvest path.
        Ptr<WifiNetDevice> rehdDev = DynamicCast<WifiNetDevice>(rehdNodes.Get(j)->GetDevice(0));
        g_rehdPhy[sta_idx] = rehdDev ? rehdDev->GetPhy() : nullptr;
        g_rehdMac[sta_idx] = rehdDev ? rehdDev->GetMac() : nullptr;
        g_rehdMob[sta_idx] = rehdNodes.Get(j)->GetObject<MobilityModel>();
        // REHD uplink buffer: extend MaxDelay to 10 s (vs 500 ms default) so packets survive an energy lockout, and hook the "Expired" trace to count drops (oracle metric).
        // Per AC queue.
        if (g_rehdMac[sta_idx])
        {
            for (AcIndex ac : {AC_BE, AC_BK, AC_VI, AC_VO})
            {
                Ptr<WifiMacQueue> q = g_rehdMac[sta_idx]->GetTxopQueue(ac);
                if (q)
                {
                    // REHD class deadline (5 s) — same as the unified per-class MaxDelay loop in SetupApplications (which also sets this); kept here with the Expired trace binding.
                    // (Was 10 s; now couples energy lockout to data-survival.)
                    q->SetMaxDelay(m_config.sta_app_configs[sta_idx].latency_requirement);
                    q->TraceConnectWithoutContext("Expired",
                                                  MakeBoundCallback(&::OnRehdExpired, sta_idx));
                }
            }
        }
    }

    // --- Wire PHY TX callbacks for ALL nodes ---

    // No RX hook: harvest is time-switching (sleep-only) and computed from the transmitter side in OnPhyTxEnd, so the REHD's own RX is irrelevant.
    // TX is hooked network-wide because every transmitter is an ambient RF source for sleeping REHDs; OnPhyTxEnd resolves REHD-self consumption separately.
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxBegin",
                    MakeCallback(&::OnPhyTxBegin));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxEnd",
                    MakeCallback(&::OnPhyTxEnd));
    // PHY state intervals → small awake-state (IDLE/RX/CCA) capacitor drain.
    // Always connected (independent of --disableTraces); self-filters to REHDs.
    Config::Connect("/NodeList/*/DeviceList/*/Phy/State/State", MakeCallback(&::OnRehdPhyState));

    if (!g_quietMode)
    {
        std::cout << "[twt-powercast] hardware-class catalog loaded from " << usedPath
                  << "; PowerCast harvesters installed on " << rehdNodes.GetN()
                  << " REHD STAs (out of " << totalSTAs << " total); PHY RX/TX callbacks connected"
                  << std::endl;
    }
}

void
TwtNetworkSetup::LogFinalEnergyStats() const
{
    if (m_harvesters.empty())
    {
        return; // SetupEnergyHarvesting was never called
    }
    if (g_quietMode)
    {
        return; // training/eval runs stay silent
    }

    std::cout << "\n========== PowerCast Energy Summary ==========" << std::endl;
    std::cout
        << " STA |    Vcap(V) |  Harv(J)  |  Cons(J)  | Avail(J)  | tActive(s) | HrvEv | TxEv "
           "| UnPwr | Expd | Ph"
        << std::endl;
    std::cout
        << "-----+------------+-----------+-----------+-----------+------------+-------+------"
           "+-------+------+----"
        << std::endl;
    for (uint32_t i = 0; i < m_harvesters.size(); ++i)
    {
        if (!m_harvesters[i])
        {
            continue;
        }
        const double tActive_s = m_harvesters[i]->GetTotalActiveTimeUs() / 1e6;
        std::cout << std::setw(4) << i << " | " << std::fixed << std::setprecision(4)
                  << std::setw(10) << m_harvesters[i]->GetVcap() << " | " << std::scientific
                  << std::setprecision(2) << std::setw(9)
                  << m_harvesters[i]->GetTotalHarvestedEnergy() << " | " << std::setw(9)
                  << m_harvesters[i]->GetTotalConsumedEnergy() << " | " << std::setw(9)
                  << m_harvesters[i]->GetAvailableEnergy() << " | " << std::fixed
                  << std::setprecision(3) << std::setw(10) << tActive_s << " | " << std::dec
                  << std::setw(5) << g_phHarvestEvents[i] << " | " << std::setw(4)
                  << g_phTxEvents[i] << " | " << std::setw(5) << g_phUnpoweredTxEvents[i] << " | "
                  << std::setw(4) << g_rehdExpiredPkts[i] << " | " << std::setw(2)
                  << static_cast<int>(m_harvesters[i]->ComputePhase()) << std::endl;
    }
    std::cout << "==============================================\n"
              << "(UnPwr = TXs the post-hoc energy audit could not sustain. The\n"
              << " queue-block gate (ReconcileRehdLock) normally stops these before\n"
              << " the MAC dequeues, so a small residual count is expected, not a bug.\n"
              << " Ph: 0=critical-low 1=active(full) 2=discharging(healthy) 3=protection(floor).)"
              << std::endl;
}

Ptr<energy::PowercastEnergyHarvester>
TwtNetworkSetup::GetHarvester(uint32_t staId) const
{
    if (staId >= m_harvesters.size())
    {
        return nullptr;
    }
    return m_harvesters[staId];
}

uint64_t
TwtNetworkSetup::GetUnpoweredTxEvents(uint32_t staId) const
{
    if (staId >= g_phUnpoweredTxEvents.size())
    {
        return 0;
    }
    return g_phUnpoweredTxEvents[staId];
}

uint64_t
TwtNetworkSetup::GetExpiredPackets(uint32_t staId) const
{
    if (staId >= g_rehdExpiredPkts.size())
    {
        return 0;
    }
    return g_rehdExpiredPkts[staId];
}

uint32_t
TwtNetworkSetup::GetStaQueueDepth(uint32_t staId) const
{
    if (staId >= wifiStaNodes.GetN())
    {
        return 0;
    }
    Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(wifiStaNodes.Get(staId)->GetDevice(0));
    if (!dev)
    {
        return 0;
    }
    Ptr<WifiMac> mac = dev->GetMac();
    uint32_t total = 0;
    for (AcIndex ac : {AC_BE, AC_BK, AC_VI, AC_VO})
    {
        Ptr<WifiMacQueue> q = mac->GetTxopQueue(ac);
        if (q)
        {
            total += q->GetNPackets();
        }
    }
    return total;
}

// TeardownEnergyHarvesting — release all harvester Ptr<> references BEFORE Simulator::Destroy().
// Without this, the file-scope statics in the anonymous namespace above are destroyed at process exit (after NS-3 globals are already gone), causing the Ptr destructors to UAF.
void
TwtNetworkSetup::TeardownEnergyHarvesting()
{
    g_phOngoingTxEvents.clear();
    g_phHarvesters.clear();
    g_phHarvestEvents.clear();
    g_phTxEvents.clear();
    g_phUnpoweredTxEvents.clear();
    g_rehdPhy.clear();
    g_rehdMob.clear();
    g_rehdMac.clear();
    g_rehdEnergyLocked.clear();
    g_rehdExpiredPkts.clear();
    g_propLoss = nullptr;
    m_harvesters.clear();
    // The BasicEnergySource container is a local in SetupEnergyHarvesting (not a member), so there is no EnergySourceContainer to release here.
    // The sources themselves are owned by the nodes and torn down by Simulator::Destroy().
}

} // namespace ns3