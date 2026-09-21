// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file twt-trace-callbacks.cc
 * @brief NS-3 trace sinks for PHY state, MAC queues, BSR, AP RX/TX, and REHD harvesting
 */

#include "twt-trace-callbacks.h"

#include "twt-simulation-config.h"

#include "ns3/ampdu-subframe-header.h" // strip A-MPDU subframe delimiter in the AP monitor sniffer
#include "ns3/bsr-manager.h"

#include <cmath>
#include <iomanip>

namespace ns3
{

// Configuration parameters (initialized from main)
// This is initialized to 0 as default but gets overwritten in twt-powercast-main-simulation.cc with the actual value from config.keepTrackOfMetricsFrom_ms.
double keepTrackOfMetricsFrom_ms = 0; // Metrics collection start time in milliseconds

// Verbosity / logging globals. Both default to false (existing behavior); set from twt-powercast-main-simulation.cc when --quietMode/--disableTraces are passed.
//   g_quietMode      -> suppress non-error std::cout in noisy hot paths
//   g_disableTraces  -> redundant signal that no trace files are open; the individual is_open() checks already handle the writes, this flag is here for symmetry / future fast-paths.
bool g_quietMode = false;
bool g_disableTraces = false;
std::unordered_map<std::string, double> TI_currentModel_mA = {{"IDLE", PHY_STATE_IDLE_MA},
                                                              {"CCA_BUSY", PHY_STATE_CCA_BUSY_MA},
                                                              {"RX", PHY_STATE_RX_MA},
                                                              {"TX", PHY_STATE_TX_MA},
                                                              {"SLEEP", PHY_STATE_SLEEP_MA}};

// Trace file definitions
std::ofstream e2eTraceFile;
std::ofstream macQueueSizeTraceFile;
std::ofstream ampduTraceFile;
std::ofstream bsrTraceFile;
std::ofstream phyStateTraceFile;
std::ofstream txRxStatsTraceFile;       // TX/RX success/fail/retry counters
std::ofstream linkMeasurementTraceFile; // RSSI, SNR, noise measurements
std::ofstream timeoutDropTraceFile;     // PSDU timeouts and MPDU drops
std::ofstream apTxTraceFile;            // AP downlink TX timing (PDW/beacon boundary compliance)
std::function<void()> g_onApBeaconTx = nullptr; // PDW burst anchor (set by TwtNetworkSetup)

// Tracking arrays definitions (all time values in milliseconds)
double* timeElapsedForSta_ms_TI = nullptr;
double* awakeTimeElapsedForSta_ms_TI = nullptr;
double* sleepTimeElapsedForSta_ms_TI = nullptr;
double* current_mA_TimesTime_ms_ForSta_TI = nullptr;
double* uplinkTimeoutsForSta = nullptr;
double downlinkTimeoutsAllSta = 0;
double* uplinkExpiredMpduForSta = nullptr;
double downlinkExpiredMpduAllSta = 0;
double* uplinkFailedEnqueueMpduForSta = nullptr;
double downlinkFailedEnqueueMpduAllSta = 0;
uint64_t* packetsGeneratedByAppForSta = nullptr;
uint64_t* packetsEnqueuedAtMacForSta = nullptr;
uint64_t* packetsTransmittedByPhyForSta = nullptr;
uint64_t* bytesTransmittedByPhyForSta = nullptr; // Track actual bytes for accurate throughput
double* latencySumMsForSta = nullptr;
uint64_t* latencyCountForSta = nullptr;

// --- 802.11k STA STATISTICS - Raw cumulative counters per STA ---
uint64_t* txSuccessCountForSta = nullptr;    // Successful TX (got ACK)
uint64_t* txRetryCountForSta = nullptr;      // TX retries (cumulative)
uint64_t* txFailedCountForSta = nullptr;     // TX failures (max retries exceeded)
uint64_t* rxSuccessCountForSta = nullptr;    // Successfully received frames
uint64_t* rxErrorCountForSta = nullptr;      // RX errors (FCS failures, decoding errors)
uint64_t* lastRxTimestampUsForSta = nullptr; // TSF (µs) of last UL frame AP RX'd from STA

// --- 802.11k LINK MEASUREMENT - Last known values per STA ---
double* lastRssiDbmForSta = nullptr;         // Last RSSI measurement (dBm)
double* lastSnrDbForSta = nullptr;           // Last SNR measurement (dB)
double* lastNoiseDbmForSta = nullptr;        // Last noise floor (dBm)
uint8_t* lastTxMcsForSta = nullptr;          // Last TX MCS index
uint8_t* lastTxNssForSta = nullptr;          // Last TX NSS (spatial streams)
uint16_t* lastTxRate100KbpsForSta = nullptr; // Last TX rate in 100 kbps units

// --- 802.11ax BSR (Buffer Status Report) - Per Access Category per STA ---
// Updated by BsrReceivedCallback when AP receives BSR from STAs
uint32_t* bsrQueueBytesAcBeForSta = nullptr; // Best Effort queue (TID 0,3)
uint32_t* bsrQueueBytesAcBkForSta = nullptr; // Background queue (TID 1,2)
uint32_t* bsrQueueBytesAcViForSta = nullptr; // Video queue (TID 4,5)
uint32_t* bsrQueueBytesAcVoForSta = nullptr; // Voice queue (TID 6,7)

// MAC address to STA ID mapping for BSR callback
std::unordered_map<std::string, uint32_t> macToStaIdMap;

// --- AP-RECEIVED TRAFFIC - Per Access Category per STA (from ApPhyRxEndTrace) ---
// TID extracted from QoS Data frames and mapped to AC
uint64_t* bytesReceivedAcBeForSta = nullptr;   // Bytes received from STA for AC_BE
uint64_t* bytesReceivedAcBkForSta = nullptr;   // Bytes received from STA for AC_BK
uint64_t* bytesReceivedAcViForSta = nullptr;   // Bytes received from STA for AC_VI
uint64_t* bytesReceivedAcVoForSta = nullptr;   // Bytes received from STA for AC_VO
uint64_t* packetsReceivedAcBeForSta = nullptr; // Packets received from STA for AC_BE
uint64_t* packetsReceivedAcBkForSta = nullptr; // Packets received from STA for AC_BK
uint64_t* packetsReceivedAcViForSta = nullptr; // Packets received from STA for AC_VI
uint64_t* packetsReceivedAcVoForSta = nullptr; // Packets received from STA for AC_VO

// --- QoEH-REALISTIC: per-SP UL accounting + AP DL accounting (see header). ---
uint64_t* bytesRxAtApInSpForSta = nullptr;
uint64_t* packetsRxAtApInSpForSta = nullptr;
uint32_t* spCompletedCountForSta = nullptr;
uint32_t* spWithDemandCountForSta = nullptr;
uint32_t* spWithStarvationCountForSta = nullptr;
uint64_t* spCurServedBytesForSta = nullptr;
uint64_t* spCurServedPktsForSta = nullptr;
uint64_t* spWakeStartUsForSta = nullptr;
uint32_t* spDemandBytesForSta = nullptr;
bool* spInitedForSta = nullptr;
uint64_t* dlBytesToStaForSta = nullptr;
uint64_t* dlDurationUsToStaForSta = nullptr;
uint32_t* dlUnicastCountToStaForSta = nullptr;
uint64_t g_dlBroadcastBytes = 0;
uint64_t g_dlBroadcastDurationUs = 0;
uint32_t g_dlBroadcastCount = 0;

// --- A-MPDU AGGREGATION - Cumulative counters per STA ---
uint64_t* ampduCountForSta = nullptr;      // Number of A-MPDU transmissions
uint64_t* ampduMpdusTotalForSta = nullptr; // Total MPDUs in A-MPDUs
uint64_t* ampduBytesTotalForSta = nullptr; // Total bytes in A-MPDUs

// --- AIRTIME & PHY PARAMETERS - Per STA tracking ---
uint64_t* airtimeUsedUsForSta = nullptr;       // Cumulative airtime in microseconds
uint8_t* lastChannelWidthMhzForSta = nullptr;  // Last channel width used (20/40/80/160)
uint16_t* lastGuardIntervalNsForSta = nullptr; // Last guard interval (800/1600/3200)
uint8_t* lastTxPowerDbmForSta = nullptr;       // Last TX power in dBm
uint8_t* lastFrameTypeForSta = nullptr;        // Last RX frame type (0=mgmt, 1=ctrl, 2=data)
uint8_t* lastFrameSubtypeForSta = nullptr;     // Last RX frame subtype (0-15)
uint8_t* lastPowerMgmtBitForSta = nullptr;     // Last Power Management bit from Frame Control

// Parse context to get NodeId
uint32_t
ContextToNodeId(std::string context)
{
    std::string sub = context.substr(10);
    uint32_t pos = sub.find("/Device");
    return atoi(sub.substr(0, pos).c_str());
}

// Get percentile value from histogram
double
GetPercentileValue(Histogram hist, double percentile)
{
    NS_ASSERT_MSG(percentile >= 0 && percentile <= 100, "Percentile should be between 0 and 100");
    double sum = 0;
    for (uint32_t i = 0; i < hist.GetNBins(); i++)
    {
        sum += hist.GetBinCount(i);
    }
    double target = sum * percentile / 100;
    sum = 0;
    for (uint32_t i = 0; i < hist.GetNBins(); i++)
    {
        sum += hist.GetBinCount(i);
        if (sum >= target)
        {
            return hist.GetBinEnd(i);
        }
    }
    return hist.GetBinEnd(hist.GetNBins() - 1);
}

// SUMMARY TRACE: Application packet generation (no time filter)
// Tracks packets created by the OnOffApplication before WiFi processing.
// Compares against MAC enqueues and PHY transmissions to detect app-layer drops.
void
TxTraceAtApp(std::string context, Ptr<const Packet> packet)
{
    uint32_t nodeId = ContextToNodeId(context);
    packetsGeneratedByAppForSta[nodeId - 1]++;
}

// SUMMARY TRACE: MAC layer packet enqueue (no time filter)
// Tracks packets successfully added to the WiFi MAC queue.
// Compares against app generation to detect app→MAC interface drops.
void
MacTxTrace(std::string context, Ptr<const Packet> packet)
{
    uint32_t nodeId = ContextToNodeId(context);
    packetsEnqueuedAtMacForSta[nodeId - 1]++;
}

// SUMMARY TRACE: PHY layer transmission start (no time filter)
// Tracks packets actually transmitted by the WiFi radio (PHY layer).
// Compares against MAC enqueues to detect queue drops/overflows.
// Also accumulates bytes for accurate throughput calculation.
// Note: txPowerW is a required NS-3 callback parameter (in Watts)
void
PhyTxBeginTrace(std::string context, Ptr<const Packet> packet, double txPowerW)
{
    uint32_t nodeId = ContextToNodeId(context);
    packetsTransmittedByPhyForSta[nodeId - 1]++;
    bytesTransmittedByPhyForSta[nodeId - 1] += packet->GetSize(); // Track actual bytes
}

// END-TO-END DETAILED TRACE CALLBACKS
// Records per-packet journey through the network stack for detailed analysis.
// Each callback logs packet transitions at different layers (APP→IP→MAC→PHY).
// All callbacks write to the same e2eTraceFile CSV with time-filtered entries.
// By cross-referencing packet UIDs across events, you can reconstruct complete packet paths and identify where/when packets are lost or delayed.
//
// Packet journey tracked: APP_TX → IP_TX → MAC_ENQUEUE → PHY_TX → PHY_RX_AP → IP_RX → APP_RX
// END-TO-END DETAILED TRACE: Application layer transmission at STA
void
E2E_AppTx(std::string context, Ptr<const Packet> packet)
{
    uint32_t nodeId = ContextToNodeId(context);
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms)
    {
        uint32_t uid = packet->GetUid();
        // Precision 6 = nanosecond resolution in milliseconds (1ns = 0.000001ms = 6 decimals)
        if (e2eTraceFile.is_open())
        {
            e2eTraceFile << std::fixed << std::setprecision(6) << Simulator::Now().GetMilliSeconds()
                         << ",APP_TX,STA" << (nodeId - 1) << "," << uid << "," << packet->GetSize()
                         << "," << std::endl;
        }
    }
}

// END-TO-END DETAILED TRACE: IP layer transmission at STA
void
E2E_IpTx(std::string context, Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface)
{
    uint32_t nodeId = ContextToNodeId(context);
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms)
    {
        uint32_t uid = packet->GetUid();
        if (e2eTraceFile.is_open())
        {
            e2eTraceFile << std::fixed << std::setprecision(6) << Simulator::Now().GetMilliSeconds()
                         << ",IP_TX,STA" << (nodeId - 1) << "," << uid << "," << packet->GetSize()
                         << "," << std::endl;
        }
    }
}

// END-TO-END DETAILED TRACE: MAC layer enqueue at STA
void
E2E_MacEnqueue(std::string context, Ptr<const Packet> packet)
{
    uint32_t nodeId = ContextToNodeId(context);
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms)
    {
        uint32_t uid = packet->GetUid();
        if (e2eTraceFile.is_open())
        {
            e2eTraceFile << std::fixed << std::setprecision(6) << Simulator::Now().GetMilliSeconds()
                         << ",MAC_ENQUEUE,STA" << (nodeId - 1) << "," << uid << ","
                         << packet->GetSize() << "," << std::endl;
        }
    }
}

// END-TO-END DETAILED TRACE: PHY transmission begins at STA
// Note: txPowerW is a required NS-3 callback parameter (in Watts)
void
E2E_PhyTx(std::string context, Ptr<const Packet> packet, double txPowerW)
{
    uint32_t nodeId = ContextToNodeId(context);
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms)
    {
        uint32_t uid = packet->GetUid();
        if (e2eTraceFile.is_open())
        {
            e2eTraceFile << std::fixed << std::setprecision(6) << Simulator::Now().GetMilliSeconds()
                         << ",PHY_TX,STA" << (nodeId - 1) << "," << uid << "," << packet->GetSize()
                         << "," << std::setprecision(6) << txPowerW << std::endl;
        }
    }
}

// END-TO-END DETAILED TRACE: PHY reception at AP
void
E2E_PhyRxAp(std::string context, Ptr<const Packet> packet)
{
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms)
    {
        uint32_t uid = packet->GetUid();
        if (e2eTraceFile.is_open())
        {
            e2eTraceFile << std::fixed << std::setprecision(6) << Simulator::Now().GetMilliSeconds()
                         << ",PHY_RX_AP,AP"
                         << "," << uid << "," << packet->GetSize() << "," << std::endl;
        }
    }
}

// END-TO-END DETAILED TRACE: IP reception at destination server
void
E2E_IpRx(std::string context, Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface)
{
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms)
    {
        uint32_t uid = packet->GetUid();
        if (e2eTraceFile.is_open())
        {
            e2eTraceFile << std::fixed << std::setprecision(6) << Simulator::Now().GetMilliSeconds()
                         << ",IP_RX,SERVER"
                         << "," << uid << "," << packet->GetSize() << "," << std::endl;
        }
    }
}

// END-TO-END DETAILED TRACE: Application reception at server
void
E2E_AppRx(Ptr<const Packet> packet, const Address& address)
{
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms)
    {
        uint32_t uid = packet->GetUid();
        if (e2eTraceFile.is_open())
        {
            e2eTraceFile << std::fixed << std::setprecision(6) << Simulator::Now().GetMilliSeconds()
                         << ",APP_RX,SERVER"
                         << "," << uid << "," << packet->GetSize() << "," << std::endl;
        }
    }
}

// QoS METRIC: MAC Queue Size Tracking
// Logs instantaneous queue size (newSize) at each queue change event.
// Specifically tracks the Best Effort (BE) traffic queue at WiFi MAC layer.
// This queue holds packets between MAC acceptance and PHY transmission.
// oldSize parameter is provided by NS-3 but not used here since we only care about the current queue state.
// Use (newSize - oldSize) to track queue change deltas.
void
QueueSizeTrace(std::string context, uint32_t oldSize, uint32_t newSize)
{
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms &&
        macQueueSizeTraceFile.is_open())
    {
        uint32_t nodeId = ContextToNodeId(context);
        macQueueSizeTraceFile << std::fixed << std::setprecision(6)
                              << Simulator::Now().GetMilliSeconds() << ",STA" << (nodeId - 1) << ","
                              << newSize << std::endl;
    }
}

// QoS METRIC: A-MPDU Aggregation Tracking
// The function breaks down each transmission's PSDU map:
// - Iterates through all PSDUs (Protocol Service Data Units) in the transmission
// - For each PSDU, extracts nMpdus (number of MPDUs aggregated in that PSDU)
// - Extracts totalBytes (size of that PSDU)
// - Logs each PSDU separately to the CSV file with its MPDU count and byte size
// This allows analysis of both single-packet (nMpdus=1) and aggregated (nMpdus>1) transmissions, providing visibility into WiFi 6 A-MPDU aggregation behavior.
// txPowerW is a required NS-3 callback parameter (in Watts).
void
AmpduAggregationTrace(std::string context,
                      WifiConstPsduMap psduMap,
                      WifiTxVector txVector,
                      double txPowerW)
{
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms)
    {
        uint32_t nodeId = ContextToNodeId(context);
        uint32_t staIdx = nodeId - 1;

        for (const auto& pair : psduMap)
        {
            Ptr<const WifiPsdu> psdu = pair.second;
            uint32_t nMpdus = psdu->GetNMpdus();
            uint32_t totalBytes = psdu->GetSize();

            // Update cumulative A-MPDU counters
            if (ampduCountForSta)
            {
                ampduCountForSta[staIdx]++;
                ampduMpdusTotalForSta[staIdx] += nMpdus;
                ampduBytesTotalForSta[staIdx] += totalBytes;
            }

            // Update last TX MCS and rate info from TxVector
            if (lastTxMcsForSta)
            {
                WifiMode mode = txVector.GetMode();
                WifiModulationClass modClass = mode.GetModulationClass();
                // Only get MCS value for HT, VHT, or HE modes
                if (modClass == WIFI_MOD_CLASS_HT || modClass == WIFI_MOD_CLASS_VHT ||
                    modClass == WIFI_MOD_CLASS_HE)
                {
                    lastTxMcsForSta[staIdx] = mode.GetMcsValue();
                }
                lastTxNssForSta[staIdx] = txVector.GetNss();
                // Rate in 100 kbps units
                lastTxRate100KbpsForSta[staIdx] =
                    static_cast<uint16_t>(mode.GetDataRate(txVector) / 100000);
            }

            // Update channel width, guard interval, and TX power
            if (lastChannelWidthMhzForSta)
            {
                lastChannelWidthMhzForSta[staIdx] = txVector.GetChannelWidth();
                lastGuardIntervalNsForSta[staIdx] = txVector.GetGuardInterval().GetNanoSeconds();
                // Convert TX power from Watts to dBm: P(dBm) = 10 * log10(P(W) * 1000)
                double txPowerDbm = 10.0 * std::log10(txPowerW * 1000.0);
                lastTxPowerDbmForSta[staIdx] =
                    static_cast<uint8_t>(std::max(0.0, std::min(30.0, txPowerDbm)));
            }

            // Calculate and accumulate airtime in microseconds
            // Airtime = PSDU duration based on MCS, NSS, and channel width
            if (airtimeUsedUsForSta)
            {
                // Use txVector to calculate PPDU duration.
                // Band must match the PHY's actual operating band (2.4 GHz — see ConfigureWifi) so airtime accounting uses the correct PHY timing.
                Time ppduDuration =
                    WifiPhy::CalculateTxDuration(totalBytes, txVector, WIFI_PHY_BAND_2_4GHZ);
                airtimeUsedUsForSta[staIdx] += ppduDuration.GetMicroSeconds();
            }

            if (ampduTraceFile.is_open())
            {
                ampduTraceFile << std::fixed << std::setprecision(6)
                               << Simulator::Now().GetMilliSeconds() << ",STA" << staIdx << ","
                               << nMpdus << "," << totalBytes << "," << std::setprecision(3)
                               << txPowerW << std::endl;
            }
        }
    }
}

// QoEH-REALISTIC: AP-side DL TX accounting. Connected to the AP's PhyTxPsduBegin.
// The AP knows exactly what it transmits, so this is fully AP-observable.
// Per-STA unicast DATA frames feed dl*ToStaForSta (destination resolved via macToStaIdMap); group-addressed frames (beacons, broadcast/multicast) feed the global g_dlBroadcast* diagnostic counters.
// Control/management unicast (ACKs, etc.) are skipped so dl_unicast_count tracks actual DL traffic, not protocol overhead.
void
ApPhyTxPsduBeginTrace(std::string context,
                      WifiConstPsduMap psduMap,
                      WifiTxVector txVector,
                      double txPowerW)
{
    if (Simulator::Now().GetMilliSeconds() < keepTrackOfMetricsFrom_ms)
    {
        return;
    }
    for (const auto& pair : psduMap)
    {
        Ptr<const WifiPsdu> psdu = pair.second;
        if (psdu->GetNMpdus() == 0)
        {
            continue;
        }
        uint32_t totalBytes = psdu->GetSize();
        uint64_t durUs = static_cast<uint64_t>(
            WifiPhy::CalculateTxDuration(totalBytes, txVector, WIFI_PHY_BAND_2_4GHZ)
                .GetMicroSeconds());
        Mac48Address ra = psdu->GetAddr1(); // receiver address (destination)

        if (ra.IsGroup())
        {
            // Broadcast / multicast (beacons, group frames) — diagnostic globals.
            g_dlBroadcastBytes += totalBytes;
            g_dlBroadcastDurationUs += durUs;
            g_dlBroadcastCount++;
            // Timing log for PDW boundary compliance.
            // Tag by frame TYPE (not size): real BEACON (so stray small group frames like ARP aren't mis-counted as beacons by the analyzer), PDW power frames (1400 B broadcast DATA), or GRP (other group-addressed frames).
            const bool isBeaconFrame = psdu->GetHeader(0).IsBeacon();
            if (apTxTraceFile.is_open())
            {
                apTxTraceFile << std::fixed << std::setprecision(6)
                              << (Simulator::Now().GetNanoSeconds() / 1e6) << ","
                              << (isBeaconFrame ? "BEACON" : (totalBytes >= 1000 ? "PDW" : "GRP"))
                              << "," << totalBytes << "," << durUs << std::endl;
            }
            // PDW anchor: each beacon TX kicks off this BI's power-delivery burst (scheduled PDW_BEACON_LEAD after the beacon, so it can never delay it).
            // Anchoring to the real beacon avoids the computed-grid/TBTT misalignment.
            if (g_onApBeaconTx && isBeaconFrame)
            {
                g_onApBeaconTx();
            }
            continue;
        }
        // Unicast: count only DATA frames as DL traffic (skip ACK/BlockAck/mgmt).
        if (!psdu->GetHeader(0).IsData())
        {
            continue;
        }
        if (apTxTraceFile.is_open())
        {
            apTxTraceFile << std::fixed << std::setprecision(6)
                          << (Simulator::Now().GetNanoSeconds() / 1e6) << ",UCAST," << totalBytes
                          << "," << durUs << std::endl;
        }
        std::ostringstream macOss;
        macOss << ra;
        auto it = macToStaIdMap.find(macOss.str());
        if (it == macToStaIdMap.end())
        {
            continue;
        }
        uint32_t staIdx = it->second;
        if (dlBytesToStaForSta)
        {
            dlBytesToStaForSta[staIdx] += totalBytes;
            dlDurationUsToStaForSta[staIdx] += durUs;
            dlUnicastCountToStaForSta[staIdx]++;
        }
    }
}

// ENERGY METRIC: PHY state tracing for energy calculation
// Fired whenever the WiFi PHY transitions to a new state.
// The callback receives the previous state (state parameter) and the duration it was active (duration).
// Example: If radio was in IDLE for 100µs then transitions to TX, callback fires with state=IDLE, duration=100µs.
// This allows tracking total time spent in each state across the simulation for energy consumption calculations.
// Energy is computed as: (state_current_mA × state_duration_ms) and accumulated per STA to derive total battery consumption at end of simulation.
void
PhyStateTrace_inPlace(std::string context, Time start, Time duration, WifiPhyState state)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        std::ostringstream oss;
        oss << state;
        uint32_t nodeId = ContextToNodeId(context);

        double duration_ms = duration.GetMilliSeconds();
        timeElapsedForSta_ms_TI[nodeId - 1] += duration_ms;

        if (oss.str() != "SLEEP")
        {
            awakeTimeElapsedForSta_ms_TI[nodeId - 1] += duration_ms;
        }
        else
        {
            sleepTimeElapsedForSta_ms_TI[nodeId - 1] += duration_ms;

            // QoEH-realistic SP-boundary handling.
            // This callback fires with the state that just ENDED: a SLEEP record means the STA was asleep from `start` (= the prior SP / awake-window end) until Now() (= the next SP / awake-window start).
            // So each SLEEP record both *finalizes* the SP that just ended and *opens* the next one.
            // (D1: PHY-sleep edges as SP boundaries; short awake windows are beacon-RX blips, not SPs.)
            //
            // REALISM — do NOT reclassify `starv_rate` as oracle (settled 2026-06-21).
            // sp_with_starvation_count / sp_with_demand_count back the REALISTIC actor obs feature `starv_rate`, and ALL THREE of its inputs are AP-side:
            //   * SP window  — AP-DEFINED. The AP authored the unilateral TWT schedule (per-STA offset tau_k + wake duration d_k), so it already KNOWS every SP boundary by construction.
            //     STAs comply with their assigned wake windows (verified, _check_twt_compliance.py: periodic wake exactly at the assigned offset, std +/-0.3 ms, never awake longer than scheduled), so the STA PHY-sleep edge read here is IDENTICAL to the AP's own published-schedule clock.
            //     Reading it off the PHY State trace is an implementation convenience, NOT a ground-truth / oracle dependency.
            //   * demand     — last-reported 802.11ax BSR (real signaling, AP-observed).
            //   * served     — bytes the AP itself received from the STA in the window.
            // => starv_rate is legitimately AP-observable and stays in the actor obs.
            uint32_t staIdx = nodeId - 1;
            if (spInitedForSta)
            {
                uint64_t nowUs = static_cast<uint64_t>(Simulator::Now().GetMicroSeconds());
                uint64_t spEndUs = static_cast<uint64_t>(start.GetMicroSeconds());
                if (spInitedForSta[staIdx])
                {
                    uint64_t awakeUs = (spEndUs > spWakeStartUsForSta[staIdx])
                                           ? spEndUs - spWakeStartUsForSta[staIdx]
                                           : 0;
                    if (awakeUs >= SP_MIN_AWAKE_US)
                    {
                        spCompletedCountForSta[staIdx]++;
                        uint32_t demand = spDemandBytesForSta[staIdx];
                        uint64_t served = spCurServedBytesForSta[staIdx];
                        bytesRxAtApInSpForSta[staIdx] += served;
                        packetsRxAtApInSpForSta[staIdx] += spCurServedPktsForSta[staIdx];
                        if (demand >= SP_DEMAND_MIN_BYTES)
                        {
                            spWithDemandCountForSta[staIdx]++;
                            if (served < static_cast<uint64_t>(SP_STARVATION_RATIO * demand))
                            {
                                spWithStarvationCountForSta[staIdx]++;
                            }
                        }
                    }
                }
                // Open the next SP at Now(): snapshot demand (last-reported BSR) and
                // reset the served accumulators.
                uint32_t bsrTotal = 0;
                if (bsrQueueBytesAcBeForSta)
                {
                    bsrTotal = bsrQueueBytesAcBeForSta[staIdx] + bsrQueueBytesAcBkForSta[staIdx] +
                               bsrQueueBytesAcViForSta[staIdx] + bsrQueueBytesAcVoForSta[staIdx];
                }
                spDemandBytesForSta[staIdx] = bsrTotal;
                spCurServedBytesForSta[staIdx] = 0;
                spCurServedPktsForSta[staIdx] = 0;
                spWakeStartUsForSta[staIdx] = nowUs;
                spInitedForSta[staIdx] = true;
            }
        }

        current_mA_TimesTime_ms_ForSta_TI[nodeId - 1] +=
            TI_currentModel_mA[oss.str()] * duration_ms;

        // Log PHY state transition to CSV
        // CSV format: Time_ms,Node,State,Duration_ms
        // - Time_ms: simulation time when this state transition occurred (nanosecond precision)
        // - Node: which STA experienced the state transition (e.g., STA0, STA1)
        // - State: the PHY state that just ENDED (IDLE, RX, TX, CCA_BUSY, SLEEP)
        // - Duration_ms: how long the radio was in that state (in milliseconds)
        if (phyStateTraceFile.is_open())
        {
            phyStateTraceFile << std::fixed << std::setprecision(6)
                              << (Simulator::Now().GetNanoSeconds() / 1e6) << ",STA" << (nodeId - 1)
                              << "," << oss.str() << "," << duration_ms << std::endl;
        }
    }
}

// RELIABILITY METRIC: PSDU Response Timeout Trace - Uplink (STA→AP)
// Oracle metric: this callback fires at STA's MAC layer when STA doesn't get ACK.
// AP cannot observe this - it fires at the transmitter (STA), not receiver (AP).
// Each timeout triggers a retry, so this is effectively the retry count.
// Kept as ORACLE for ground-truth analysis and RL training.
void
PsduResponseTimeoutTraceSta(std::string context,
                            uint8_t reason,
                            Ptr<const WifiPsdu> psdu,
                            const WifiTxVector& txVector)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        uint32_t staIdx = ContextToNodeId(context) - 1;
        uplinkTimeoutsForSta[staIdx]++;

        // Each timeout triggers a retry - increment retry counter
        if (txRetryCountForSta)
        {
            txRetryCountForSta[staIdx]++;
        }

        // Log to trace file
        if (timeoutDropTraceFile.is_open())
        {
            timeoutDropTraceFile << std::fixed << std::setprecision(6)
                                 << Simulator::Now().GetMilliSeconds() << ",STA" << staIdx
                                 << ",PSDU_TIMEOUT," << +reason << std::endl;
        }
    }
}

// RELIABILITY METRIC: PSDU Response Timeout Trace - Downlink (AP→STA)
// Fired when AP transmission times out waiting for STA ACK.
// AP sends PSDU → waits for ACK → if ACK doesn't arrive within timeout window, this callback fires and increments the aggregate downlink timeout counter.
// Same timeout mechanism as uplink but from AP's perspective (sending to all STAs).
void
PsduResponseTimeoutTraceAp(std::string context,
                           uint8_t reason,
                           Ptr<const WifiPsdu> psdu,
                           const WifiTxVector& txVector)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        downlinkTimeoutsAllSta++;

        // Log to trace file
        if (timeoutDropTraceFile.is_open())
        {
            timeoutDropTraceFile << std::fixed << std::setprecision(6)
                                 << Simulator::Now().GetMilliSeconds() << ",AP"
                                 << ",PSDU_TIMEOUT_DL," << +reason << std::endl;
        }
    }
}

// QoS METRIC: MPDU Drop Traces - Uplink (STA packets)
// Tracks individual MPDU (MAC Protocol Data Unit) packet drops at STA.
// Distinguishes between two drop reasons:
// 1. EXPIRED_LIFETIME: Packet sat in queue too long (TTL exceeded); indicates queue congestion or slow transmission rate; packet validity expired before transmission.
// 2. FAILED_ENQUEUE: Queue was full (MAX_QUEUE_SIZE_BYTES exceeded), new packet rejected; indicates insufficient buffer capacity or backlog.
//
// Difference from PSDU timeouts: MPDU drops track individual packet rejections/expiries before or after queuing, while PSDU timeouts track frame-level transmission failures (missing ACKs).
// A packet may be dropped as MPDU before it's even assembled into a PSDU.
void
MpduDropped_atSta(std::string context, WifiMacDropReason dropReason, Ptr<const WifiMpdu> mpdu)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        uint32_t staIdx = ContextToNodeId(context) - 1;
        std::string reasonStr;

        if (dropReason == WifiMacDropReason::WIFI_MAC_DROP_EXPIRED_LIFETIME)
        {
            uplinkExpiredMpduForSta[staIdx]++;
            reasonStr = "EXPIRED_LIFETIME";
        }
        if (dropReason == WifiMacDropReason::WIFI_MAC_DROP_FAILED_ENQUEUE)
        {
            uplinkFailedEnqueueMpduForSta[staIdx]++;
            reasonStr = "FAILED_ENQUEUE";
        }

        // Log to trace file
        if (timeoutDropTraceFile.is_open() && !reasonStr.empty())
        {
            timeoutDropTraceFile << std::fixed << std::setprecision(6)
                                 << Simulator::Now().GetMilliSeconds() << ",STA" << staIdx
                                 << ",MPDU_DROP," << reasonStr << std::endl;
        }
    }
}

// QoS METRIC: MPDU Drop Traces - Downlink (AP packets to STAs)
// Tracks individual MPDU packet drops at AP with same two drop reasons:
// 1. EXPIRED_LIFETIME: AP packet validity expired while buffered (congestion).
// 2. FAILED_ENQUEUE: AP queue full, couldn't buffer packet for STA.
// Aggregate counter tracks total drops across all STAs (not per-STA breakdown).
void
MpduDropped_atAp(std::string context, WifiMacDropReason dropReason, Ptr<const WifiMpdu> mpdu)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        std::string reasonStr;

        if (dropReason == WifiMacDropReason::WIFI_MAC_DROP_EXPIRED_LIFETIME)
        {
            downlinkExpiredMpduAllSta++;
            reasonStr = "EXPIRED_LIFETIME";
        }
        if (dropReason == WifiMacDropReason::WIFI_MAC_DROP_FAILED_ENQUEUE)
        {
            downlinkFailedEnqueueMpduAllSta++;
            reasonStr = "FAILED_ENQUEUE";
        }

        // Log to trace file
        if (timeoutDropTraceFile.is_open() && !reasonStr.empty())
        {
            timeoutDropTraceFile << std::fixed << std::setprecision(6)
                                 << Simulator::Now().GetMilliSeconds() << ",AP"
                                 << ",MPDU_DROP_DL," << reasonStr << std::endl;
        }
    }
}

// 802.11k LINK MEASUREMENT: PHY RX End Trace (STA-side, for downlink quality)
// Oracle metric: this traces at STA's PHY - measures what STA receives FROM AP.
// Kept for oracle metrics (STA's view of channel quality).
// The AP-observable version is ApPhyRxEndTrace below.
void
PhyRxEndTrace(std::string context,
              Ptr<const WifiPsdu> psdu,
              RxSignalInfo rxSignalInfo,
              const WifiTxVector& txVector,
              const std::vector<bool>& perMpduStatus)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        uint32_t nodeId = ContextToNodeId(context);
        uint32_t staIdx = nodeId - 1;

        // Only process for valid STA indices (this is STA-side trace)
        if (staIdx >= MAX_NUM_STA)
        {
            return;
        }

        // Update link measurement values (STA's view of downlink)
        if (lastRssiDbmForSta && lastSnrDbForSta)
        {
            double snrDb = 10.0 * std::log10(rxSignalInfo.snr);
            double rssiDbm = rxSignalInfo.rssi;
            double noiseDbm = rssiDbm - snrDb;

            lastSnrDbForSta[staIdx] = snrDb;
            lastRssiDbmForSta[staIdx] = rssiDbm;
            lastNoiseDbmForSta[staIdx] = noiseDbm;
        }
    }
}

// AP PHY RX End Trace - AP receives frames FROM STAs (UPLINK)
// Realistic metric: this is the AP's view of uplink traffic.
// Connected to AP's PHY, extracts source MAC to identify which STA sent the frame.
// Populates: rx_fragment_count, fcs_error_count, RSSI/SNR, MCS, frame type, etc.
void
ApPhyRxEndTrace(std::string context,
                Ptr<const WifiPsdu> psdu,
                RxSignalInfo rxSignalInfo,
                const WifiTxVector& txVector,
                const std::vector<bool>& perMpduStatus)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        // Extract source MAC address from the first MPDU
        if (psdu->GetNMpdus() == 0)
        {
            return;
        }

        Ptr<const WifiMpdu> firstMpdu = *psdu->begin();
        const WifiMacHeader& hdr = firstMpdu->GetHeader();
        Mac48Address srcAddr = hdr.GetAddr2(); // Addr2 is transmitter address

        // Look up STA ID from MAC address
        std::ostringstream macOss;
        macOss << srcAddr;
        std::string macStr = macOss.str();
        auto it = macToStaIdMap.find(macStr);
        if (it == macToStaIdMap.end())
        {
            // Unknown source (could be from AP itself or unregistered device)
            return;
        }
        uint32_t staIdx = it->second;

        // Count successful/failed MPDUs for this STA
        bool sawSuccess = false;
        if (rxSuccessCountForSta && rxErrorCountForSta)
        {
            for (bool success : perMpduStatus)
            {
                if (success)
                {
                    rxSuccessCountForSta[staIdx]++;
                    sawSuccess = true;
                }
                else
                {
                    rxErrorCountForSta[staIdx]++;
                }
            }
        }
        // Stamp last-UL timestamp for silence-since-last-rx obs feature.
        if (sawSuccess && lastRxTimestampUsForSta)
        {
            lastRxTimestampUsForSta[staIdx] =
                static_cast<uint64_t>(Simulator::Now().GetMicroSeconds());
        }

        // Track per-AC received traffic (from QoS Control TID in data frames)
        // Only count successful MPDUs in QoS Data frames
        for (auto mpduIt = psdu->begin(); mpduIt != psdu->end(); ++mpduIt)
        {
            Ptr<const WifiMpdu> mpdu = *mpduIt;
            const WifiMacHeader& mpduHdr = mpdu->GetHeader();

            // Only process QoS Data frames (type=2, has QoS field)
            if (mpduHdr.IsQosData())
            {
                uint8_t tid = mpduHdr.GetQosTid();
                uint32_t mpduSize = mpdu->GetSize();

                // Map TID to Access Category (IEEE 802.11-2020 Table 10-1)
                // TID 1,2 → AC_BK, TID 0,3 → AC_BE, TID 4,5 → AC_VI, TID 6,7 → AC_VO
                if (tid == 1 || tid == 2)
                {
                    // AC_BK (Background)
                    if (bytesReceivedAcBkForSta)
                    {
                        bytesReceivedAcBkForSta[staIdx] += mpduSize;
                    }
                    if (packetsReceivedAcBkForSta)
                    {
                        packetsReceivedAcBkForSta[staIdx]++;
                    }
                }
                else if (tid == 0 || tid == 3)
                {
                    // AC_BE (Best Effort)
                    if (bytesReceivedAcBeForSta)
                    {
                        bytesReceivedAcBeForSta[staIdx] += mpduSize;
                    }
                    if (packetsReceivedAcBeForSta)
                    {
                        packetsReceivedAcBeForSta[staIdx]++;
                    }
                }
                else if (tid == 4 || tid == 5)
                {
                    // AC_VI (Video)
                    if (bytesReceivedAcViForSta)
                    {
                        bytesReceivedAcViForSta[staIdx] += mpduSize;
                    }
                    if (packetsReceivedAcViForSta)
                    {
                        packetsReceivedAcViForSta[staIdx]++;
                    }
                }
                else // tid == 6 || tid == 7
                {
                    // AC_VO (Voice)
                    if (bytesReceivedAcVoForSta)
                    {
                        bytesReceivedAcVoForSta[staIdx] += mpduSize;
                    }
                    if (packetsReceivedAcVoForSta)
                    {
                        packetsReceivedAcVoForSta[staIdx]++;
                    }
                }
            }
            else if (mpduHdr.IsData())
            {
                // Non-QoS data frames default to AC_BE
                uint32_t mpduSize = mpdu->GetSize();
                if (bytesReceivedAcBeForSta)
                {
                    bytesReceivedAcBeForSta[staIdx] += mpduSize;
                }
                if (packetsReceivedAcBeForSta)
                {
                    packetsReceivedAcBeForSta[staIdx]++;
                }
            }
        }

        // Update link measurement values (AP's view of uplink from this STA)
        if (lastRssiDbmForSta && lastSnrDbForSta)
        {
            double snrDb = 10.0 * std::log10(rxSignalInfo.snr);
            double rssiDbm = rxSignalInfo.rssi;
            double noiseDbm = rssiDbm - snrDb;

            lastSnrDbForSta[staIdx] = snrDb;
            lastRssiDbmForSta[staIdx] = rssiDbm;
            lastNoiseDbmForSta[staIdx] = noiseDbm;
        }

        // Update frame type and power management bit
        if (lastFrameTypeForSta)
        {
            lastFrameTypeForSta[staIdx] =
                static_cast<uint8_t>(hdr.GetType() >> 2); // Type is bits 2-3
            lastFrameSubtypeForSta[staIdx] = static_cast<uint8_t>((hdr.GetType() >> 4) & 0x0F);
            lastPowerMgmtBitForSta[staIdx] = hdr.IsPowerManagement() ? 1 : 0;
        }

        // Update PHY parameters from TX vector
        // Note: GetMcsValue() asserts for non-MCS modes (legacy/control frames)
        if (lastTxMcsForSta && lastTxNssForSta)
        {
            WifiMode mode = txVector.GetMode();
            WifiModulationClass modClass = mode.GetModulationClass();
            // Only get MCS value for HT, VHT, or HE modes
            if (modClass == WIFI_MOD_CLASS_HT || modClass == WIFI_MOD_CLASS_VHT ||
                modClass == WIFI_MOD_CLASS_HE)
            {
                lastTxMcsForSta[staIdx] = mode.GetMcsValue();
            }
            lastTxNssForSta[staIdx] = txVector.GetNss();
            lastChannelWidthMhzForSta[staIdx] = txVector.GetChannelWidth();
            lastGuardIntervalNsForSta[staIdx] = txVector.GetGuardInterval().GetNanoSeconds();
        }

        // Log link measurement to trace file
        if (linkMeasurementTraceFile.is_open())
        {
            double snrDb = 10.0 * std::log10(rxSignalInfo.snr);
            double rssiDbm = rxSignalInfo.rssi;
            WifiMode mode = txVector.GetMode();
            WifiModulationClass modClass = mode.GetModulationClass();
            uint8_t mcs = 0;
            // Only get MCS value for HT, VHT, or HE modes
            if (modClass == WIFI_MOD_CLASS_HT || modClass == WIFI_MOD_CLASS_VHT ||
                modClass == WIFI_MOD_CLASS_HE)
            {
                mcs = mode.GetMcsValue();
            }
            uint8_t nss = txVector.GetNss();
            uint64_t rateKbps = mode.GetDataRate(txVector) / 100;

            linkMeasurementTraceFile
                << std::fixed << std::setprecision(6) << Simulator::Now().GetMilliSeconds()
                << ",STA" << staIdx << "," << std::setprecision(2) << rssiDbm << "," << snrDb << ","
                << mcs << "," << nss << "," << rateKbps << std::endl;
        }
    }
}

// PHY RX Drop Trace (STA-side) - for oracle metrics
// Oracle metric: this traces at STA's PHY - frames STA failed to receive from AP.
void
PhyRxDropTrace(std::string context, Ptr<const WifiPsdu> psdu, WifiPhyRxfailureReason reason)
{
    // STA-side drop trace - not used for AP-observable metrics
    // Kept for potential oracle analysis
}

// AP Monitor Sniffer RX Trace - Uses MonitorSnifferRx which exists in NS-3.44
// Realistic metric: AP's view of uplink traffic via monitor mode sniffer.
// Signature: (Ptr<const Packet>, uint16_t channelFreqMhz, WifiTxVector, MpduInfo, SignalNoiseDbm, uint16_t staId).
// Populates: rx_fragment_count, RSSI/SNR, MCS, NSS, channel width, etc.
void
ApMonitorSnifferRxTrace(std::string context,
                        Ptr<const Packet> packet,
                        uint16_t channelFreqMhz,
                        WifiTxVector txVector,
                        MpduInfo aMpdu,
                        SignalNoiseDbm signalNoise,
                        uint16_t staId)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        // Extract MAC header from packet to get source address.
        Ptr<Packet> pktCopy = packet->Copy();

        // A-MPDU FIX: for aggregated PPDUs, WifiPhy::NotifyMonitorSniffRx hands us each subframe via WifiPsdu::GetAmpduSubframe(), which PREPENDS a 4-byte AmpduSubframeHeader (delimiter) + trailing padding.
        // In 802.11ax virtually all DATA is sent as (single-)A-MPDU, so without stripping that delimiter, PeekHeader<WifiMacHeader> parses the delimiter as the MAC header → garbage Addr2 → macToStaIdMap miss → the frame is dropped here.
        // That silently zeroed the AP-side realistic UL counters (bytes_received_at_ap, rx_fragment_count, per-AC bytes) for all aggregated traffic.
        // Strip the delimiter and trim padding to the real MPDU first (mirrors ns-3's own wifi-helper.cc radiotap path).
        // aMpdu.type==NORMAL_MPDU means a legacy non-aggregated frame.
        if (txVector.IsAggregation())
        {
            AmpduSubframeHeader subHdr;
            pktCopy->RemoveHeader(subHdr);
            uint32_t mpduLen = subHdr.GetLength();
            if (mpduLen > 0 && mpduLen <= pktCopy->GetSize())
            {
                pktCopy = pktCopy->CreateFragment(0, mpduLen); // drop A-MPDU padding
            }
        }

        WifiMacHeader hdr;
        pktCopy->PeekHeader(hdr);

        Mac48Address srcAddr = hdr.GetAddr2(); // Addr2 is transmitter address

        // Look up STA ID from MAC address
        std::ostringstream macOss;
        macOss << srcAddr;
        std::string macStr = macOss.str();
        auto it = macToStaIdMap.find(macStr);
        if (it == macToStaIdMap.end())
        {
            // Unknown source (could be from AP itself or unregistered device)
            return;
        }
        uint32_t staIdx = it->second;

        // Count successful RX for this STA
        if (rxSuccessCountForSta)
        {
            rxSuccessCountForSta[staIdx]++;
        }
        // Stamp last-UL timestamp for silence-since-last-rx obs feature.
        if (lastRxTimestampUsForSta)
        {
            lastRxTimestampUsForSta[staIdx] =
                static_cast<uint64_t>(Simulator::Now().GetMicroSeconds());
        }

        // Track per-AC received traffic (from QoS Control TID in data frames).
        // Use pktCopy (the real MPDU, post A-MPDU-delimiter strip), not the raw `packet` which still carries the 4-byte delimiter + aggregation padding.
        if (hdr.IsQosData())
        {
            uint8_t tid = hdr.GetQosTid();
            uint32_t pktSize = pktCopy->GetSize();

            // Map TID to Access Category (IEEE 802.11-2020 Table 10-1)
            if (tid == 1 || tid == 2)
            {
                // AC_BK (Background)
                if (bytesReceivedAcBkForSta)
                {
                    bytesReceivedAcBkForSta[staIdx] += pktSize;
                }
                if (packetsReceivedAcBkForSta)
                {
                    packetsReceivedAcBkForSta[staIdx]++;
                }
            }
            else if (tid == 0 || tid == 3)
            {
                // AC_BE (Best Effort)
                if (bytesReceivedAcBeForSta)
                {
                    bytesReceivedAcBeForSta[staIdx] += pktSize;
                }
                if (packetsReceivedAcBeForSta)
                {
                    packetsReceivedAcBeForSta[staIdx]++;
                }
            }
            else if (tid == 4 || tid == 5)
            {
                // AC_VI (Video)
                if (bytesReceivedAcViForSta)
                {
                    bytesReceivedAcViForSta[staIdx] += pktSize;
                }
                if (packetsReceivedAcViForSta)
                {
                    packetsReceivedAcViForSta[staIdx]++;
                }
            }
            else // tid == 6 || tid == 7
            {
                // AC_VO (Voice)
                if (bytesReceivedAcVoForSta)
                {
                    bytesReceivedAcVoForSta[staIdx] += pktSize;
                }
                if (packetsReceivedAcVoForSta)
                {
                    packetsReceivedAcVoForSta[staIdx]++;
                }
            }
        }
        else if (hdr.IsData())
        {
            // Non-QoS data frames default to AC_BE
            uint32_t pktSize = pktCopy->GetSize();
            if (bytesReceivedAcBeForSta)
            {
                bytesReceivedAcBeForSta[staIdx] += pktSize;
            }
            if (packetsReceivedAcBeForSta)
            {
                packetsReceivedAcBeForSta[staIdx]++;
            }
        }

        // QoEH-realistic per-SP UL accounting: fold this UL DATA frame into the STA's in-progress service-period accumulator.
        // The SP is finalized (and its served-vs-demand compared for starvation) at the SP boundary in PhyStateTrace_inPlace.
        // Only data frames count toward "served".
        if (hdr.IsData() && spCurServedBytesForSta && spCurServedPktsForSta)
        {
            spCurServedBytesForSta[staIdx] += pktCopy->GetSize();
            spCurServedPktsForSta[staIdx]++;
        }

        // Update link measurement values from SignalNoiseDbm.
        // signalNoise.signal is RSSI in dBm, signalNoise.noise is noise floor.
        if (lastRssiDbmForSta && lastSnrDbForSta)
        {
            double rssiDbm = static_cast<double>(signalNoise.signal);
            double noiseDbm = static_cast<double>(signalNoise.noise);
            double snrDb = rssiDbm - noiseDbm;

            lastSnrDbForSta[staIdx] = snrDb;
            lastRssiDbmForSta[staIdx] = rssiDbm;
            lastNoiseDbmForSta[staIdx] = noiseDbm;
        }

        // Update frame type and power management bit
        if (lastFrameTypeForSta)
        {
            lastFrameTypeForSta[staIdx] = static_cast<uint8_t>(hdr.GetType() >> 2);
            lastFrameSubtypeForSta[staIdx] = static_cast<uint8_t>((hdr.GetType() >> 4) & 0x0F);
            lastPowerMgmtBitForSta[staIdx] = hdr.IsPowerManagement() ? 1 : 0;
        }

        // Update PHY parameters from TX vector
        // Note: GetMcsValue() asserts for non-MCS modes (legacy/control frames)
        // Only update MCS if this is an HE/VHT/HT mode
        if (lastTxMcsForSta && lastTxNssForSta)
        {
            WifiMode mode = txVector.GetMode();
            WifiModulationClass modClass = mode.GetModulationClass();
            // Only get MCS value for HT, VHT, or HE modes
            if (modClass == WIFI_MOD_CLASS_HT || modClass == WIFI_MOD_CLASS_VHT ||
                modClass == WIFI_MOD_CLASS_HE)
            {
                lastTxMcsForSta[staIdx] = mode.GetMcsValue();
            }
            lastTxNssForSta[staIdx] = txVector.GetNss();
            lastChannelWidthMhzForSta[staIdx] = txVector.GetChannelWidth();
            lastGuardIntervalNsForSta[staIdx] = txVector.GetGuardInterval().GetNanoSeconds();
        }

        // Log link measurement to trace file
        if (linkMeasurementTraceFile.is_open())
        {
            double rssiDbm = static_cast<double>(signalNoise.signal);
            double noiseDbm = static_cast<double>(signalNoise.noise);
            double snrDb = rssiDbm - noiseDbm;
            WifiMode mode = txVector.GetMode();
            WifiModulationClass modClass = mode.GetModulationClass();
            uint8_t mcs = 0;
            // Only get MCS value for HT, VHT, or HE modes
            if (modClass == WIFI_MOD_CLASS_HT || modClass == WIFI_MOD_CLASS_VHT ||
                modClass == WIFI_MOD_CLASS_HE)
            {
                mcs = mode.GetMcsValue();
            }
            uint8_t nss = txVector.GetNss();
            uint64_t rateKbps = mode.GetDataRate(txVector) / 100;

            linkMeasurementTraceFile
                << std::fixed << std::setprecision(6) << Simulator::Now().GetMilliSeconds()
                << ",STA" << staIdx << "," << std::setprecision(2) << rssiDbm << "," << snrDb << ","
                << +mcs << "," << +nss << "," << rateKbps << std::endl;
        }
    }
}

// AP PHY RX Drop Trace (simplified) - uses PhyRxDrop signature
// Realistic metric: AP counts RX failures.
// Uses Ptr<const Packet> signature.
void
ApPhyRxDropTraceSimple(std::string context, Ptr<const Packet> packet, WifiPhyRxfailureReason reason)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        // Try to extract source address from packet if possible
        WifiMacHeader hdr;
        Ptr<Packet> pktCopy = packet->Copy();
        if (pktCopy->PeekHeader(hdr))
        {
            Mac48Address srcAddr = hdr.GetAddr2();

            std::ostringstream macOss;
            macOss << srcAddr;
            std::string macStr = macOss.str();
            auto it = macToStaIdMap.find(macStr);
            if (it != macToStaIdMap.end() && rxErrorCountForSta)
            {
                uint32_t staIdx = it->second;
                rxErrorCountForSta[staIdx]++;
            }
        }
        // Note: If frame is too corrupted to parse, we can't attribute it to a STA
    }
}

// AP PHY RX Drop Trace - AP failed to receive frame (FCS error, etc.)
// Realistic metric: AP directly counts RX failures.
// However, for dropped frames we often cannot identify the source STA (frame is corrupted).
// This increments a global AP RX error counter rather than per-STA.
void
ApPhyRxDropTrace(std::string context, Ptr<const WifiPsdu> psdu, WifiPhyRxfailureReason reason)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        // For dropped frames, try to extract source address if possible
        if (psdu && psdu->GetNMpdus() > 0)
        {
            Ptr<const WifiMpdu> firstMpdu = *psdu->begin();
            const WifiMacHeader& hdr = firstMpdu->GetHeader();
            Mac48Address srcAddr = hdr.GetAddr2();

            std::ostringstream macOss;
            macOss << srcAddr;
            std::string macStr = macOss.str();
            auto it = macToStaIdMap.find(macStr);
            if (it != macToStaIdMap.end() && rxErrorCountForSta)
            {
                uint32_t staIdx = it->second;
                rxErrorCountForSta[staIdx]++;
            }
        }
        // Note: If frame is too corrupted to parse, we can't attribute it to a STA
    }
}

// 802.11k STA STATISTICS: Successful TX (got ACK)
// Oracle metric: this callback fires at the STA's MAC layer when STA successfully transmits (receives ACK from AP).
// This is STA-side local counter (dot11TransmittedFragmentCount).
// AP cannot directly observe this - would require 802.11k STA Statistics Request/Response.
// Kept as ORACLE for ground-truth analysis and RL training.
void
MacTxOkTrace(std::string context, Ptr<const WifiMpdu> mpdu)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        uint32_t nodeId = ContextToNodeId(context);
        uint32_t staIdx = nodeId - 1;

        if (txSuccessCountForSta)
        {
            txSuccessCountForSta[staIdx]++;
        }

        // MAC queue-sojourn latency: time from enqueue (mpdu construction) to successful ACK.
        if (latencySumMsForSta && latencyCountForSta && mpdu)
        {
            double soj_ms = (Simulator::Now() - mpdu->GetTimestamp()).GetMicroSeconds() / 1000.0;
            if (soj_ms >= 0.0)
            {
                latencySumMsForSta[staIdx] += soj_ms;
                latencyCountForSta[staIdx]++;
            }
        }

        // Log to trace file
        if (txRxStatsTraceFile.is_open())
        {
            txRxStatsTraceFile << std::fixed << std::setprecision(6)
                               << Simulator::Now().GetMilliSeconds() << ",STA" << staIdx
                               << ",TX_SUCCESS,1" << std::endl;
        }
    }
}

// 802.11k STA STATISTICS: TX Drop (max retries exceeded or other failure)
// Oracle metric: this callback fires at the STA's MAC layer when STA's transmission fails (max retries exceeded).
// This is STA-side local counter (dot11FailedCount).
// AP cannot directly observe this - would require 802.11k STA Statistics Request/Response.
// Kept as ORACLE for ground-truth analysis and RL training.
void
MacTxDropTrace(std::string context, WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu)
{
    if (Simulator::Now().GetMilliSeconds() > keepTrackOfMetricsFrom_ms)
    {
        uint32_t nodeId = ContextToNodeId(context);
        uint32_t staIdx = nodeId - 1;

        if (txFailedCountForSta)
        {
            txFailedCountForSta[staIdx]++;
        }

        // Log to trace file
        if (txRxStatsTraceFile.is_open())
        {
            txRxStatsTraceFile << std::fixed << std::setprecision(6)
                               << Simulator::Now().GetMilliSeconds() << ",STA" << staIdx
                               << ",TX_FAILED," << static_cast<int>(reason) << std::endl;
        }
    }
}

// Initialize 802.11k tracking arrays
void
Initialize802dot11kArrays(uint32_t numSta)
{
    // 802.11k STA Statistics
    txSuccessCountForSta = new uint64_t[numSta]();
    txRetryCountForSta = new uint64_t[numSta]();
    txFailedCountForSta = new uint64_t[numSta]();
    rxSuccessCountForSta = new uint64_t[numSta]();
    rxErrorCountForSta = new uint64_t[numSta]();
    lastRxTimestampUsForSta = new uint64_t[numSta](); // zero-init = "no UL seen yet"

    // 802.11k Link Measurement
    lastRssiDbmForSta = new double[numSta]();
    lastSnrDbForSta = new double[numSta]();
    lastNoiseDbmForSta = new double[numSta]();
    lastTxMcsForSta = new uint8_t[numSta]();
    lastTxNssForSta = new uint8_t[numSta]();
    lastTxRate100KbpsForSta = new uint16_t[numSta]();

    // A-MPDU aggregation
    ampduCountForSta = new uint64_t[numSta]();
    ampduMpdusTotalForSta = new uint64_t[numSta]();
    ampduBytesTotalForSta = new uint64_t[numSta]();

    // Airtime and PHY parameters
    airtimeUsedUsForSta = new uint64_t[numSta]();
    lastChannelWidthMhzForSta = new uint8_t[numSta]();
    lastGuardIntervalNsForSta = new uint16_t[numSta]();
    lastTxPowerDbmForSta = new uint8_t[numSta]();
    lastFrameTypeForSta = new uint8_t[numSta]();
    lastFrameSubtypeForSta = new uint8_t[numSta]();
    lastPowerMgmtBitForSta = new uint8_t[numSta]();

    // 802.11ax BSR per-AC arrays
    bsrQueueBytesAcBeForSta = new uint32_t[numSta]();
    bsrQueueBytesAcBkForSta = new uint32_t[numSta]();
    bsrQueueBytesAcViForSta = new uint32_t[numSta]();
    bsrQueueBytesAcVoForSta = new uint32_t[numSta]();

    // AP-received traffic per-AC arrays
    bytesReceivedAcBeForSta = new uint64_t[numSta]();
    bytesReceivedAcBkForSta = new uint64_t[numSta]();
    bytesReceivedAcViForSta = new uint64_t[numSta]();
    bytesReceivedAcVoForSta = new uint64_t[numSta]();
    packetsReceivedAcBeForSta = new uint64_t[numSta]();
    packetsReceivedAcBkForSta = new uint64_t[numSta]();
    packetsReceivedAcViForSta = new uint64_t[numSta]();
    packetsReceivedAcVoForSta = new uint64_t[numSta]();

    // QoEH-realistic: per-SP UL accounting + per-SP working state + DL accounting
    bytesRxAtApInSpForSta = new uint64_t[numSta]();
    packetsRxAtApInSpForSta = new uint64_t[numSta]();
    spCompletedCountForSta = new uint32_t[numSta]();
    spWithDemandCountForSta = new uint32_t[numSta]();
    spWithStarvationCountForSta = new uint32_t[numSta]();
    spCurServedBytesForSta = new uint64_t[numSta]();
    spCurServedPktsForSta = new uint64_t[numSta]();
    spWakeStartUsForSta = new uint64_t[numSta]();
    spDemandBytesForSta = new uint32_t[numSta]();
    spInitedForSta = new bool[numSta]();
    dlBytesToStaForSta = new uint64_t[numSta]();
    dlDurationUsToStaForSta = new uint64_t[numSta]();
    dlUnicastCountToStaForSta = new uint32_t[numSta]();

    // Initialize with NAN for double arrays (no valid measurement yet) and 0 for integer counters.
    // NAN indicates "no data received" which will be converted to INT8_MIN (-128) sentinel in the struct.
    for (uint32_t i = 0; i < numSta; i++)
    {
        // Use NAN for measurement arrays - indicates no measurement received yet
        lastRssiDbmForSta[i] = NAN;
        lastSnrDbForSta[i] = NAN;
        lastNoiseDbmForSta[i] = NAN;
        // Integer arrays default to 0 (no frames observed)
        lastTxMcsForSta[i] = 0;
        lastTxNssForSta[i] = 0;
        lastTxRate100KbpsForSta[i] = 0;
        lastChannelWidthMhzForSta[i] = 0;
        lastGuardIntervalNsForSta[i] = 0;
        lastTxPowerDbmForSta[i] = 0;
        lastFrameTypeForSta[i] = 0;
        lastFrameSubtypeForSta[i] = 0;
        lastPowerMgmtBitForSta[i] = 0;
        // BSR defaults to 0 (empty queues or no BSR received)
        bsrQueueBytesAcBeForSta[i] = 0;
        bsrQueueBytesAcBkForSta[i] = 0;
        bsrQueueBytesAcViForSta[i] = 0;
        bsrQueueBytesAcVoForSta[i] = 0;
        // RX per-AC defaults to 0 (no traffic received yet)
        bytesReceivedAcBeForSta[i] = 0;
        bytesReceivedAcBkForSta[i] = 0;
        bytesReceivedAcViForSta[i] = 0;
        bytesReceivedAcVoForSta[i] = 0;
        packetsReceivedAcBeForSta[i] = 0;
        packetsReceivedAcBkForSta[i] = 0;
        packetsReceivedAcViForSta[i] = 0;
        packetsReceivedAcVoForSta[i] = 0;
    }
}

// 802.11ax BSR (Buffer Status Report) Callback
// Realistic metric: this is 802.11ax BSR protocol data.
// STA explicitly sends BSR to AP (in Trigger Frame response or QoS Data frame).
// AP receives and parses BSR from frame payload - this is the actual protocol exchange in action.
// TID (Traffic Identifier, 0-7) maps to Access Category:
// - TID 1,2 → AC_BK (Background)
// - TID 0,3 → AC_BE (Best Effort)
// - TID 4,5 → AC_VI (Video)
// - TID 6,7 → AC_VO (Voice)
// BSR reports how much data is buffered for each traffic class per STA.
void
BsrReceivedCallback(Mac48Address staAddress,
                    uint8_t tid,
                    uint8_t queueSizeUnits,
                    uint32_t queueSizeBytes)
{
    if (Simulator::Now().GetMilliSeconds() >= keepTrackOfMetricsFrom_ms)
    {
        // Log to trace file
        if (bsrTraceFile.is_open())
        {
            bsrTraceFile << Simulator::Now().GetMilliSeconds() << "," << staAddress << "," << +tid
                         << "," << queueSizeBytes << "," << +queueSizeUnits << std::endl;
        }

        // Update per-STA per-AC BSR arrays
        std::ostringstream macOss;
        macOss << staAddress;
        std::string macStr = macOss.str();
        auto it = macToStaIdMap.find(macStr);
        if (it != macToStaIdMap.end())
        {
            uint32_t staId = it->second;
            // Map TID to Access Category and update corresponding array
            // 802.11e TID to AC mapping (IEEE 802.11-2020 Table 10-1)
            switch (tid)
            {
            case 1:
            case 2:
                // AC_BK (Background)
                bsrQueueBytesAcBkForSta[staId] = queueSizeBytes;
                break;
            case 0:
            case 3:
                // AC_BE (Best Effort)
                bsrQueueBytesAcBeForSta[staId] = queueSizeBytes;
                break;
            case 4:
            case 5:
                // AC_VI (Video)
                bsrQueueBytesAcViForSta[staId] = queueSizeBytes;
                break;
            case 6:
            case 7:
                // AC_VO (Voice)
                bsrQueueBytesAcVoForSta[staId] = queueSizeBytes;
                break;
            default:
                // Unknown TID, treat as Best Effort
                bsrQueueBytesAcBeForSta[staId] = queueSizeBytes;
                break;
            }
        }
    }
}

// Register MAC address to STA ID mapping for BSR callback
void
RegisterStaMacAddress(Mac48Address macAddr, uint32_t staId)
{
    std::ostringstream oss;
    oss << macAddr;
    macToStaIdMap[oss.str()] = staId;
}

// Trace file management
void
OpenTraceFiles(std::string simIdString)
{
    std::string e2eTraceFileName = TwtResultsPath("data-log/ns3-e2e-trace-") + simIdString + ".csv";
    e2eTraceFile.open(e2eTraceFileName);
    e2eTraceFile << "Time_ms,Event,Node,PacketUID,PacketSize_bytes,TxPower_W" << std::endl;

    std::string macQueueSizeTraceFileName =
        TwtResultsPath("data-log/ns3-macqueuesize-trace-") + simIdString + ".csv";
    macQueueSizeTraceFile.open(macQueueSizeTraceFileName);
    macQueueSizeTraceFile << "Time_ms,Node,QueueSize_packets" << std::endl;

    std::string ampduTraceFileName =
        TwtResultsPath("data-log/ns3-ampdu-trace-") + simIdString + ".csv";
    ampduTraceFile.open(ampduTraceFileName);
    ampduTraceFile << "Time_ms,Node,NumMPDUs,TotalBytes,TxPower_W" << std::endl;

    std::string bsrTraceFileName = TwtResultsPath("data-log/ns3-bsr-trace-") + simIdString + ".csv";
    bsrTraceFile.open(bsrTraceFileName);
    if (bsrTraceFile.is_open())
    {
        // BSR uses MAC address instead of STA ID because it's fired at MAC layer where only the MAC address is available (no context string to extract node ID).
        bsrTraceFile << "Time_ms,StaMacAddress,TID,QueueSizeBytes,QueueSizeUnits" << std::endl;
    }

    std::string phyStateTraceFileName =
        TwtResultsPath("data-log/ns3-phystate-trace-") + simIdString + ".csv";
    phyStateTraceFile.open(phyStateTraceFileName);
    if (phyStateTraceFile.is_open())
    {
        phyStateTraceFile << "Time_ms,Node,State,Duration_ms" << std::endl;
    }

    // AP downlink TX timing — every PSDU the AP transmits, tagged by Kind so the PDW/beacon boundary can be checked (PDW power frames are group-addressed and large; beacons are group + small; UCAST = unicast DATA).
    // Used by the compliance analyzer to confirm PDW broadcasts stay within [0, D] of each BI.
    std::string apTxTraceFileName =
        TwtResultsPath("data-log/ns3-aptx-trace-") + simIdString + ".csv";
    apTxTraceFile.open(apTxTraceFileName);
    if (apTxTraceFile.is_open())
    {
        apTxTraceFile << "Time_ms,Kind,Bytes,DurUs" << std::endl;
    }

    // TX/RX Statistics trace file (802.11k STA Statistics)
    std::string txRxStatsTraceFileName =
        TwtResultsPath("data-log/ns3-txrx-stats-trace-") + simIdString + ".csv";
    txRxStatsTraceFile.open(txRxStatsTraceFileName);
    if (txRxStatsTraceFile.is_open())
    {
        txRxStatsTraceFile << "Time_ms,Node,Event,Count" << std::endl;
    }

    // Link Measurement trace file (RSSI, SNR, Noise)
    std::string linkMeasurementTraceFileName =
        TwtResultsPath("data-log/ns3-link-measurement-trace-") + simIdString + ".csv";
    linkMeasurementTraceFile.open(linkMeasurementTraceFileName);
    if (linkMeasurementTraceFile.is_open())
    {
        linkMeasurementTraceFile << "Time_ms,Node,RSSI_dBm,SNR_dB,Noise_dBm,MCS,NSS,Rate_100kbps"
                                 << std::endl;
    }

    // Timeout and Drop trace file
    std::string timeoutDropTraceFileName =
        TwtResultsPath("data-log/ns3-timeout-drop-trace-") + simIdString + ".csv";
    timeoutDropTraceFile.open(timeoutDropTraceFileName);
    if (timeoutDropTraceFile.is_open())
    {
        timeoutDropTraceFile << "Time_ms,Node,Event,Reason" << std::endl;
    }
}

void
CloseTraceFiles()
{
    if (e2eTraceFile.is_open())
    {
        e2eTraceFile.close();
    }
    if (macQueueSizeTraceFile.is_open())
    {
        macQueueSizeTraceFile.close();
    }
    if (ampduTraceFile.is_open())
    {
        ampduTraceFile.close();
    }
    if (bsrTraceFile.is_open())
    {
        bsrTraceFile.close();
    }
    if (phyStateTraceFile.is_open())
    {
        phyStateTraceFile.close();
    }
    if (apTxTraceFile.is_open())
    {
        apTxTraceFile.close();
    }
    if (txRxStatsTraceFile.is_open())
    {
        txRxStatsTraceFile.close();
    }
    if (linkMeasurementTraceFile.is_open())
    {
        linkMeasurementTraceFile.close();
    }
    if (timeoutDropTraceFile.is_open())
    {
        timeoutDropTraceFile.close();
    }
}

// TRACE CONNECTION FUNCTIONS
// Connects trigger-based trace callback functions to their trigger sources (NS-3 events).
// Each trace callback function defined above needs to be "hooked" to the actual trigger that fires it.
// NS-3 provides these triggers as callback hooks.
// Config::Connect() does this registration: it links a callback function to a trace source so that when the trigger fires, the callback gets invoked automatically with event data.

// Connect summary traces
// Updated to support heterogeneous traffic with multiple application types
void
ConnectSummaryTraces(NodeContainer staNodes)
{
    for (std::size_t ii = 0; ii < staNodes.GetN(); ii++)
    {
        std::stringstream nodeIndexStringTemp, macTxStr, phyTxStr;
        nodeIndexStringTemp << staNodes.Get(ii)->GetId();

        // App-layer TX (offered demand). A STA runs exactly one of these two app types, so each Connect legitimately misses on some nodes — hence the FailSafe variants, which return false instead of aborting.
        //
        // BUGFIX: this used to guard with Config::LookupMatches(<path>/Tx).
        // LookupMatches resolves an OBJECT path, but "Tx" is a TraceSource, not an object child — so the lookup matched NOTHING on every node and Config::Connect was never reached.
        // The result was packets_generated / bytes_generated silently pinned at 0 for the whole run (they are the only policy-INVARIANT measure of offered demand, so this also silently forced served-ratio metrics onto the policy-dependent MAC-admission counter).
        // ConnectFailSafe takes the full trace path and does the match itself.
        std::stringstream onoffTxStr;
        onoffTxStr << "/NodeList/" << nodeIndexStringTemp.str()
                   << "/ApplicationList/*/$ns3::OnOffApplication/Tx";
        Config::ConnectFailSafe(onoffTxStr.str(), MakeCallback(&TxTraceAtApp));

        std::stringstream udpTxStr;
        udpTxStr << "/NodeList/" << nodeIndexStringTemp.str()
                 << "/ApplicationList/*/$ns3::UdpClient/Tx";
        Config::ConnectFailSafe(udpTxStr.str(), MakeCallback(&TxTraceAtApp));

        macTxStr << "/NodeList/" << nodeIndexStringTemp.str()
                 << "/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx";
        Config::Connect(macTxStr.str(), MakeCallback(&MacTxTrace));

        phyTxStr << "/NodeList/" << nodeIndexStringTemp.str() << "/DeviceList/*/Phy/PhyTxBegin";
        Config::Connect(phyTxStr.str(), MakeCallback(&PhyTxBeginTrace));
    }
}

// Connect E2E traces
void
ConnectE2ETraces(NodeContainer staNodes,
                 NodeContainer apNodes,
                 Ptr<Node> serverNode,
                 ApplicationContainer serverApps)
{
    // E2E: Application Tx at STA - support heterogeneous application types
    for (std::size_t ii = 0; ii < staNodes.GetN(); ii++)
    {
        std::stringstream nodeIndexStringTemp;
        nodeIndexStringTemp << staNodes.Get(ii)->GetId();

        // Try OnOffApplication (Voice Assistant, Video Streaming)
        std::stringstream onoffTxStr;
        onoffTxStr << "/NodeList/" << nodeIndexStringTemp.str()
                   << "/ApplicationList/*/$ns3::OnOffApplication/Tx";
        Config::MatchContainer onoffMatches = Config::LookupMatches(onoffTxStr.str());
        if (onoffMatches.GetN() > 0)
        {
            Config::Connect(onoffTxStr.str(), MakeCallback(&E2E_AppTx));
        }

        // Try UdpClient (IoT Sensors, Video Cameras)
        std::stringstream udpTxStr;
        udpTxStr << "/NodeList/" << nodeIndexStringTemp.str()
                 << "/ApplicationList/*/$ns3::UdpClient/Tx";
        Config::MatchContainer udpMatches = Config::LookupMatches(udpTxStr.str());
        if (udpMatches.GetN() > 0)
        {
            Config::Connect(udpTxStr.str(), MakeCallback(&E2E_AppTx));
        }
    }

    // E2E: IP Tx at STA
    for (std::size_t ii = 0; ii < staNodes.GetN(); ii++)
    {
        std::stringstream nodeIndexStringTemp, e2eIpTxStr;
        nodeIndexStringTemp << staNodes.Get(ii)->GetId();
        e2eIpTxStr << "/NodeList/" << nodeIndexStringTemp.str() << "/$ns3::Ipv4L3Protocol/Tx";
        Config::Connect(e2eIpTxStr.str(), MakeCallback(&E2E_IpTx));
    }

    // E2E: MAC Tx at STA
    for (std::size_t ii = 0; ii < staNodes.GetN(); ii++)
    {
        std::stringstream nodeIndexStringTemp, e2eMacTxStr;
        nodeIndexStringTemp << staNodes.Get(ii)->GetId();
        e2eMacTxStr << "/NodeList/" << nodeIndexStringTemp.str()
                    << "/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx";
        Config::Connect(e2eMacTxStr.str(), MakeCallback(&E2E_MacEnqueue));
    }

    // E2E: PHY Tx at STA
    for (std::size_t ii = 0; ii < staNodes.GetN(); ii++)
    {
        std::stringstream nodeIndexStringTemp, e2ePhyTxStr;
        nodeIndexStringTemp << staNodes.Get(ii)->GetId();
        e2ePhyTxStr << "/NodeList/" << nodeIndexStringTemp.str() << "/DeviceList/*/Phy/PhyTxBegin";
        Config::Connect(e2ePhyTxStr.str(), MakeCallback(&E2E_PhyTx));
    }

    // E2E: PHY Rx at AP
    std::stringstream apNodeStr, e2ePhyRxApStr;
    apNodeStr << apNodes.Get(0)->GetId();
    e2ePhyRxApStr << "/NodeList/" << apNodeStr.str() << "/DeviceList/*/Phy/PhyRxEnd";
    Config::Connect(e2ePhyRxApStr.str(), MakeCallback(&E2E_PhyRxAp));

    // E2E: IP Rx at Server
    std::stringstream serverNodeStr, e2eIpRxStr;
    serverNodeStr << serverNode->GetId();
    e2eIpRxStr << "/NodeList/" << serverNodeStr.str() << "/$ns3::Ipv4L3Protocol/Rx";
    Config::Connect(e2eIpRxStr.str(), MakeCallback(&E2E_IpRx));

    // E2E: Application Rx at Server
    for (uint32_t i = 0; i < serverApps.GetN(); i++)
    {
        Ptr<PacketSink> sink = DynamicCast<PacketSink>(serverApps.Get(i));
        if (sink)
        {
            sink->TraceConnectWithoutContext("Rx", MakeCallback(&E2E_AppRx));
        }
    }
}

// Connect QoS metric traces
void
ConnectQosMetricTraces(NodeContainer staNodes)
{
    // Queue Size tracing
    for (std::size_t ii = 0; ii < staNodes.GetN(); ii++)
    {
        std::stringstream nodeIndexStringTemp, queueSizeStr;
        nodeIndexStringTemp << staNodes.Get(ii)->GetId();
        queueSizeStr
            << "/NodeList/" << nodeIndexStringTemp.str()
            << "/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::WifiMac/BE_Txop/Queue/PacketsInQueue";
        Config::Connect(queueSizeStr.str(), MakeCallback(&QueueSizeTrace));
    }

    // A-MPDU aggregation tracing
    for (std::size_t ii = 0; ii < staNodes.GetN(); ii++)
    {
        std::stringstream nodeIndexStringTemp, ampduStr;
        nodeIndexStringTemp << staNodes.Get(ii)->GetId();
        ampduStr << "/NodeList/" << nodeIndexStringTemp.str() << "/DeviceList/*/Phy/PhyTxPsduBegin";
        Config::Connect(ampduStr.str(), MakeCallback(&AmpduAggregationTrace));
    }
}

// Connect PHY state traces
void
ConnectPhyStateTraces(NodeContainer staNodes)
{
    for (std::size_t ii = 0; ii < staNodes.GetN(); ii++)
    {
        std::stringstream nodeIndexStringTemp, phyStateStr;
        nodeIndexStringTemp << staNodes.Get(ii)->GetId();
        phyStateStr << "/NodeList/" << nodeIndexStringTemp.str() << "/DeviceList/*/Phy/State/State";
        Config::Connect(phyStateStr.str(), MakeCallback(&PhyStateTrace_inPlace));
    }
}

// Connect timeout and drop traces
void
ConnectTimeoutAndDropTraces(NodeContainer staNodes, NodeContainer apNodes)
{
    // PSDU timeout traces for STAs
    for (uint32_t i = 0; i < staNodes.GetN(); i++)
    {
        std::stringstream nodeIndexStringTemp, macStr;
        nodeIndexStringTemp << staNodes.Get(i)->GetId();
        macStr << "/NodeList/" << nodeIndexStringTemp.str()
               << "/DeviceList/*/$ns3::WifiNetDevice/Mac/PsduResponseTimeout";
        Config::Connect(macStr.str(), MakeCallback(&PsduResponseTimeoutTraceSta));
    }

    // PSDU timeout trace for AP
    std::stringstream nodeIndexStringTemp, macStr;
    nodeIndexStringTemp << apNodes.Get(0)->GetId();
    macStr << "/NodeList/" << nodeIndexStringTemp.str()
           << "/DeviceList/*/$ns3::WifiNetDevice/Mac/PsduResponseTimeout";
    Config::Connect(macStr.str(), MakeCallback(&PsduResponseTimeoutTraceAp));

    // MPDU drop traces for STAs
    for (uint32_t i = 0; i < staNodes.GetN(); i++)
    {
        std::stringstream nodeIndexStringTemp2, macStr2;
        nodeIndexStringTemp2 << staNodes.Get(i)->GetId();
        macStr2 << "/NodeList/" << nodeIndexStringTemp2.str()
                << "/DeviceList/*/$ns3::WifiNetDevice/Mac/DroppedMpdu";
        Config::Connect(macStr2.str(), MakeCallback(&MpduDropped_atSta));
    }

    // MPDU drop trace for AP
    std::stringstream nodeIndexStringTemp3, macStr3;
    nodeIndexStringTemp3 << apNodes.Get(0)->GetId();
    macStr3 << "/NodeList/" << nodeIndexStringTemp3.str()
            << "/DeviceList/*/$ns3::WifiNetDevice/Mac/DroppedMpdu";
    Config::Connect(macStr3.str(), MakeCallback(&MpduDropped_atAp));
}

// Connect 802.11k traces for RSSI/SNR, TX success/failure, RX statistics
void
Connect802dot11kTraces(NodeContainer staNodes, NodeContainer apNodes)
{
    // --- STA-side traces (ORACLE metrics) ---

    for (std::size_t ii = 0; ii < staNodes.GetN(); ii++)
    {
        std::stringstream nodeIndexStringTemp;
        nodeIndexStringTemp << staNodes.Get(ii)->GetId();

        // MAC TX OK trace - successful transmissions (ACK received) - ORACLE
        std::stringstream macTxOkStr;
        macTxOkStr << "/NodeList/" << nodeIndexStringTemp.str()
                   << "/DeviceList/*/$ns3::WifiNetDevice/Mac/AckedMpdu";
        Config::Connect(macTxOkStr.str(), MakeCallback(&MacTxOkTrace));

        // MAC TX Drop trace - transmission failures - ORACLE
        std::stringstream macTxDropStr;
        macTxDropStr << "/NodeList/" << nodeIndexStringTemp.str()
                     << "/DeviceList/*/$ns3::WifiNetDevice/Mac/DroppedMpdu";
        Config::Connect(macTxDropStr.str(), MakeCallback(&MacTxDropTrace));
    }

    // --- AP-side traces (REALISTIC - AP observes uplink from STAs) ---

    if (apNodes.GetN() > 0)
    {
        std::stringstream apNodeIndexStr;
        apNodeIndexStr << apNodes.Get(0)->GetId();

        // AP Monitor Sniffer RX trace - AP receives frames from STAs.
        // Uses MonitorSnifferRx which provides: Packet, WifiTxVector, SignalNoiseDbm.
        // This populates: rx_fragment_count, RSSI, SNR, MCS, frame type, per-AC traffic.
        std::stringstream apMonitorRxStr;
        apMonitorRxStr << "/NodeList/" << apNodeIndexStr.str()
                       << "/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx";
        Config::Connect(apMonitorRxStr.str(), MakeCallback(&ApMonitorSnifferRxTrace));

        // AP PHY RX Drop trace - AP failed to receive frame.
        // Uses PhyRxDrop which provides: Packet, WifiPhyRxfailureReason.
        // This populates: fcs_error_count.
        std::stringstream apPhyRxDropStr;
        apPhyRxDropStr << "/NodeList/" << apNodeIndexStr.str()
                       << "/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxDrop";
        Config::Connect(apPhyRxDropStr.str(), MakeCallback(&ApPhyRxDropTraceSimple));

        // AP DL TX trace - AP transmits to STAs (QoEH-realistic DL accounting).
        // Populates dl*ToStaForSta (per-STA unicast data) + g_dlBroadcast* (globals).
        std::stringstream apDlTxStr;
        apDlTxStr << "/NodeList/" << apNodeIndexStr.str()
                  << "/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxPsduBegin";
        Config::Connect(apDlTxStr.str(), MakeCallback(&ApPhyTxPsduBeginTrace));
    }
}

} // namespace ns3