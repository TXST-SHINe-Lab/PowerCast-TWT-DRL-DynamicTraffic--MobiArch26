// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

#ifndef TWT_TRACE_CALLBACKS_H
#define TWT_TRACE_CALLBACKS_H

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <fstream>
#include <functional>

namespace ns3
{

// Verbosity / logging globals — defined in twt-trace-callbacks.cc, set by twt-powercast-main-simulation.cc from CLI flags --quietMode and --disableTraces.
extern bool g_quietMode;
extern bool g_disableTraces;

// PDW beacon anchor: invoked once per AP beacon transmission (from ApPhyTxPsduBeginTrace).
// TwtNetworkSetup registers a callback here so the power-delivery burst is anchored to the REAL beacon TX (not a computed BI grid, which is offset from the AP's beacon TBTT).
// nullptr = no-op.
extern std::function<void()> g_onApBeaconTx;

// Global trace files
extern std::ofstream e2eTraceFile;
extern std::ofstream macQueueSizeTraceFile;
extern std::ofstream ampduTraceFile;
extern std::ofstream bsrTraceFile;
extern std::ofstream phyStateTraceFile;
extern std::ofstream txRxStatsTraceFile;       // TX/RX success/fail/retry counters
extern std::ofstream linkMeasurementTraceFile; // RSSI, SNR, noise measurements
extern std::ofstream timeoutDropTraceFile;     // PSDU timeouts and MPDU drops
extern std::ofstream apTxTraceFile; // AP downlink TX timing (PDW/beacon boundary compliance)

// Tracking arrays
// This time elapsed is from STA perspective, if STA is off it's time will freeze
extern double* timeElapsedForSta_ms_TI;
extern double* awakeTimeElapsedForSta_ms_TI;
extern double* sleepTimeElapsedForSta_ms_TI;
extern double* current_mA_TimesTime_ms_ForSta_TI;
extern double* uplinkTimeoutsForSta;
extern double downlinkTimeoutsAllSta; // not used -- DL
extern double* uplinkExpiredMpduForSta;
extern double downlinkExpiredMpduAllSta; // not used --DL
extern double* uplinkFailedEnqueueMpduForSta;
extern double downlinkFailedEnqueueMpduAllSta; // not used -- DL
extern uint64_t* packetsGeneratedByAppForSta;
extern uint64_t*
    packetsEnqueuedAtMacForSta; // incremented in MacTxTrace; exposed via oracle.packets_enqueued
                                // (key v4 reward input as pkts_eq denominator)
extern uint64_t* packetsTransmittedByPhyForSta;
extern uint64_t* bytesTransmittedByPhyForSta;
// MAC queue-sojourn latency (enqueue -> successful ACK), per STA.
// ORACLE (STA-internal queue delay; AP can't directly see it).
// Sum(ms)/count -> oracle.avg_latency_ms for the v3 latency reward.
extern double* latencySumMsForSta;
extern uint64_t* latencyCountForSta;

// --- 802.11k STA STATISTICS - Raw cumulative counters per STA ---
extern uint64_t* txSuccessCountForSta; // Successful TX (got ACK)
extern uint64_t* txRetryCountForSta;   // TX retries (cumulative)
extern uint64_t* txFailedCountForSta;  // TX failures (max retries exceeded)
extern uint64_t* rxSuccessCountForSta; // Successfully received frames
extern uint64_t* rxErrorCountForSta;   // RX errors (FCS failures, decoding errors)
// Timestamp (µs) of the most recent successful UL frame the AP RX'd from each STA.
// Updated at the AP RX trace sites alongside rxSuccessCountForSta.
// Distinct from StaRealisticMetrics::observation_time_ms (which is the time of the observation, not the last frame).
// Used by the obs builder to compute silence-since-last-rx.
extern uint64_t* lastRxTimestampUsForSta;

// --- 802.11k LINK MEASUREMENT - Last known values per STA ---
extern double* lastRssiDbmForSta;         // Last RSSI measurement (dBm)
extern double* lastSnrDbForSta;           // Last SNR measurement (dB)
extern double* lastNoiseDbmForSta;        // Last noise floor (dBm)
extern uint8_t* lastTxMcsForSta;          // Last TX MCS index
extern uint8_t* lastTxNssForSta;          // Last TX NSS (spatial streams)
extern uint16_t* lastTxRate100KbpsForSta; // Last TX rate in 100 kbps units

// --- 802.11ax BSR (Buffer Status Report) - Per Access Category per STA ---
// Updated by BsrReceivedCallback when AP receives BSR from STAs.
// TID to AC mapping: TID 0,3→BE, TID 1,2→BK, TID 4,5→VI, TID 6,7→VO.
extern uint32_t* bsrQueueBytesAcBeForSta; // Best Effort queue (TID 0,3)
extern uint32_t* bsrQueueBytesAcBkForSta; // Background queue (TID 1,2)
extern uint32_t* bsrQueueBytesAcViForSta; // Video queue (TID 4,5)
extern uint32_t* bsrQueueBytesAcVoForSta; // Voice queue (TID 6,7)

// --- AP-RECEIVED TRAFFIC - Per Access Category per STA (from ApPhyRxEndTrace) ---
// TID extracted from QoS Data frames and mapped to AC.
extern uint64_t* bytesReceivedAcBeForSta;   // Bytes received from STA for AC_BE
extern uint64_t* bytesReceivedAcBkForSta;   // Bytes received from STA for AC_BK
extern uint64_t* bytesReceivedAcViForSta;   // Bytes received from STA for AC_VI
extern uint64_t* bytesReceivedAcVoForSta;   // Bytes received from STA for AC_VO
extern uint64_t* packetsReceivedAcBeForSta; // Packets received from STA for AC_BE
extern uint64_t* packetsReceivedAcBkForSta; // Packets received from STA for AC_BK
extern uint64_t* packetsReceivedAcViForSta; // Packets received from STA for AC_VI
extern uint64_t* packetsReceivedAcVoForSta; // Packets received from STA for AC_VO

// --- QoEH-REALISTIC: per-Service-Period (SP) UL accounting + AP DL accounting ---
// Cumulative; Python computes deltas.
// SP boundaries are detected in PhyStateTrace_inPlace (PHY SLEEP edges); UL bytes are bucketed in ApMonitorSnifferRxTrace; DL is counted in ApPhyTxPsduBeginTrace.
// See the SP_* tunables in twt-constants.h.
// All AP-observable (realistic).
extern uint64_t* bytesRxAtApInSpForSta;   // Cumulative UL data bytes served within SP windows
extern uint64_t* packetsRxAtApInSpForSta; // Cumulative UL data packets served within SP windows
extern uint32_t* spCompletedCountForSta;  // # SPs whose end boundary this STA crossed
extern uint32_t*
    spWithDemandCountForSta; // # SPs that had demand (BSR >= SP_DEMAND_MIN_BYTES) at start
extern uint32_t* spWithStarvationCountForSta; // # SPs flagged starved (demand & under-served)
// Internal per-SP working state (not exposed in the schema):
extern uint64_t* spCurServedBytesForSta; // bytes served in the in-progress SP
extern uint64_t* spCurServedPktsForSta;  // packets served in the in-progress SP
extern uint64_t* spWakeStartUsForSta;    // wake-start TSF (us) of the in-progress SP
extern uint32_t* spDemandBytesForSta;    // BSR demand snapshot at the in-progress SP's start
extern bool* spInitedForSta;             // has the first wake-start been observed for this STA

// AP DL accounting (AP knows what it sent — fully realistic).
extern uint64_t* dlBytesToStaForSta;        // Cumulative DL unicast data bytes AP -> STA i
extern uint64_t* dlDurationUsToStaForSta;   // Cumulative DL unicast airtime (us) AP -> STA i
extern uint32_t* dlUnicastCountToStaForSta; // # DL unicast data frames AP -> STA i
// Global broadcast/multicast DL (diagnostic; not load-bearing for the reward):
extern uint64_t g_dlBroadcastBytes;
extern uint64_t g_dlBroadcastDurationUs;
extern uint32_t g_dlBroadcastCount;

// --- A-MPDU AGGREGATION - Cumulative counters per STA ---
extern uint64_t* ampduCountForSta;      // Number of A-MPDU transmissions
extern uint64_t* ampduMpdusTotalForSta; // Total MPDUs in A-MPDUs
extern uint64_t* ampduBytesTotalForSta; // Total bytes in A-MPDUs

// --- AIRTIME & PHY PARAMETERS - Per STA tracking ---
extern uint64_t* airtimeUsedUsForSta;       // Cumulative airtime in microseconds
extern uint8_t* lastChannelWidthMhzForSta;  // Last channel width used (20/40/80/160)
extern uint16_t* lastGuardIntervalNsForSta; // Last guard interval (800/1600/3200)
extern uint8_t* lastTxPowerDbmForSta;       // Last TX power in dBm
extern uint8_t* lastFrameTypeForSta;        // Last RX frame type (0=mgmt, 1=ctrl, 2=data)
extern uint8_t* lastFrameSubtypeForSta;     // Last RX frame subtype (0-15)
extern uint8_t* lastPowerMgmtBitForSta;     // Last Power Management bit from Frame Control

// Configuration parameters needed by callbacks
extern double keepTrackOfMetricsFrom_ms; // Metrics collection start time in milliseconds
extern std::unordered_map<std::string, double> TI_currentModel_mA; // mA values for phy states

// Helper functions
uint32_t ContextToNodeId(std::string context);
double GetPercentileValue(Histogram hist, double percentile);

// Summary trace callbacks (no time filter)
void TxTraceAtApp(std::string context, Ptr<const Packet> packet);
void MacTxTrace(std::string context, Ptr<const Packet> packet);
// Here TX power is in W because of ns3 source code
void PhyTxBeginTrace(std::string context, Ptr<const Packet> packet, double txPowerW);

// End-to-end detailed trace callbacks (time filtered)
void E2E_AppTx(std::string context, Ptr<const Packet> packet);
void E2E_IpTx(std::string context, Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);
void E2E_MacEnqueue(std::string context, Ptr<const Packet> packet);
// Here TX power is in W because of ns3 source code
void E2E_PhyTx(std::string context, Ptr<const Packet> packet, double txPowerW);
void E2E_PhyRxAp(std::string context, Ptr<const Packet> packet);
void E2E_IpRx(std::string context, Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);
void E2E_AppRx(Ptr<const Packet> packet, const Address& address);

// QoS metric callbacks
void QueueSizeTrace(std::string context, uint32_t oldSize, uint32_t newSize);
void AmpduAggregationTrace(std::string context,
                           WifiConstPsduMap psduMap,
                           WifiTxVector txVector,
                           // Here TX power is in W because of ns3 source code
                           double txPowerW);
// AP-side DL TX accounting (REALISTIC — AP knows what it transmits).
// Connected to the AP's PhyTxPsduBegin; per-STA unicast data into dl*ToStaForSta, group-addressed frames into the global g_dlBroadcast* counters.
void ApPhyTxPsduBeginTrace(std::string context,
                           WifiConstPsduMap psduMap,
                           WifiTxVector txVector,
                           double txPowerW);

// PHY state and error callbacks
void PhyStateTrace_inPlace(std::string context, Time start, Time duration, WifiPhyState state);
void PsduResponseTimeoutTraceSta(std::string context,
                                 uint8_t reason,
                                 Ptr<const WifiPsdu> psdu,
                                 const WifiTxVector& txVector);
void PsduResponseTimeoutTraceAp(std::string context,
                                uint8_t reason,
                                Ptr<const WifiPsdu> psdu,
                                const WifiTxVector& txVector);
void MpduDropped_atSta(std::string context, WifiMacDropReason dropReason, Ptr<const WifiMpdu> mpdu);
void MpduDropped_atAp(std::string context, WifiMacDropReason dropReason, Ptr<const WifiMpdu> mpdu);

// 802.11k Link Measurement callbacks (STA-side, for oracle/downlink)
void PhyRxEndTrace(std::string context,
                   Ptr<const WifiPsdu> psdu,
                   RxSignalInfo rxSignalInfo,
                   const WifiTxVector& txVector,
                   const std::vector<bool>& perMpduStatus);
void PhyRxDropTrace(std::string context, Ptr<const WifiPsdu> psdu, WifiPhyRxfailureReason reason);

// AP-side RX callbacks (REALISTIC - AP receives from STAs)
void ApPhyRxEndTrace(std::string context,
                     Ptr<const WifiPsdu> psdu,
                     RxSignalInfo rxSignalInfo,
                     const WifiTxVector& txVector,
                     const std::vector<bool>& perMpduStatus);
void ApPhyRxDropTrace(std::string context, Ptr<const WifiPsdu> psdu, WifiPhyRxfailureReason reason);

// NEW: AP Monitor Sniffer RX callback - matches MonitorSnifferRx signature in NS-3.44
void ApMonitorSnifferRxTrace(std::string context,
                             Ptr<const Packet> packet,
                             uint16_t channelFreqMhz,
                             WifiTxVector txVector,
                             MpduInfo aMpdu,
                             SignalNoiseDbm signalNoise,
                             uint16_t staId);

// NEW: AP PHY RX Drop callback (simplified) - matches PhyRxDrop signature in NS-3.44
void ApPhyRxDropTraceSimple(std::string context,
                            Ptr<const Packet> packet,
                            WifiPhyRxfailureReason reason);

// 802.11k TX statistics callbacks - track successful TX and retries
void MacTxOkTrace(std::string context, Ptr<const WifiMpdu> mpdu);
void MacTxDropTrace(std::string context, WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu);

// BSR callbacks
void BsrReceivedCallback(Mac48Address staAddress,
                         uint8_t tid,
                         uint8_t queueSizeUnits,
                         uint32_t queueSizeBytes);
void PeriodicBsrCheck();

// Register MAC address to STA ID mapping for BSR callback
void RegisterStaMacAddress(Mac48Address macAddr, uint32_t staId);

// Trace file management
void OpenTraceFiles(std::string simIdString);
void CloseTraceFiles();

// Initialize 802.11k tracking arrays (call after InitializeArrays)
void Initialize802dot11kArrays(uint32_t numSta);

// Connect all traces
void ConnectSummaryTraces(NodeContainer staNodes);
void ConnectE2ETraces(NodeContainer staNodes,
                      NodeContainer apNodes,
                      Ptr<Node> serverNode,
                      ApplicationContainer serverApps);
void ConnectQosMetricTraces(NodeContainer staNodes);
void ConnectPhyStateTraces(NodeContainer staNodes);
void ConnectTimeoutAndDropTraces(NodeContainer staNodes, NodeContainer apNodes);
void Connect802dot11kTraces(NodeContainer staNodes,
                            NodeContainer apNodes); // Connect 802.11k traces (STA + AP)

} // namespace ns3

#endif // TWT_TRACE_CALLBACKS_H