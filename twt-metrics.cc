// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

#include "twt-metrics.h"

#include "pb-twt-core.h"
#include "twt-constants.h"
#include "twt-trace-callbacks.h"

#include "ns3/mobility-model.h"

#include <climits>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ns3
{

TwtMetrics::TwtMetrics(const TwtSimulationConfig& config, NodeContainer staNodes)
      : m_config(config)
      , m_staNodes(staNodes)
      , m_networkSetup(nullptr)
      , m_biLevelLoggingEnabled(false)
      , m_biLevelObservationCount(0)
      , m_lastBiLogTime_ms(0.0)
      , m_biLevelFirstCall(true)
      , m_callLevelLoggingEnabled(false)
      , m_callLevelObservationCount(0)
      , m_lastCallLogTime_ms(0.0)
      , m_callLevelFirstCall(true)
      , m_fallbackLoggingEnabled(false)
{
    m_totalRxBytes = nullptr;
    m_throughput = nullptr;

    // Initialize BSR window accumulators
    std::size_t nStations = staNodes.GetN();
    m_bsrOccupancySum.resize(nStations, 0.0);
    m_bsrSampleCount.resize(nStations, 0);
    m_bsrAbove50Count.resize(nStations, 0);
    m_bsrAbove75Count.resize(nStations, 0);
    m_bsrAbove95Count.resize(nStations, 0);
}

TwtMetrics::~TwtMetrics()
{
    if (m_totalRxBytes)
    {
        delete[] m_totalRxBytes;
    }
    if (m_throughput)
    {
        delete[] m_throughput;
    }
}

void
TwtMetrics::InitializeArrays()
{
    std::size_t nStations = m_staNodes.GetN();

    // Initialize tracking arrays in trace callbacks module (all time values in ms)
    timeElapsedForSta_ms_TI = new double[nStations];
    awakeTimeElapsedForSta_ms_TI = new double[nStations];
    sleepTimeElapsedForSta_ms_TI = new double[nStations];
    current_mA_TimesTime_ms_ForSta_TI = new double[nStations];
    uplinkTimeoutsForSta = new double[nStations];
    uplinkExpiredMpduForSta = new double[nStations];
    uplinkFailedEnqueueMpduForSta = new double[nStations];
    packetsGeneratedByAppForSta = new uint64_t[nStations];
    packetsEnqueuedAtMacForSta = new uint64_t[nStations];
    packetsTransmittedByPhyForSta = new uint64_t[nStations];
    bytesTransmittedByPhyForSta = new uint64_t[nStations];
    latencySumMsForSta = new double[nStations];
    latencyCountForSta = new uint64_t[nStations];

    for (std::size_t i = 0; i < nStations; i++)
    {
        timeElapsedForSta_ms_TI[i] = 0;
        awakeTimeElapsedForSta_ms_TI[i] = 0;
        sleepTimeElapsedForSta_ms_TI[i] = 0;
        current_mA_TimesTime_ms_ForSta_TI[i] = 0;
        uplinkTimeoutsForSta[i] = 0;
        uplinkExpiredMpduForSta[i] = 0;
        uplinkFailedEnqueueMpduForSta[i] = 0;
        packetsGeneratedByAppForSta[i] = 0;
        packetsEnqueuedAtMacForSta[i] = 0;
        packetsTransmittedByPhyForSta[i] = 0;
        bytesTransmittedByPhyForSta[i] = 0;
        latencySumMsForSta[i] = 0;
        latencyCountForSta[i] = 0;
    }

    downlinkTimeoutsAllSta = 0;
    downlinkExpiredMpduAllSta = 0;
    downlinkFailedEnqueueMpduAllSta = 0;

    // Initialize 802.11k tracking arrays
    Initialize802dot11kArrays(nStations);

    // Initialize metrics arrays
    m_totalRxBytes = new uint64_t[nStations];
    m_throughput = new double[nStations];
}

void
TwtMetrics::PopulateStaObservationRaw(StaEnvStruct& sta, uint32_t staId)
{
    if (staId >= m_staNodes.GetN())
    {
        return;
    }

    Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(m_staNodes.Get(staId)->GetDevice(0));
    Ptr<WifiMac> wifi_mac = wifi_dev->GetMac();
    Ptr<StaWifiMac> sta_mac = DynamicCast<StaWifiMac>(wifi_mac);

    // Get references to sub-structs
    StaRealisticMetrics& real = sta.realistic;
    StaOracleMetrics& oracle = sta.oracle;

    // --- REALISTIC: IDENTIFICATION ---

    real.sta_id = staId;
    oracle.sta_id = staId;
    Mac48Address macAddr = sta_mac->GetAddress();
    macAddr.CopyTo(real.sta_mac);
    real.is_active = 1;

    // Register MAC address mapping for BSR callback (idempotent)
    RegisterStaMacAddress(macAddr, staId);

    // --- REALISTIC: 802.11ax BSR - Per Access Category (from trace arrays) ---

    // These are populated by BsrReceivedCallback when AP receives BSR from STAs.
    // Values are in bytes; convert to quantized 0-255 (256 bytes per unit).
    constexpr uint32_t BSR_SCALING_FACTOR = 256;
    real.bsr_queue_ac_be = bsrQueueBytesAcBeForSta
                               ? static_cast<uint8_t>(std::min(
                                     bsrQueueBytesAcBeForSta[staId] / BSR_SCALING_FACTOR, 255u))
                               : 0;
    real.bsr_queue_ac_bk = bsrQueueBytesAcBkForSta
                               ? static_cast<uint8_t>(std::min(
                                     bsrQueueBytesAcBkForSta[staId] / BSR_SCALING_FACTOR, 255u))
                               : 0;
    real.bsr_queue_ac_vi = bsrQueueBytesAcViForSta
                               ? static_cast<uint8_t>(std::min(
                                     bsrQueueBytesAcViForSta[staId] / BSR_SCALING_FACTOR, 255u))
                               : 0;
    real.bsr_queue_ac_vo = bsrQueueBytesAcVoForSta
                               ? static_cast<uint8_t>(std::min(
                                     bsrQueueBytesAcVoForSta[staId] / BSR_SCALING_FACTOR, 255u))
                               : 0;
    real.bsr_scaling_factor = BSR_SCALING_FACTOR;

    // --- REALISTIC: AP-OBSERVABLE RX COUNTERS ---

    // Only RX counters are truly realistic - AP directly observes these.
    // TX counters moved to oracle (would need 802.11k request in real deployment).
    real.rx_fragment_count = rxSuccessCountForSta ? rxSuccessCountForSta[staId] : 0;
    if (!rxSuccessCountForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "rx_fragment_count");
    }
    real.fcs_error_count = rxErrorCountForSta ? rxErrorCountForSta[staId] : 0;
    if (!rxErrorCountForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "fcs_error_count");
    }

    // --- REALISTIC: 802.11k LINK MEASUREMENT - Raw + converted ---

    // Use INT8_MIN (-128) as sentinel for "no valid measurement" since struct uses int8_t.
    // The trace arrays use NAN to indicate no measurement received yet.
    int8_t rssiDbm = INT8_MIN;
    int8_t snrDb = INT8_MIN;
    if (lastRssiDbmForSta && !std::isnan(lastRssiDbmForSta[staId]))
    {
        rssiDbm = static_cast<int8_t>(lastRssiDbmForSta[staId]);
    }
    if (lastSnrDbForSta && !std::isnan(lastSnrDbForSta[staId]))
    {
        snrDb = static_cast<int8_t>(lastSnrDbForSta[staId]);
    }
    // RCPI = (rssi_dbm + 110) * 2 (0.5 dBm units, range 0-220)
    // Only compute if valid, otherwise use INT8_MIN sentinel
    real.rcpi = (rssiDbm != INT8_MIN) ? static_cast<int8_t>((rssiDbm + 110) * 2) : INT8_MIN;
    // RSNI = snr_db * 2 (0.5 dB units)
    real.rsni = (snrDb != INT8_MIN) ? static_cast<int8_t>(snrDb * 2) : INT8_MIN;
    real.rssi_dbm = rssiDbm;
    real.snr_db = snrDb;
    real.link_margin_db = snrDb; // Simplified: link margin ≈ SNR (or INT8_MIN if no data)
    // TX power: 0 means no observation yet (valid TX power is typically 1-30 dBm)
    // ORACLE (moved 2026-06-07): STA TX power needs a TPC Report, not modeled here.
    oracle.tx_power_dbm = lastTxPowerDbmForSta ? lastTxPowerDbmForSta[staId] : 0;

    // --- REALISTIC: MAC LAYER OBSERVATIONS (from frame headers in trace) ---

    // 0 means no observation yet for all these fields
    real.last_rx_frame_type = lastFrameTypeForSta ? lastFrameTypeForSta[staId] : 0;
    if (!lastFrameTypeForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "last_rx_frame_type");
    }
    real.last_rx_frame_subtype = lastFrameSubtypeForSta ? lastFrameSubtypeForSta[staId] : 0;
    if (!lastFrameSubtypeForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "last_rx_frame_subtype");
    }
    real.last_rx_mcs = lastTxMcsForSta ? lastTxMcsForSta[staId] : 0;
    if (!lastTxMcsForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "last_rx_mcs");
    }
    real.last_rx_nss = lastTxNssForSta ? lastTxNssForSta[staId] : 0;
    if (!lastTxNssForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "last_rx_nss");
    }
    real.channel_width_mhz = lastChannelWidthMhzForSta ? lastChannelWidthMhzForSta[staId] : 0;
    if (!lastChannelWidthMhzForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "channel_width_mhz");
    }
    real.guard_interval_ns = lastGuardIntervalNsForSta ? lastGuardIntervalNsForSta[staId] : 0;
    if (!lastGuardIntervalNsForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "guard_interval_ns");
    }
    real.power_mgmt_bit = lastPowerMgmtBitForSta ? lastPowerMgmtBitForSta[staId] : 0;
    if (!lastPowerMgmtBitForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "power_mgmt_bit");
    }
    // Source: lastRxTimestampUsForSta (updated by AP RX callbacks).
    // 0 = no UL frame received yet from this STA; the obs builder treats 0 as "fully silent".
    real.last_rx_timestamp_us = lastRxTimestampUsForSta ? lastRxTimestampUsForSta[staId] : 0;

    // --- REALISTIC: AP-DERIVED (from received frames) ---

    // Compute total from per-AC arrays
    uint64_t totalBytesRx = 0;
    uint64_t totalPacketsRx = 0;

    real.bytes_received_ac_be = bytesReceivedAcBeForSta ? bytesReceivedAcBeForSta[staId] : 0;
    real.bytes_received_ac_bk = bytesReceivedAcBkForSta ? bytesReceivedAcBkForSta[staId] : 0;
    real.bytes_received_ac_vi = bytesReceivedAcViForSta ? bytesReceivedAcViForSta[staId] : 0;
    real.bytes_received_ac_vo = bytesReceivedAcVoForSta ? bytesReceivedAcVoForSta[staId] : 0;
    real.packets_received_ac_be = packetsReceivedAcBeForSta ? packetsReceivedAcBeForSta[staId] : 0;
    real.packets_received_ac_bk = packetsReceivedAcBkForSta ? packetsReceivedAcBkForSta[staId] : 0;
    real.packets_received_ac_vi = packetsReceivedAcViForSta ? packetsReceivedAcViForSta[staId] : 0;
    real.packets_received_ac_vo = packetsReceivedAcVoForSta ? packetsReceivedAcVoForSta[staId] : 0;

    totalBytesRx = real.bytes_received_ac_be + real.bytes_received_ac_bk +
                   real.bytes_received_ac_vi + real.bytes_received_ac_vo;
    totalPacketsRx = real.packets_received_ac_be + real.packets_received_ac_bk +
                     real.packets_received_ac_vi + real.packets_received_ac_vo;

    real.bytes_received_at_ap = totalBytesRx;
    real.packets_received_at_ap = totalPacketsRx;

    // --- REALISTIC: Airtime tracking (from trace) ---

    real.airtime_used_us = airtimeUsedUsForSta ? airtimeUsedUsForSta[staId] : 0;
    if (!airtimeUsedUsForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "airtime_used_us");
    }

    // --- ORACLE: DEVICE CHARACTERISTICS (moved 2026-06-07; config stand-in for TSPEC) ---

    if (staId < m_config.sta_app_configs.size())
    {
        const auto& cfg = m_config.sta_app_configs[staId];
        oracle.device_class = static_cast<uint8_t>(cfg.device_class);
        oracle.nominal_msdu_size = cfg.packet_size_bytes;
        oracle.mean_data_rate_kbps = cfg.traffic_rate_bps / 1000.0;
        oracle.delay_bound_ms = cfg.latency_requirement.GetMilliSeconds();
        // UNIFORM AC (2026-06-07): ALL devices use AC_BE — uniform importance, no per-class AC priority.
        // The OnOff/UDP sockets never set a TID, so all UL already defaults to AC_BE (per-AC vi/vo/bk BSR were structurally zero); this just makes the metadata honest.
        // user_priority = 0 (AC_BE UP) for every STA. (Was a per-class UP map that never reached the socket.)
        // Only the bsr_queue_ac_be BSR type is used; the other three AC BSR fields stay collected-but-unused.
        real.user_priority = 0; // AC_BE for every STA
    }
    else
    {
        oracle.device_class = 0;
        oracle.nominal_msdu_size = PAYLOAD_SIZE_BYTES;
        oracle.mean_data_rate_kbps = 0.0;
        oracle.delay_bound_ms = 0.0;
        real.user_priority = 0;
    }

    // --- REALISTIC: OBSERVATION METADATA ---

    real.observation_time_ms = Simulator::Now().GetSeconds() * 1000.0;
    real.observation_sequence_num = m_callLevelObservationCount;

    // --- REALISTIC: QoEH per-SP UL accounting + AP DL accounting ---

    // All AP-observable (SP boundaries from PHY-sleep edges the AP schedules; UL bytes the AP receives within each SP; DL the AP itself sends).
    // Cumulative; Python computes deltas.
    // Populated by the SP/DL trace callbacks; null-guarded.
    real.bytes_rx_at_ap_in_sp = bytesRxAtApInSpForSta ? bytesRxAtApInSpForSta[staId] : 0;
    real.packets_rx_at_ap_in_sp = packetsRxAtApInSpForSta ? packetsRxAtApInSpForSta[staId] : 0;
    real.sp_completed_count = spCompletedCountForSta ? spCompletedCountForSta[staId] : 0;
    real.sp_with_demand_count = spWithDemandCountForSta ? spWithDemandCountForSta[staId] : 0;
    real.sp_with_starvation_count =
        spWithStarvationCountForSta ? spWithStarvationCountForSta[staId] : 0;
    real.dl_bytes_to_sta = dlBytesToStaForSta ? dlBytesToStaForSta[staId] : 0;
    real.dl_duration_us_to_sta = dlDurationUsToStaForSta ? dlDurationUsToStaForSta[staId] : 0;
    real.dl_unicast_count_to_sta = dlUnicastCountToStaForSta ? dlUnicastCountToStaForSta[staId] : 0;

    // --- ORACLE METRICS (Simulation-only, for validation) ---
    // --- ORACLE: STA TX COUNTERS (Would need 802.11k request in real deployment) ---

    oracle.tx_fragment_count = txSuccessCountForSta ? txSuccessCountForSta[staId] : 0;
    if (!txSuccessCountForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "tx_fragment_count");
    }
    oracle.tx_failed_count = txFailedCountForSta ? txFailedCountForSta[staId] : 0;
    if (!txFailedCountForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "tx_failed_count");
    }
    oracle.tx_retry_count = txRetryCountForSta ? txRetryCountForSta[staId] : 0;
    if (!txRetryCountForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "tx_retry_count");
    }
    oracle.ack_failure_count = static_cast<uint64_t>(uplinkTimeoutsForSta[staId]);

    // --- ORACLE: ENERGY (STA-private) ---

    double awakeTime = awakeTimeElapsedForSta_ms_TI[staId];
    double sleepTime = sleepTimeElapsedForSta_ms_TI[staId];
    double totalTime = awakeTime + sleepTime;
    oracle.awake_time_ms = awakeTime;
    oracle.sleep_time_ms = sleepTime;
    oracle.duty_cycle = (totalTime > 0) ? awakeTime / totalTime : 1.0;
    oracle.current_times_time_ma_ms = current_mA_TimesTime_ms_ForSta_TI[staId];
    // Energy in mJ = (mA * ms) * V / 1000 = (mA * ms) * 3.3 / 1000
    oracle.total_energy_consumed_mj = oracle.current_times_time_ma_ms * 3.3 / 1000.0;

    // --- ORACLE: APPLICATION LAYER ---

    oracle.packets_generated = packetsGeneratedByAppForSta[staId];
    oracle.packets_enqueued = packetsEnqueuedAtMacForSta[staId];
    oracle.bytes_generated = packetsGeneratedByAppForSta[staId] * PAYLOAD_SIZE_BYTES;

    // --- ORACLE: STA QUEUE DROPS ---

    oracle.mpdu_drops_expired = static_cast<uint64_t>(uplinkExpiredMpduForSta[staId]);
    oracle.mpdu_drops_queue_full = static_cast<uint64_t>(uplinkFailedEnqueueMpduForSta[staId]);
    oracle.psdu_response_timeouts = static_cast<uint64_t>(uplinkTimeoutsForSta[staId]);

    // --- ORACLE: TX SIDE METRICS ---

    oracle.packets_transmitted = packetsTransmittedByPhyForSta[staId];
    oracle.bytes_transmitted = bytesTransmittedByPhyForSta[staId];
    oracle.ampdu_count = ampduCountForSta ? ampduCountForSta[staId] : 0;
    if (!ampduCountForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "ampdu_count");
    }
    oracle.ampdu_mpdus_total = ampduMpdusTotalForSta ? ampduMpdusTotalForSta[staId] : 0;
    if (!ampduMpdusTotalForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "ampdu_mpdus_total");
    }
    oracle.ampdu_bytes_total = ampduBytesTotalForSta ? ampduBytesTotalForSta[staId] : 0;
    if (!ampduBytesTotalForSta)
    {
        LogFallback(Simulator::Now().GetSeconds() * 1000.0, staId, "ampdu_bytes_total");
    }

    // --- ORACLE: QUEUE STATE ---

    // Compute total queue from per-AC BSR arrays
    uint32_t totalQueueBytes = 0;
    if (bsrQueueBytesAcBeForSta)
    {
        totalQueueBytes += bsrQueueBytesAcBeForSta[staId];
    }
    if (bsrQueueBytesAcBkForSta)
    {
        totalQueueBytes += bsrQueueBytesAcBkForSta[staId];
    }
    if (bsrQueueBytesAcViForSta)
    {
        totalQueueBytes += bsrQueueBytesAcViForSta[staId];
    }
    if (bsrQueueBytesAcVoForSta)
    {
        totalQueueBytes += bsrQueueBytesAcVoForSta[staId];
    }
    oracle.queue_size_packets = totalQueueBytes / PAYLOAD_SIZE_BYTES;
    oracle.queue_size_bytes = totalQueueBytes;
    oracle.queue_max_size = MAX_QUEUE_SIZE_BYTES / PAYLOAD_SIZE_BYTES; // From constants

    // --- ORACLE: LATENCY ---
    // MAC queue-sojourn: enqueue -> successful ACK, populated by MacTxOkTrace via mpdu->GetTimestamp().
    // Cumulative sum/count; Python computes per-step delta.
    oracle.queue_delay_sum_ms = latencySumMsForSta ? latencySumMsForSta[staId] : 0.0;
    oracle.queue_delay_count = latencyCountForSta ? latencyCountForSta[staId] : 0;
    oracle.avg_latency_ms =
        oracle.queue_delay_count > 0 ? oracle.queue_delay_sum_ms / oracle.queue_delay_count : 0.0;

    // --- ORACLE: STA POSITION (from MobilityModel) ---

    Ptr<MobilityModel> mobility = m_staNodes.Get(staId)->GetObject<MobilityModel>();
    if (mobility)
    {
        Vector pos = mobility->GetPosition();
        oracle.position_x_m = pos.x;
        oracle.position_y_m = pos.y;
        oracle.position_z_m = pos.z;
    }
    else
    {
        oracle.position_x_m = 0.0;
        oracle.position_y_m = 0.0;
        oracle.position_z_m = 0.0;
    }

    // --- ORACLE: PowerCast Energy (QoEH inputs) ---

    // Only REHD-class STAs have a harvester installed (filtered inside SetupEnergyHarvesting).
    // Non-REHDs return nullptr → all 11 fields stay 0 by virtue of the std::memset(&sta, 0, …) earlier in the caller.
    if (m_networkSetup)
    {
        Ptr<energy::PowercastEnergyHarvester> h = m_networkSetup->GetHarvester(staId);
        if (h)
        {
            oracle.vcap_v = h->GetVcap();
            oracle.vcap_max_v = h->GetCapMaxVoltage();
            oracle.e_nominal_j = h->GetNominalEnergy();
            oracle.e_sunk_j = h->GetSunkEnergy();
            oracle.e_avail_j = h->GetAvailableEnergy();
            oracle.harvested_total_j = h->GetTotalHarvestedEnergy();
            oracle.consumed_total_j = h->GetTotalConsumedEnergy();
            oracle.t_active_us = h->GetTotalActiveTimeUs();
            oracle.unpowered_tx_events = m_networkSetup->GetUnpoweredTxEvents(staId);
            oracle.output_enabled = h->IsOutputEnabled() ? 1 : 0;
            oracle.harvester_phase = h->ComputePhase();
            // est_rf_energy_delivered_j stays 0 here — populated separately by the AP-side DL-TX trace callback once that work lands.
        }
    }
}

// BI-level logging: Initialize CSV file and logging state
void
TwtMetrics::InitializeBiLevelLogging(std::string logdir)
{
    // Generate timestamp for unique filename
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = oss.str();

    // Create CSV file for BI-level observations
    std::string biLevelFile = logdir + "/ns3-BI-log-" + timestamp + ".csv";
    m_biLevelCsv.open(biLevelFile, std::ios::app);

    if (!m_biLevelCsv.is_open())
    {
        std::cerr << "ERROR: Failed to open BI-level logging file: " << biLevelFile << std::endl;
        return;
    }

    m_biLevelLoggingEnabled = true;
    m_biLevelObservationCount = 0;
    m_lastBiLogTime_ms = Simulator::Now().GetSeconds() * 1000.0;
    m_biLevelFirstCall = true;

    // Write CSV header - Separated into REALISTIC (802.11ax/k) and ORACLE sections
    m_biLevelCsv
        << "bi_index,observation_time_ms,sta_id,"
        // --- REALISTIC: 802.11ax BSR per-AC (quantized 0-255) ---

        << "bsr_ac_be,bsr_ac_bk,bsr_ac_vi,bsr_ac_vo,bsr_scaling_factor,"
        // --- REALISTIC: 802.11k STA Statistics ---

        << "tx_fragment_count,tx_failed_count,tx_retry_count,"
        << "ack_failure_count,rx_fragment_count,fcs_error_count,"
        // --- REALISTIC: 802.11k Link Measurement - Raw + converted ---

        << "rcpi,rsni,rssi_dbm,snr_db,link_margin_db,tx_power_dbm,"
        // --- REALISTIC: MAC Layer Observations ---

        << "last_rx_frame_type,last_rx_frame_subtype,"
        << "last_rx_mcs,last_rx_nss,channel_width_mhz,guard_interval_ns,"
        << "power_mgmt_bit,last_rx_timestamp_us,"
        // --- REALISTIC: AP-derived ---

        << "bytes_received_at_ap,packets_received_at_ap,"
        // --- REALISTIC: Per-AC TX tracking ---

        << "bytes_received_ac_vo,bytes_received_ac_vi,bytes_received_ac_be,bytes_received_ac_bk,"
        << "packets_received_ac_vo,packets_received_ac_vi,packets_received_ac_be,packets_received_"
           "ac_bk,"
        << "airtime_used_us,"
        // --- REALISTIC: Device characteristics ---

        << "device_class,mean_data_rate_kbps,delay_bound_ms,user_priority,"
        << "observation_sequence_num,"
        // --- ORACLE: Energy ---

        << "oracle_total_energy_mj,oracle_awake_time_ms,oracle_sleep_time_ms,oracle_duty_cycle,"
        // --- ORACLE: Application layer ---

        << "oracle_packets_generated,oracle_packets_enqueued,oracle_bytes_generated,"
        // --- ORACLE: Queue drops ---

        << "oracle_mpdu_drops_expired,oracle_mpdu_drops_queue_full,oracle_psdu_timeouts,"
        // --- ORACLE: TX metrics ---

        << "oracle_packets_transmitted,oracle_bytes_transmitted,"
        << "oracle_ampdu_count,oracle_ampdu_mpdus_total,oracle_ampdu_bytes_total,"
        // --- ORACLE: Queue state ---

        << "oracle_queue_size_packets,oracle_queue_size_bytes\n";

    m_biLevelCsv.flush();
    std::cout << "BI-level logging initialized: " << biLevelFile << std::endl;
}

// BI-level logging: Collect and log RAW metrics for every beacon interval
void
TwtMetrics::LogBiLevelMetrics()
{
    if (!m_biLevelLoggingEnabled || !m_biLevelCsv.is_open())
    {
        return;
    }

    double simTime_ms = Simulator::Now().GetSeconds() * 1000.0;

    // On first call, just initialize and return (skip pre-warmup)
    if (m_biLevelFirstCall)
    {
        m_biLevelFirstCall = false;
        m_lastBiLogTime_ms = simTime_ms;
        return;
    }

    uint64_t biIndex = static_cast<uint64_t>(
        std::round(simTime_ms / (m_config.beaconInterval_s.GetSeconds() * 1000.0)));

    // Collect per-STA observations using new nested structure
    for (uint32_t staId = 0; staId < m_staNodes.GetN(); staId++)
    {
        StaEnvStruct sta;
        std::memset(&sta, 0, sizeof(StaEnvStruct));

        // Populate with raw data (fills realistic + oracle sub-structs)
        PopulateStaObservationRaw(sta, staId);

        // References to sub-structs
        const StaRealisticMetrics& real = sta.realistic;
        const StaOracleMetrics& oracle = sta.oracle;

        // Accumulate BSR statistics for call-level window metrics
        uint32_t totalBsr = real.bsr_queue_ac_be + real.bsr_queue_ac_bk + real.bsr_queue_ac_vi +
                            real.bsr_queue_ac_vo;
        double occupancy = static_cast<double>(totalBsr) / 255.0; // Max quantized value
        m_bsrOccupancySum[staId] += occupancy;
        m_bsrSampleCount[staId]++;
        if (occupancy > 0.50)
        {
            m_bsrAbove50Count[staId]++;
        }
        if (occupancy > 0.75)
        {
            m_bsrAbove75Count[staId]++;
        }
        if (occupancy > 0.95)
        {
            m_bsrAbove95Count[staId]++;
        }

        // Write data to CSV - separated REALISTIC and ORACLE sections
        m_biLevelCsv << biIndex << "," << std::fixed << std::setprecision(3)
                     << real.observation_time_ms << ","
                     << real.sta_id
                     // --- REALISTIC: 802.11ax BSR per-AC (quantized) ---

                     << "," << (int)real.bsr_queue_ac_be << "," << (int)real.bsr_queue_ac_bk << ","
                     << (int)real.bsr_queue_ac_vi << "," << (int)real.bsr_queue_ac_vo << ","
                     << (int)real.bsr_scaling_factor
                     // --- REALISTIC: AP-Observable RX Counters ---

                     << "," << real.rx_fragment_count << ","
                     << real.fcs_error_count
                     // --- ORACLE: STA TX Counters (would need 802.11k request) ---

                     << "," << oracle.tx_fragment_count << "," << oracle.tx_failed_count << ","
                     << oracle.tx_retry_count << ","
                     << oracle.ack_failure_count
                     // --- REALISTIC: 802.11k Link Measurement - Raw + converted ---

                     << "," << (int)real.rcpi << "," << (int)real.rsni << "," << (int)real.rssi_dbm
                     << "," << (int)real.snr_db << "," << (int)real.link_margin_db << ","
                     << (int)oracle.tx_power_dbm
                     // --- REALISTIC: MAC Layer Observations ---

                     << "," << (int)real.last_rx_frame_type << ","
                     << (int)real.last_rx_frame_subtype << "," << (int)real.last_rx_mcs << ","
                     << (int)real.last_rx_nss << "," << (int)real.channel_width_mhz << ","
                     << real.guard_interval_ns << "," << (int)real.power_mgmt_bit << ","
                     << real.last_rx_timestamp_us
                     // --- REALISTIC: AP-derived ---

                     << "," << real.bytes_received_at_ap << ","
                     << real.packets_received_at_ap
                     // --- REALISTIC: Per-AC TX tracking ---

                     << "," << real.bytes_received_ac_vo << "," << real.bytes_received_ac_vi << ","
                     << real.bytes_received_ac_be << "," << real.bytes_received_ac_bk << ","
                     << real.packets_received_ac_vo << "," << real.packets_received_ac_vi << ","
                     << real.packets_received_ac_be << "," << real.packets_received_ac_bk << ","
                     << real.airtime_used_us
                     // --- REALISTIC: Device characteristics ---

                     << "," << (int)oracle.device_class << "," << std::fixed << std::setprecision(2)
                     << oracle.mean_data_rate_kbps << "," << std::fixed << std::setprecision(1)
                     << oracle.delay_bound_ms << "," << (int)real.user_priority << ","
                     << real.observation_sequence_num
                     // --- ORACLE: Energy ---

                     << "," << std::fixed << std::setprecision(4) << oracle.total_energy_consumed_mj
                     << "," << std::fixed << std::setprecision(3) << oracle.awake_time_ms << ","
                     << std::fixed << std::setprecision(3) << oracle.sleep_time_ms << ","
                     << std::fixed << std::setprecision(6)
                     << oracle.duty_cycle
                     // --- ORACLE: Application layer ---

                     << "," << oracle.packets_generated << "," << oracle.packets_enqueued << ","
                     << oracle.bytes_generated
                     // --- ORACLE: Queue drops ---

                     << "," << oracle.mpdu_drops_expired << "," << oracle.mpdu_drops_queue_full
                     << ","
                     << oracle.psdu_response_timeouts
                     // --- ORACLE: TX metrics ---

                     << "," << oracle.packets_transmitted << "," << oracle.bytes_transmitted << ","
                     << oracle.ampdu_count << "," << oracle.ampdu_mpdus_total << ","
                     << oracle.ampdu_bytes_total
                     // --- ORACLE: Queue state ---

                     << "," << oracle.queue_size_packets << "," << oracle.queue_size_bytes << "\n";
    }

    m_lastBiLogTime_ms = simTime_ms;
    m_biLevelCsv.flush();
    m_biLevelObservationCount++;
}

// Call-level logging: Initialize CSV file and logging state
void
TwtMetrics::InitializeCallLevelLogging(std::string logdir)
{
    // Generate timestamp for unique filename
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = oss.str();

    // Create CSV file for Call-level observations
    std::string callLevelFile = logdir + "/ns3-call-log-" + timestamp + ".csv";
    m_callLevelCsv.open(callLevelFile, std::ios::app);

    if (!m_callLevelCsv.is_open())
    {
        std::cerr << "ERROR: Failed to open Call-level logging file: " << callLevelFile
                  << std::endl;
        return;
    }

    m_callLevelLoggingEnabled = true;
    m_callLevelObservationCount = 0;
    m_lastCallLogTime_ms = Simulator::Now().GetSeconds() * 1000.0;
    m_callLevelFirstCall = true;

    // Write CSV header - Separated REALISTIC (802.11ax/k) and ORACLE sections
    m_callLevelCsv
        << "call_index,simulation_time_ms,sta_id,"
        // --- REALISTIC: 802.11ax BSR per-AC (quantized 0-255) ---

        << "bsr_ac_be,bsr_ac_bk,bsr_ac_vi,bsr_ac_vo,bsr_scaling_factor,"
        // --- REALISTIC: 802.11k STA Statistics ---

        << "tx_fragment_count,tx_failed_count,tx_retry_count,"
        << "ack_failure_count,rx_fragment_count,fcs_error_count,"
        // --- REALISTIC: 802.11k Link Measurement - Raw + converted ---

        << "rcpi,rsni,rssi_dbm,snr_db,link_margin_db,tx_power_dbm,"
        // --- REALISTIC: MAC Layer Observations ---

        << "last_rx_frame_type,last_rx_frame_subtype,"
        << "last_rx_mcs,last_rx_nss,channel_width_mhz,guard_interval_ns,"
        << "power_mgmt_bit,last_rx_timestamp_us,"
        // --- REALISTIC: AP-derived ---

        << "bytes_received_at_ap,packets_received_at_ap,"
        // --- REALISTIC: Per-AC TX tracking ---

        << "bytes_received_ac_vo,bytes_received_ac_vi,bytes_received_ac_be,bytes_received_ac_bk,"
        << "packets_received_ac_vo,packets_received_ac_vi,packets_received_ac_be,packets_received_"
           "ac_bk,"
        << "airtime_used_us,"
        // --- REALISTIC: Device characteristics ---

        << "device_class,mean_data_rate_kbps,delay_bound_ms,user_priority,"
        << "observation_sequence_num,"
        // --- ORACLE: Energy ---

        << "oracle_total_energy_mj,oracle_awake_time_ms,oracle_sleep_time_ms,oracle_duty_cycle,"
        // --- ORACLE: Application layer ---

        << "oracle_packets_generated,oracle_packets_enqueued,oracle_bytes_generated,"
        // --- ORACLE: Queue drops ---

        << "oracle_mpdu_drops_expired,oracle_mpdu_drops_queue_full,oracle_psdu_timeouts,"
        // --- ORACLE: TX metrics ---

        << "oracle_packets_transmitted,oracle_bytes_transmitted,"
        << "oracle_ampdu_count,oracle_ampdu_mpdus_total,oracle_ampdu_bytes_total,"
        // --- ORACLE: Queue state ---

        << "oracle_queue_size_packets,oracle_queue_size_bytes\n";

    m_callLevelCsv.flush();
    std::cout << "Call-level logging initialized: " << callLevelFile << std::endl;
}

// Fallback logging: Initialize CSV file for tracking null pointer fallbacks
void
TwtMetrics::InitializeFallbackLogging(std::string logdir)
{
    // Generate timestamp for unique filename
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = oss.str();

    // Create CSV file for fallback logging
    std::string fallbackFile = logdir + "/ns3-fallback-log-" + timestamp + ".csv";
    m_fallbackCsv.open(fallbackFile, std::ios::app);

    if (!m_fallbackCsv.is_open())
    {
        std::cerr << "ERROR: Failed to open fallback logging file: " << fallbackFile << std::endl;
        return;
    }

    m_fallbackLoggingEnabled = true;

    // Write CSV header
    m_fallbackCsv << "simulation_time_ms,sta_id,variable_name\n";
    m_fallbackCsv.flush();
    std::cout << "Fallback logging initialized: " << fallbackFile << std::endl;
}

// Fallback logging: Log when trace array is NULL and fallback to 0 is triggered
void
TwtMetrics::LogFallback(double time_ms, uint32_t staId, const std::string& variable_name)
{
    if (!m_fallbackLoggingEnabled || !m_fallbackCsv.is_open())
    {
        return;
    }

    m_fallbackCsv << std::fixed << std::setprecision(3) << time_ms << "," << staId << ","
                  << variable_name << "\n";
    m_fallbackCsv.flush();
}

// Log call-level metrics and send to Python controller
EnvStruct
TwtMetrics::LogAndSendCallLevelMetrics()
{
    EnvStruct env;
    std::memset(&env, 0, sizeof(EnvStruct));

    double simTime_ms = Simulator::Now().GetSeconds() * 1000.0;

    // On first call, just initialize and reset BSR accumulators
    if (m_callLevelFirstCall)
    {
        m_callLevelFirstCall = false;
        m_lastCallLogTime_ms = simTime_ms;
        for (uint32_t staId = 0; staId < m_staNodes.GetN(); staId++)
        {
            // Reset BSR accumulators for next call window
            m_bsrOccupancySum[staId] = 0.0;
            m_bsrSampleCount[staId] = 0;
            m_bsrAbove50Count[staId] = 0;
            m_bsrAbove75Count[staId] = 0;
            m_bsrAbove95Count[staId] = 0;
        }
        env.num_sta = m_staNodes.GetN();
        env.simulation_time_sec = simTime_ms / 1000.0;
        env.beacon_interval_ms = m_config.beaconInterval_s.GetSeconds() * 1000.0;
        return env;
    }

    m_callLevelObservationCount++;

    // Metadata
    env.num_sta = m_staNodes.GetN();
    env.simulation_time_sec = simTime_ms / 1000.0;
    env.observation_timestamp_ms = static_cast<uint64_t>(simTime_ms);
    env.observation_count = m_callLevelObservationCount;
    env.beacon_interval_ms = m_config.beaconInterval_s.GetSeconds() * 1000.0;

    // Global AP TX state (REALISTIC — the AP knows its own TX power and what it broadcasts).
    // Diagnostic/visibility only; not load-bearing for the reward.
    env.ap_tx_power_dbm = AP_TX_POWER_DBM;
    env.dl_broadcast_bytes = g_dlBroadcastBytes;
    env.dl_broadcast_duration_us = g_dlBroadcastDurationUs;
    env.dl_broadcast_count = g_dlBroadcastCount;

    // Collect per-STA observations (raw cumulative)
    for (uint32_t staId = 0; staId < m_staNodes.GetN(); staId++)
    {
        StaEnvStruct& sta = env.sta_observations[staId];
        std::memset(&sta, 0, sizeof(StaEnvStruct));

        // Populate with raw data (fills realistic + oracle sub-structs)
        PopulateStaObservationRaw(sta, staId);

        // Reset BSR accumulators for next call window
        m_bsrOccupancySum[staId] = 0.0;
        m_bsrSampleCount[staId] = 0;
        m_bsrAbove50Count[staId] = 0;
        m_bsrAbove75Count[staId] = 0;
        m_bsrAbove95Count[staId] = 0;
    }

    // Aggregate TWT metrics (group assignments, STA counts) are NOT included in EnvStruct because the Python controller already knows this info from the ActionStruct it sent.
    // All aggregations should be computed in Python.

    // Log call-level metrics to CSV
    if (m_callLevelLoggingEnabled && m_callLevelCsv.is_open())
    {
        for (uint32_t staId = 0; staId < m_staNodes.GetN(); staId++)
        {
            const StaEnvStruct& sta = env.sta_observations[staId];
            const StaRealisticMetrics& real = sta.realistic;
            const StaOracleMetrics& oracle = sta.oracle;

            m_callLevelCsv
                << m_callLevelObservationCount << "," << std::fixed << std::setprecision(3)
                << simTime_ms << ","
                << real.sta_id
                // --- REALISTIC: BSR per-AC (quantized) ---

                << "," << (int)real.bsr_queue_ac_be << "," << (int)real.bsr_queue_ac_bk << ","
                << (int)real.bsr_queue_ac_vi << "," << (int)real.bsr_queue_ac_vo << ","
                << (int)real.bsr_scaling_factor
                // --- REALISTIC: AP-Observable RX Counters ---

                << "," << real.rx_fragment_count << ","
                << real.fcs_error_count
                // --- ORACLE: STA TX Counters (would need 802.11k request) ---

                << "," << oracle.tx_fragment_count << "," << oracle.tx_failed_count << ","
                << oracle.tx_retry_count << ","
                << oracle.ack_failure_count
                // --- REALISTIC: Link Measurement - Raw + converted ---

                << "," << (int)real.rcpi << "," << (int)real.rsni << "," << (int)real.rssi_dbm
                << "," << (int)real.snr_db << "," << (int)real.link_margin_db << ","
                << (int)oracle.tx_power_dbm
                // --- REALISTIC: MAC Layer Observations ---

                << "," << (int)real.last_rx_frame_type << "," << (int)real.last_rx_frame_subtype
                << "," << (int)real.last_rx_mcs << "," << (int)real.last_rx_nss << ","
                << (int)real.channel_width_mhz << "," << real.guard_interval_ns << ","
                << (int)real.power_mgmt_bit << ","
                << real.last_rx_timestamp_us
                // --- REALISTIC: AP-derived ---

                << "," << real.bytes_received_at_ap << ","
                << real.packets_received_at_ap
                // --- REALISTIC: Per-AC TX tracking ---

                << "," << real.bytes_received_ac_vo << "," << real.bytes_received_ac_vi << ","
                << real.bytes_received_ac_be << "," << real.bytes_received_ac_bk << ","
                << real.packets_received_ac_vo << "," << real.packets_received_ac_vi << ","
                << real.packets_received_ac_be << "," << real.packets_received_ac_bk << ","
                << real.airtime_used_us
                // --- REALISTIC: Device characteristics ---

                << "," << (int)oracle.device_class << "," << std::fixed << std::setprecision(2)
                << oracle.mean_data_rate_kbps << "," << std::fixed << std::setprecision(1)
                << oracle.delay_bound_ms << "," << (int)real.user_priority << ","
                << real.observation_sequence_num
                // --- ORACLE: Energy ---

                << "," << std::fixed << std::setprecision(4) << oracle.total_energy_consumed_mj
                << "," << std::fixed << std::setprecision(3) << oracle.awake_time_ms << ","
                << std::fixed << std::setprecision(3) << oracle.sleep_time_ms << "," << std::fixed
                << std::setprecision(6)
                << oracle.duty_cycle
                // --- ORACLE: Application layer ---

                << "," << oracle.packets_generated << "," << oracle.packets_enqueued << ","
                << oracle.bytes_generated
                // --- ORACLE: Queue drops ---

                << "," << oracle.mpdu_drops_expired << "," << oracle.mpdu_drops_queue_full << ","
                << oracle.psdu_response_timeouts
                // --- ORACLE: TX metrics ---

                << "," << oracle.packets_transmitted << "," << oracle.bytes_transmitted << ","
                << oracle.ampdu_count << "," << oracle.ampdu_mpdus_total << ","
                << oracle.ampdu_bytes_total
                // --- ORACLE: Queue state ---

                << "," << oracle.queue_size_packets << "," << oracle.queue_size_bytes << "\n";
        }
        m_callLevelCsv.flush();
    }

    m_lastCallLogTime_ms = simTime_ms;
    return env;
}

} // namespace ns3