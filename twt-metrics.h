// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

#ifndef TWT_METRICS_H
#define TWT_METRICS_H

#include "pb-twt-core.h"
#include "twt-simulation-config.h"

#include "ns3/bsr-manager.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace ns3
{

// Metrics collection and reporting class
// Collects RAW cumulative counters from NS-3 and passes them directly to Python/CSV.
// All calculations (rates, deltas, normalization) are done in Python.
class TwtMetrics
{
  public:
    TwtMetrics(const TwtSimulationConfig& config, NodeContainer staNodes);
    ~TwtMetrics();

    void InitializeArrays();
    void CalculateAndPrintResults(FlowMonitorHelper* flowmon, Ptr<FlowMonitor> monitor);
    void PrintPacketFlowStatistics();
    void PrintEnergyStatistics();

    // BI-level metrics logging (per beacon interval)
    void LogBiLevelMetrics();
    void InitializeBiLevelLogging(std::string logdir);

    // Call-level metrics logging (per Python controller call)
    void InitializeCallLevelLogging(std::string logdir);

    // Log call-level metrics and send to Python controller
    EnvStruct LogAndSendCallLevelMetrics();

    // Fallback logging (when trace arrays are NULL)
    void InitializeFallbackLogging(std::string logdir);
    void LogFallback(double time_ms, uint32_t staId, const std::string& variable_name);

    void SetNetworkSetup(TwtNetworkSetup* setup)
    {
        m_networkSetup = setup;
    }

  private:
    const TwtSimulationConfig& m_config;
    NodeContainer m_staNodes;
    TwtNetworkSetup* m_networkSetup;

    // Helper to populate per-STA observation with raw cumulative data
    void PopulateStaObservationRaw(StaEnvStruct& sta, uint32_t staId);

    // BI-level logging members
    std::ofstream m_biLevelCsv;
    bool m_biLevelLoggingEnabled;
    uint64_t m_biLevelObservationCount;
    double m_lastBiLogTime_ms;
    bool m_biLevelFirstCall;

    // Call-level logging members
    std::ofstream m_callLevelCsv;
    bool m_callLevelLoggingEnabled;
    uint64_t m_callLevelObservationCount;
    double m_lastCallLogTime_ms;
    bool m_callLevelFirstCall;

    // Fallback logging (when trace arrays are NULL, fallback to 0)
    std::ofstream m_fallbackCsv;
    bool m_fallbackLoggingEnabled;

    // BSR window accumulators (accumulated during BI-level logging, reset at call-level)
    std::vector<double> m_bsrOccupancySum;   // Sum of occupancy values (0-1) per STA
    std::vector<uint32_t> m_bsrSampleCount;  // Number of BI samples per STA
    std::vector<uint32_t> m_bsrAbove50Count; // Count of BIs with occupancy > 50%
    std::vector<uint32_t> m_bsrAbove75Count; // Count of BIs with occupancy > 75%
    std::vector<uint32_t> m_bsrAbove95Count; // Count of BIs with occupancy > 95%

    // Server apps and throughput tracking
    uint64_t* m_totalRxBytes;
    double* m_throughput;
};

} // namespace ns3

#endif // TWT_METRICS_H