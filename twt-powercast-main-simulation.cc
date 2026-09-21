// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file twt-powercast-main-simulation.cc
 * @brief NS-3 main() for the TWT + PowerCast RL environment
 */

#include "pb-twt-wrapper-ns3.h"
#include "twt-metrics.h"
#include "twt-simulation-config.h"
#include "twt-trace-callbacks.h"

#include "ns3/applications-module.h"
#include "ns3/bsr-manager.h"
#include "ns3/mobility-module.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/log.h"
#include "ns3/rng-seed-manager.h"

#include <fstream> // std::ofstream (diagnostic --logVcap time-series CSV)
#include <iomanip> // std::fixed, std::setprecision (for the always-on TWT progress beacon)

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UnilateralTwtDemo");

// Global pointers for periodic TWT update
static Ptr<TWTWrapper> g_twtWrapper = nullptr;
static TwtNetworkSetup* g_networkSetup = nullptr;
static TwtMetrics* g_metrics = nullptr;
static Time g_updateInterval;
static Time g_beaconInterval;
static uint32_t g_updateCount = 0;
// Identifier carried into the always-on per-update progress beacon so users can tell which worker each line came from when 4 workers run in parallel.
// Set in main() from the unique segmentName CLI arg.
static std::string g_simLabel = "ns3";

// --- Diagnostic: per-REHD Vcap(t) time-series CSV (opt-in via --logVcap) ---
// Disabled by default; the Python training/eval pipeline never passes the flag, so this adds zero overhead to normal runs.
// Samples every g_vcapIntervalMs.
static std::ofstream g_vcapFile;
static double g_vcapIntervalMs = 50.0;
static uint32_t g_vcapNumSta = 0;

// Periodic Vcap sampler — writes one row per REHD harvester each tick.
// Iterates all STA indices and skips null harvesters (non-REHD STAs).
void
SampleVcapTrace()
{
    if (!g_networkSetup || !g_vcapFile.is_open())
    {
        return;
    }
    const double now_ms = Simulator::Now().GetMilliSeconds();
    for (uint32_t i = 0; i < g_vcapNumSta; ++i)
    {
        auto h = g_networkSetup->GetHarvester(i);
        if (!h)
        {
            continue; // non-REHD STA — no harvester
        }
        double dist_m = 0.0;
        Ptr<Node> rehdNode = g_networkSetup->GetStaNodes().Get(i);
        Ptr<MobilityModel> rehdMob = rehdNode ? rehdNode->GetObject<MobilityModel>() : nullptr;
        if (rehdMob)
        {
            dist_m = CalculateDistance(rehdMob->GetPosition(), Vector(0.0, 0.0, 0.0));
        }
        g_vcapFile << now_ms << "," << i << "," << h->GetVcap() << "," << h->GetCapMinVoltage()
                   << "," << h->GetCapMaxVoltage() << "," << static_cast<int>(h->ComputePhase())
                   << "," << g_networkSetup->GetStaQueueDepth(i) << "," << dist_m << "\n";
    }
    Simulator::Schedule(MilliSeconds(g_vcapIntervalMs), &SampleVcapTrace);
}

// BI-level metrics logging callback - runs every beacon interval
void
PeriodicBiLevelLogging()
{
    if (!g_metrics)
    {
        return;
    }

    g_metrics->LogBiLevelMetrics();

    // Schedule next logging event
    Simulator::Schedule(g_beaconInterval, &PeriodicBiLevelLogging);
}

// Periodic TWT update callback - runs during simulation
void
PeriodicTWTUpdate()
{
    if (!g_twtWrapper || !g_networkSetup || !g_metrics)
    {
        std::cout << "[TWT ERROR: Null pointers detected!" << std::endl;
        return;
    }

    g_updateCount++;

    // One-line per-update progress beacon.
    // Useful when running NS-3 directly to see episodes advancing, but suppressed under --quietMode so parallel training workers stay silent (the orchestrator's tqdm bar already shows batch-level progress).
    // Format: [<segmentName> twt 7/38 t=23.55s] -- short enough to interleave cleanly when 4 workers run in parallel.
    if (!g_quietMode)
    {
        std::cout << "[" << g_simLabel << " twt " << g_updateCount << "/" << DURATION_IN_UPDATE
                  << " t=" << std::fixed << std::setprecision(2) << Simulator::Now().GetSeconds()
                  << "s]" << std::endl;
    }

    if (!g_quietMode)
    {
        std::cout << "\n\033[34m========== Dynamic TWT Update #" << g_updateCount
                  << " (t=" << Simulator::Now().GetSeconds() << "s) ==========\033[0m" << std::endl;
    }

    // 1. Collect current environment metrics (also logs call-level metrics internally)
    EnvStruct env = g_metrics->LogAndSendCallLevelMetrics();

    if (!g_quietMode)
    {
        std::cout << "\033[36m[Metrics] Collected environment data:\033[0m" << std::endl;
        std::cout << "  • STAs: " << env.num_sta << std::endl;
        std::cout << "  • Simulation Time: " << env.simulation_time_sec << "s" << std::endl;
        // 2. Request new TWT schedule from Python
        std::cout << "\033[32m[Controller] Requesting TWT schedule from Python...\033[0m"
                  << std::endl;
    }
    ActionStruct action = g_twtWrapper->RequestTWTSchedule(env);

    // 3. Apply new TWT schedule
    if (!g_quietMode)
    {
        std::cout << "\033[32m[Controller] Received action with " << action.num_active_twt_groups
                  << " active TWT groups\033[0m" << std::endl;
    }
    g_networkSetup->ApplyTWTSchedule(action);

    // 5. Schedule next update only if we haven't reached the limit
    if (g_updateCount < DURATION_IN_UPDATE)
    {
        Simulator::Schedule(g_updateInterval, &PeriodicTWTUpdate);
    }
    else if (!g_quietMode)
    {
        std::cout << "\033[32m[TWT] All " << DURATION_IN_UPDATE
                  << " updates completed - no more updates scheduled\033[0m" << std::endl;
    }
}

int
main(int argc, char* argv[])
{
    // --- CONFIGURATION ---

    TwtSimulationConfig config;

    // Import beacon interval for timing calculations
    Time beaconInterval_s = config.beaconInterval_s;

    bool enableDynamicTWT = true;
    // In what interval we will update TWT settings (in BI)
    double twtUpdateInterval_s = (TWT_UPDATE_INTERVAL_BI * beaconInterval_s).GetSeconds();
    // When would be the first update from python coming in (in BI)
    double twtUpdateStart_s = (TWT_UPDATE_START_BI * beaconInterval_s).GetSeconds();

    // Custom shared memory segment names (for parallel/subprocess execution)
    std::string segmentName = "My Seg";
    std::string cpp2pyMsgName = "My Cpp to Python Msg";
    std::string py2cppMsgName = "My Python to Cpp Msg";
    std::string lockableName = "My Lockable";

    // Verbosity / logging controls (set by Python wrapper to TRUE during training, default FALSE here so direct `./ns3 run twt-powercast-main-simulation` invocations keep the existing chatty/full-CSV behavior).
    bool quietMode = false;       // suppress non-error std::cout in the dynamic-TWT loop
    bool disableTraces = false;   // skip OpenTraceFiles + Initialize*Logging + EnableLogging
    bool logVcap = false;         // diagnostic: write per-REHD Vcap(t) CSV (opt-in)
    double vcapIntervalMs = 50.0; // sampling period for --logVcap

    // --- SINGLE SOURCE OF TRUTH: NUMBER OF STAs ---

    // Number of active STAs is initialized in TwtSimulationConfig constructor with DEFAULT_NUM_STA.
    // Can be overridden by --nStations command line argument.
    // This value propagates to:
    //   1. Node creation: TwtNetworkSetup::CreateNodes() uses m_config.nStations
    //   2. Environment data: TwtMetrics::LogAndSendCallLevelMetrics() sets env.num_sta from node count
    //   3. Python controller: Reads env_dict["num_sta"] and echoes it back in action_dict["num_sta"]
    //   4. TWT application: Uses action.num_sta to iterate over STAs when applying schedules
    // nStations is now initialized in TwtSimulationConfig constructor (pb-twt-core.h: DEFAULT_NUM_STA = 5).

    CommandLine cmd(__FILE__);
    cmd.AddValue("simId", "Simulation ID", config.simId);
    cmd.AddValue("randSeed", "Random seed", config.randSeed);
    cmd.AddValue("parallelSim", "Parallel simulation mode", config.parallelSim);
    cmd.AddValue("scenario", "Scenario name", config.scenario);
    cmd.AddValue("nStations",
                 "Number of NON-REHD stations (split uniformly across IoT/Camera/Voice/Video)",
                 config.nStations);
    cmd.AddValue("nRehd",
                 "Number of REHD (RF energy-harvesting) stations; sub-type drawn uniformly from "
                 "the REHD type catalog at episode start. Total active STAs = nStations + nRehd.",
                 config.nRehd);
    cmd.AddValue("trafficScale",
                 "Multiplier on non-REHD offered traffic rate (1.0 = default; raise to load the "
                 "comms window)",
                 config.trafficScale);
    cmd.AddValue("simulationTime", "Simulation time in milliseconds", config.simulationTime_ms);
    cmd.AddValue("twtSleepGroups",
                 "Standalone grouped-sleep test: round-robin STAs into N TWT groups with "
                 "staggered SPs + short wake (0 = permissive default).",
                 config.twtSleepGroups);
    cmd.AddValue("twtGroupWakeMs",
                 "Per-SP wake duration (ms) for --twtSleepGroups (controls sleep/harvest ratio).",
                 config.twtGroupWakeMs);
    cmd.AddValue("p2pLinkDelay", "P2P link delay in ms", config.p2pLinkDelay_ms);
    cmd.AddValue("enableStateLogs", "Enable state logs", config.enableStateLogs);
    cmd.AddValue("enablePcap", "Enable PCAP", config.enablePcap);
    cmd.AddValue("enableDynamicTWT", "Enable dynamic TWT reconfiguration", enableDynamicTWT);
    cmd.AddValue("twtUpdateInterval", "TWT update interval in seconds", twtUpdateInterval_s);
    cmd.AddValue("twtUpdateStart", "When to start dynamic TWT updates (seconds)", twtUpdateStart_s);
    cmd.AddValue("segmentName", "Shared memory segment name (for parallel episodes)", segmentName);
    cmd.AddValue("cpp2pyMsgName", "Cpp to Python message name", cpp2pyMsgName);
    cmd.AddValue("py2cppMsgName", "Python to Cpp message name", py2cppMsgName);
    cmd.AddValue("lockableName", "Lockable name for synchronization", lockableName);
    cmd.AddValue("quietMode",
                 "Suppress non-error std::cout in the dynamic-TWT loop (default false)",
                 quietMode);
    cmd.AddValue("disableTraces",
                 "Skip OpenTraceFiles + Initialize*Logging + EnableLogging (default false). "
                 "Internal trace callbacks still fire to populate env-data arrays.",
                 disableTraces);
    cmd.AddValue("logVcap",
                 "Diagnostic: write per-REHD Vcap(t) time-series to "
                 "data-log/vcap_trace_<simId>.csv (default false)",
                 logVcap);
    cmd.AddValue("vcapIntervalMs", "Sampling period (ms) for --logVcap", vcapIntervalMs);
    cmd.AddValue("pdwDurationMs",
                 "Phase-1 Power Delivery Window: AP broadcasts dummy frames over [0,D] ms of "
                 "each beacon interval (D>0 enables; REHD TWT SPs auto-shift to start after D so "
                 "they doze through it and harvest). Pure power delivery, not tracked. 0=disabled.",
                 config.pdwDurationMs);
    cmd.Parse(argc, argv);

    if (config.pdwDurationMs < 0.0 || config.pdwDurationMs >= 94.0)
    {
        std::cerr << "ERROR: --pdwDurationMs must be in [0, 94) (must leave room for the TWT SP "
                  << "window inside the " << BEACON_INTERVAL_MS << " ms BI); got "
                  << config.pdwDurationMs << std::endl;
        return 1;
    }

    // Propagate to globals so twt-trace-callbacks.cc / twt-simulation-config.cc can early-return on hot paths.
    g_quietMode = quietMode;
    g_disableTraces = disableTraces;
    g_simLabel = segmentName; // unique per worker (e.g. "MySeg_90000")

    if (!g_quietMode)
    {
        std::cout << "finished setting up command line" << std::endl;
    }

    if (!g_quietMode)
    {
        std::cout << "\n\n========== UNILATERAL TWT SIMULATION ==========" << std::endl;
        std::cout << "Simulation ID: " << config.simId << std::endl;
        std::cout << "Number of STAs: " << config.nStations << " non-REHD + " << config.nRehd
                  << " REHD = " << config.nTotal() << " total" << std::endl;
        std::cout << "TWT Mode: UNILATERAL (AP announces, no STA negotiation)" << std::endl;
        if (enableDynamicTWT)
        {
            std::cout << "Dynamic TWT: ENABLED (updates every " << twtUpdateInterval_s
                      << "s starting at t=" << twtUpdateStart_s << "s)" << std::endl;
        }
        std::cout << "===============================================\n\n" << std::endl;
    }

    // Generate simulation ID string
    config.GenerateSimIdString();

    // Write device class assignments to file
    config.WriteDeviceClassAssignments();

    // Export config parameters to trace callbacks module (in milliseconds)
    keepTrackOfMetricsFrom_ms = config.keepTrackOfMetricsFrom_ms;
    TI_currentModel_mA = config.TI_currentModel_mA;

    // --- RNG SEED (must be before ANY CreateObject<*RandomVariable>) ---

    // InitializeHeterogeneousNetwork() (called by CreateNodes) draws random device-class assignments using UniformRandomVariable.
    // Setting the seed here — after cmd.Parse() has populated config.randSeed — ensures every unique seed yields a unique traffic pattern.
    // The duplicate SetSeed/SetRun inside ConfigureWifi() is harmless (same values).
    RngSeedManager::SetSeed(config.randSeed);
    RngSeedManager::SetRun(RNG_DEFAULT_RUN_NUMBER);

    // --- NS-3 GLOBAL CONFIGURATION ---

    if (!g_quietMode)
    {
        std::cout << "setting up ns-3 global configuration" << std::endl;
    }
    Config::SetDefault("ns3::ArpCache::MaxRetries", UintegerValue(20));
    Config::SetDefault("ns3::ArpCache::WaitReplyTimeout", TimeValue(MilliSeconds(1000)));
    Config::SetDefault("ns3::WifiMacQueue::MaxDelay", TimeValue(MilliSeconds(1000)));
    Config::SetDefault("ns3::WifiMacQueue::MaxSize", QueueSizeValue(QueueSize("1000p")));
    Config::SetDefault("ns3::QosFrameExchangeManager::SetQueueSize", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(config.payloadSize));

    // --- NETWORK SETUP ---

    if (!g_quietMode)
    {
        std::cout << "setting up network and nodes" << std::endl;
    }
    TwtNetworkSetup networkSetup(config);
    networkSetup.CreateNodes();
    networkSetup.ConfigureWifi();
    networkSetup.SetupMobility();
    networkSetup.ScheduleDynamicMobility(); // no-op when ENABLE_DYNAMIC_MOBILITY=0
    networkSetup.ConfigureInternet();
    networkSetup.SetupEnergyHarvesting(); // phase 1: PowerCast harvester per STA
    networkSetup.SetupApplications();
    networkSetup.ScheduleDynamicTraffic(); // no-op when ENABLE_DYNAMIC_TRAFFIC=0
    if (!g_quietMode)
    {
        std::cout << "setting up initial TWT schedule without python input" << std::endl;
    }
    networkSetup.SetupTwtSchedule();
    networkSetup.EnablePcap();

    // --- METRICS INITIALIZATION ---

    if (!g_quietMode)
    {
        std::cout << "setting up metrics collection" << std::endl;
    }
    TwtMetrics metrics(config, networkSetup.GetStaNodes());
    metrics.InitializeArrays();
    metrics.SetNetworkSetup(&networkSetup);

    // Initialize BI-level + call-level CSV logging.
    // Skip both when --disableTraces is set: env data still flows over SHM via PopulateStaObservationRaw, and the CSV-write branches inside the metrics module are gated by m_*Enabled+is_open().
    std::string biLevelLogDir = TwtResultsPath("data-log");
    if (!disableTraces)
    {
        metrics.InitializeBiLevelLogging(biLevelLogDir);
        metrics.InitializeCallLevelLogging(biLevelLogDir);
    }

    // --- DYNAMIC TWT SETUP ---

    if (!g_quietMode)
    {
        std::cout << "setting up dynamic TWT wrapper" << std::endl;
    }
    Ptr<TWTWrapper> twtWrapper = nullptr;
    if (enableDynamicTWT)
    {
        if (!g_quietMode)
        {
            std::cout << "\n\033[34m========== DYNAMIC TWT INITIALIZATION ==========\033[0m"
                      << std::endl;
        }

        // Configure shared memory segment names BEFORE creating TWTWrapper.
        // This is critical for subprocess-based training where each episode needs unique names.
        auto msgInterface = Ns3AiMsgInterface::Get();
        msgInterface->SetNames(segmentName, cpp2pyMsgName, py2cppMsgName, lockableName);
        if (!g_quietMode)
        {
            std::cout << "\033[36m• Shared memory segment: " << segmentName << "\033[0m"
                      << std::endl;
        }

        twtWrapper = CreateObject<TWTWrapper>();

        std::string twtLogFile =
            TwtResultsPath("data-log/ns3-twt-wrapper-") + config.currentsimId_string + ".csv";
        if (!disableTraces)
        {
            twtWrapper->EnableLogging(true, twtLogFile);
        }

        if (!twtWrapper->Initialize())
        {
            std::cout
                << "\033[31m[ERROR] Failed to initialize TWTWrapper! Disabling dynamic TWT.\033[0m"
                << std::endl;
            enableDynamicTWT = false;
        }
        else
        {
            if (!g_quietMode)
            {
                std::cout << "\033[32m✓ TWTWrapper initialized successfully\033[0m" << std::endl;
                std::cout << "\033[32m✓ Python controller ready\033[0m" << std::endl;
                std::cout << "\033[36m• Update interval: " << twtUpdateInterval_s
                          << " seconds\033[0m" << std::endl;
                std::cout << "\033[36m• First update at: t=" << twtUpdateStart_s << "s\033[0m"
                          << std::endl;
                std::cout << "\033[36m• Log file: " << twtLogFile << "\033[0m" << std::endl;
            }

            // Set global pointers for periodic update
            g_twtWrapper = twtWrapper;
            g_networkSetup = &networkSetup;
            g_metrics = &metrics;
            g_updateInterval = Seconds(twtUpdateInterval_s);
            g_beaconInterval = config.beaconInterval_s;
        }

        if (!g_quietMode)
        {
            std::cout << "\033[34m===============================================\033[0m\n"
                      << std::endl;
        }
    }

    // --- TRACE FILES ---

    // Skip when --disableTraces: e2e/queue/ampdu/bsr/etc. CSV files stay closed.
    // The trace callbacks (connected below) still fire and update env-data arrays;
    // their CSV-write branches are guarded with is_open() so they auto-no-op.
    if (!disableTraces)
    {
        OpenTraceFiles(config.currentsimId_string);
    }

    if (!g_quietMode)
    {
        std::cout << "\n===== TRACING ENABLED =====" << std::endl;
    }
    // std::cout << "E2E Trace file: data-log/e2e_trace_" << config.currentsimId_string
    //           << ".csv (from 10s to end)" << std::endl;
    // std::cout
    //     << "Packet journey: STA App → STA IP → STA MAC → STA PHY → AP PHY → Server IP → Server
    //     App"
    //     << std::endl;
    // std::cout << "Shows: Complete packet path with timestamps and delays at each stage\n"
    //           << std::endl;

    // std::cout << "===== QoS METRICS TRACING ENABLED =====" << std::endl;
    // std::cout << "Queue Size Trace: data-log/queue_size_trace_" << config.currentsimId_string
    //           << ".csv" << std::endl;
    // std::cout << "  Tracks: MAC queue depth at each STA over time (ground truth)" << std::endl;
    // std::cout << "  Compare with BSR to see reporting accuracy\n" << std::endl;
    // std::cout << "A-MPDU Trace: data-log/ampdu_trace_" << config.currentsimId_string << ".csv"
    //           << std::endl;
    // std::cout << "  Tracks: Frame aggregation events (MPDUs per A-MPDU)" << std::endl;
    // std::cout << "  Shows: Aggregation efficiency during TWT wake windows\n" << std::endl;
    // std::cout << "BSR Tracking: Real-time via BsrManager (no CSV file)" << std::endl;
    // std::cout << "  Runtime access: BsrManager::GetInstance()->GetCurrentQueueSize()" <<
    // std::endl; std::cout << "  Immediate callbacks available for adaptive control\n" <<
    // std::endl;

    // --- CONNECT TRACES ---

    if (!g_quietMode)
    {
        std::cout << "setting up trace connections" << std::endl;
    }
    ConnectSummaryTraces(networkSetup.GetStaNodes());
    ConnectE2ETraces(networkSetup.GetStaNodes(),
                     networkSetup.GetApNodes(),
                     networkSetup.GetServerNode(),
                     networkSetup.GetServerApps());
    ConnectQosMetricTraces(networkSetup.GetStaNodes());
    ConnectPhyStateTraces(networkSetup.GetStaNodes());
    ConnectTimeoutAndDropTraces(networkSetup.GetStaNodes(), networkSetup.GetApNodes());
    Connect802dot11kTraces(networkSetup.GetStaNodes(), networkSetup.GetApNodes());

    // --- BSR MANAGER SETUP ---

    if (!g_quietMode)
    {
        std::cout << "\n===== BSR MANAGER DEMO ENABLED =====" << std::endl;
    }

    Ptr<BsrManager> bsrManager = BsrManager::GetInstance();

    // DEMO 1: Connect callback for immediate BSR notifications
    bool connectCallback = true;
    if (connectCallback)
    {
        bsrManager->TraceConnectWithoutContext("BsrReceived", MakeCallback(&BsrReceivedCallback));
        if (!g_quietMode)
        {
            std::cout << "✓ BSR callback connected - will log BSR events to CSV" << std::endl;
        }
    }

    if (!g_quietMode)
    {
        std::cout << "======================================\n" << std::endl;
    }

    // --- SCHEDULE DYNAMIC TWT UPDATES ---

    if (!g_quietMode)
    {
        std::cout
            << "first cycle starts here, then it is recursively called thru PeriodicTWTUpdate in "
               "the scheduler. Only schedules for now, does not run yet."
            << std::endl;
    }
    if (enableDynamicTWT && twtWrapper)
    {
        if (!g_quietMode)
        {
            std::cout << "\n\033[34m[Dynamic TWT] Scheduling periodic updates...\033[0m"
                      << std::endl;
        }
        Simulator::Schedule(Seconds(twtUpdateStart_s), &PeriodicTWTUpdate);
        if (!g_quietMode)
        {
            std::cout << "\033[32m✓ First update scheduled at t=" << twtUpdateStart_s
                      << "s\033[0m\n"
                      << std::endl;
        }
    }

    // --- SCHEDULE BI-LEVEL LOGGING ---

    // Schedule per-beacon-interval metrics logging starting at TWT setup time
    double twtSetupTime_s = TWT_SETUP_TIME_BI * config.beaconInterval_s.GetSeconds();
    if (!g_quietMode)
    {
        std::cout << "\n\033[35m[BI-Level Logging] Scheduling per-beacon-interval metrics...\033[0m"
                  << std::endl;
    }
    Simulator::Schedule(Seconds(twtSetupTime_s), &PeriodicBiLevelLogging);
    if (!g_quietMode)
    {
        std::cout << "\033[35m✓ BI-level logging scheduled at t=" << twtSetupTime_s << "s (every "
                  << config.beaconInterval_s.GetMilliSeconds() << "ms)\033[0m\n"
                  << std::endl;
    }

    // --- DIAGNOSTIC: per-REHD Vcap(t) sampler (opt-in) ---

    if (logVcap)
    {
        g_networkSetup = &networkSetup; // ensure set even in standalone (non-dynamic) runs
        g_vcapIntervalMs = vcapIntervalMs;
        g_vcapNumSta = static_cast<uint32_t>(config.nTotal());
        const std::string vcapPath =
            TwtResultsPath("data-log/vcap_trace_") + std::to_string(config.simId) + ".csv";
        g_vcapFile.open(vcapPath);
        if (g_vcapFile.is_open())
        {
            g_vcapFile << "time_ms,sta_id,vcap_v,vmin_v,vmax_v,phase,queue_pkts,dist_m\n";
            Simulator::Schedule(MilliSeconds(g_vcapIntervalMs), &SampleVcapTrace);
            if (!g_quietMode)
            {
                std::cout << "[Vcap] logging per-REHD Vcap(t) every " << g_vcapIntervalMs << "ms → "
                          << vcapPath << std::endl;
            }
        }
    }

    // --- RUN SIMULATION ---

    if (!g_quietMode)
    {
        std::cout << "running simulation... going back to python" << std::endl;
    }
    Simulator::Stop(MilliSeconds(config.simulationTime_ms));
    Simulator::Run();

    if (g_vcapFile.is_open())
    {
        g_vcapFile.close();
    }

    if (!g_quietMode)
    {
        std::cout << "\n\n====== SIMULATION COMPLETE ======\n" << std::endl;
    }

    // Phase-1 sanity: print per-STA energy totals if harvesters were installed.
    networkSetup.LogFinalEnergyStats();

    // --- SIGNAL PYTHON THAT SIMULATION IS DONE ---

    // This must happen BEFORE we exit, while Python may still be waiting.
    // The ns3-ai library's CppSetFinished() sends a signal through shared memory that unblocks Python's PyRecvBegin() call.
    if (enableDynamicTWT && twtWrapper)
    {
        if (!g_quietMode)
        {
            std::cout << "[TWT] Sending finish signal to Python..." << std::endl;
        }
        auto msgInterface = Ns3AiMsgInterface::Get();
        // Get the typed interface and call finish
        auto typedInterface = msgInterface->GetInterface<EnvStruct, ActionStruct>();
        typedInterface->CppSetFinished();
        if (!g_quietMode)
        {
            std::cout << "[TWT] Finish signal sent" << std::endl;
        }
    }

    // --- RESULTS ---

    CloseTraceFiles();

    if (!g_quietMode)
    {
        std::cout << "\n===== TRACE FILES SAVED =====" << std::endl;
        std::cout << "End-to-end trace: data-log/e2e_trace_" << config.currentsimId_string << ".csv"
                  << std::endl;
        std::cout << "  Complete packet journey with timestamps at each stage" << std::endl;
        std::cout << "  Each packet tracked by UID: STA App → IP → MAC → PHY → AP → Server\n"
                  << std::endl;

        std::cout << "QoS Metrics traces:" << std::endl;
        std::cout << "  Queue Size: data-log/queue_size_trace_" << config.currentsimId_string
                  << ".csv" << std::endl;
        std::cout << "  A-MPDU Aggregation: data-log/ampdu_trace_" << config.currentsimId_string
                  << ".csv" << std::endl;
        std::cout << "  BSR (Buffer Status Report): data-log/bsr_trace_"
                  << config.currentsimId_string << ".csv" << std::endl;

        std::cout << "\n\nSimulation with ID " << config.simId << " completed." << std::endl;
        std::cout << "=========================================\n\n" << std::endl;
    }

    // Release PowerCast harvester Ptr<>s BEFORE Simulator::Destroy() — otherwise file-scope statics in twt-simulation-config.cc are torn down after NS-3 globals at process exit and the Ptr dtors UAF (segfault on teardown).
    networkSetup.TeardownEnergyHarvesting();

    Simulator::Destroy();

    return 0;
}