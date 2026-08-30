// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

#include "pb-twt-wrapper-ns3.h"

#include "pb-twt-core.h"

#include "ns3/ai-module.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

#include <cstring>
#include <fstream>
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TWTWrapper");
NS_OBJECT_ENSURE_REGISTERED(TWTWrapper);

TypeId
TWTWrapper::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TWTWrapper")
                            .SetParent<Object>()
                            .SetGroupName("Ai")
                            .AddConstructor<TWTWrapper>();
    return tid;
}

TWTWrapper::TWTWrapper()
      : m_msgInterface(nullptr)
      , m_initialized(false)
      , m_loggingEnabled(false)
      , m_logFile(TwtResultsPath("data-log/twt-controller-log.csv"))
      , m_txCount(0)
      , m_rxCount(0)
{
    NS_LOG_FUNCTION(this);
}

TWTWrapper::~TWTWrapper()
{
    NS_LOG_FUNCTION(this);
    if (m_csvLogFile.is_open())
    {
        m_csvLogFile.close();
    }
}

bool
TWTWrapper::Initialize()
{
    NS_LOG_FUNCTION(this);

    if (m_initialized)
    {
        NS_LOG_WARN("TWTWrapper already initialized");
        return true;
    }

    NS_LOG_INFO("Initializing TWT Controller Wrapper (Ahmed Maksud, SHINE Lab)");

    auto interface = Ns3AiMsgInterface::Get();
    interface->SetIsMemoryCreator(false); // Python creates shared memory
    interface->SetUseVector(false);       // Simple struct communication
    interface->SetHandleFinish(true);     // Enable cleanup on simulation end

    m_msgInterface = interface->GetInterface<EnvStruct, ActionStruct>();

    if (m_msgInterface)
    {
        m_initialized = true;
        NS_LOG_INFO("TWT Controller Wrapper initialized successfully");
        return true;
    }

    NS_LOG_ERROR("Failed to initialize TWT Controller Wrapper");
    return false;
}

ActionStruct
TWTWrapper::RequestTWTSchedule(const EnvStruct& env)
{
    NS_LOG_FUNCTION(this);

    if (!m_initialized || !m_msgInterface)
    {
        NS_LOG_ERROR("TWTWrapper not initialized - returning error action");
        return CreateErrorAction();
    }

    try
    {
        // Send observations to Python controller
        m_msgInterface->CppSendBegin();
        *m_msgInterface->GetCpp2PyStruct() = env;
        m_msgInterface->CppSendEnd();

        // Receive TWT scheduling decisions
        m_msgInterface->CppRecvBegin();
        ActionStruct action = *m_msgInterface->GetPy2CppStruct();
        m_msgInterface->CppRecvEnd();

        m_txCount++;
        m_rxCount++;

        if (m_loggingEnabled)
        {
            LogInteraction(env, action, true);
        }

        NS_LOG_DEBUG("TWT schedule request completed successfully");
        return action;
    }
    catch (const std::exception& e)
    {
        NS_LOG_ERROR("TWT communication failed: " << e.what());

        ActionStruct error_action = CreateErrorAction();

        if (m_loggingEnabled)
        {
            LogInteraction(env, error_action, false);
        }

        return error_action;
    }
}

void
TWTWrapper::EnableLogging(bool enable, const std::string& logFile)
{
    NS_LOG_FUNCTION(this << enable << logFile);

    m_loggingEnabled = enable;
    if (!logFile.empty())
    {
        m_logFile = logFile;
    }

    if (enable && !m_csvLogFile.is_open())
    {
        m_csvLogFile.open(m_logFile);
        if (m_csvLogFile.is_open())
        {
            // Write CSV header (matches LogInteraction output)
            m_csvLogFile << "Timestamp_Sec,Success,"
                         << "Num_STA,Simulation_Time_Sec,"
                         << "Num_Active_TWT_Groups,"
                         << "TX_Count,RX_Count" << std::endl;
            NS_LOG_INFO("TWT Controller CSV logging enabled: " << m_logFile);
        }
        else
        {
            NS_LOG_ERROR("Failed to open CSV log file: " << m_logFile);
            m_loggingEnabled = false;
        }
    }
    else if (!enable && m_csvLogFile.is_open())
    {
        m_csvLogFile.close();
        NS_LOG_INFO("TWT Controller CSV logging disabled");
    }
}

void
TWTWrapper::GetStatistics(uint32_t& txCount, uint32_t& rxCount) const
{
    txCount = m_txCount;
    rxCount = m_rxCount;
}

void
TWTWrapper::ResetStatistics()
{
    NS_LOG_FUNCTION(this);
    m_txCount = 0;
    m_rxCount = 0;
}

bool
TWTWrapper::IsInitialized() const
{
    return m_initialized;
}

void
TWTWrapper::LogInteraction(const EnvStruct& env, const ActionStruct& act, bool success)
{
    if (!m_csvLogFile.is_open())
    {
        return;
    }

    double timestamp = Simulator::Now().GetSeconds();

    // Simplified logging - aggregate metrics removed since Python controller already knows them
    m_csvLogFile << std::fixed << std::setprecision(6) << timestamp << "," << (success ? "1" : "0")
                 << "," << env.num_sta << "," << env.simulation_time_sec << ","
                 << act.num_active_twt_groups << "," // Use from action instead
                 << m_txCount << "," << m_rxCount << std::endl;
}

ActionStruct
TWTWrapper::CreateErrorAction() const
{
    ActionStruct error_action;
    std::memset(&error_action, 0, sizeof(ActionStruct));
    return error_action;
}