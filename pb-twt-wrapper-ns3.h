// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file pb-twt-wrapper-ns3.h
 * @brief NS-3 Object TWTWrapper declaration for the C++/Python scheduling bridge
 */

#ifndef PB_TWT_WRAPPER_H
#define PB_TWT_WRAPPER_H

#include "pb-twt-core.h"

#include "ns3/ns3-ai-msg-interface.h"
#include "ns3/object.h"

#include <fstream>
#include <string>

using namespace ns3;

/**
 * @brief TWT Controller Wrapper for NS3-AI Python Integration
 *
 * Manages communication between NS-3 TWT simulation and Python RL controller.
 * Provides methods for sending aggregate + per-STA observations and receiving
 * TWT scheduling actions (group assignments, wake intervals, durations).
 */
class TWTWrapper : public Object
{
  public:
    static TypeId GetTypeId();

    TWTWrapper();
    virtual ~TWTWrapper();

    /**
     * @brief Initialize NS3-AI message interface for TWT controller
     * @return true if successfully initialized
     */
    bool Initialize();

    /**
     * @brief Send TWT observations to Python and receive scheduling actions
     * @param env Complete environment state (aggregate + per-STA metrics)
     * @return ActionStruct TWT scheduling decisions from Python controller
     */
    ActionStruct RequestTWTSchedule(const EnvStruct& env);

    /**
     * @brief Enable detailed CSV logging of TWT controller interactions
     * @param enable true to enable logging
     * @param logFile path to CSV log file (default: "twt-controller-log.csv")
     */
    void EnableLogging(bool enable, const std::string& logFile = "twt-controller-log.csv");

    /**
     * @brief Get communication statistics
     * @param txCount Reference to store number of observations sent
     * @param rxCount Reference to store number of actions received
     */
    void GetStatistics(uint32_t& txCount, uint32_t& rxCount) const;

    /**
     * @brief Reset statistics counters to zero
     */
    void ResetStatistics();

    /**
     * @brief Check if wrapper is properly initialized
     * @return true if initialized and ready for communication
     */
    bool IsInitialized() const;

  private:
    Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>* m_msgInterface;
    bool m_initialized;
    bool m_loggingEnabled;
    std::string m_logFile;
    uint32_t m_txCount;
    uint32_t m_rxCount;
    std::ofstream m_csvLogFile;

    void LogInteraction(const EnvStruct& env, const ActionStruct& act, bool success);
    ActionStruct CreateErrorAction() const;
};

#endif // PB_TWT_WRAPPER_H