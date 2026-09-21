/*
 * Copyright (c) 2025 Texas State University
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
 * PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>
 * Texas State University
 */

/**
 * @file bsr-manager.cc
 * @brief Singleton BSR manager implementation with traced callback notification
 */

#include "bsr-manager.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("BsrManager");
NS_OBJECT_ENSURE_REGISTERED(BsrManager);

// Initialize singleton
Ptr<BsrManager> BsrManager::s_instance = nullptr;

TypeId
BsrManager::GetTypeId()
{
    static TypeId tid = TypeId("ns3::BsrManager")
                            .SetParent<Object>()
                            .SetGroupName("Wifi")
                            .AddConstructor<BsrManager>()
                            .AddTraceSource("BsrReceived",
                                            "A Buffer Status Report was received",
                                            MakeTraceSourceAccessor(&BsrManager::m_bsrReceivedTrace),
                                            "ns3::BsrManager::BsrReceivedCallback");
    return tid;
}

BsrManager::BsrManager()
{
    NS_LOG_FUNCTION(this);
}

BsrManager::~BsrManager()
{
    NS_LOG_FUNCTION(this);
}

Ptr<BsrManager>
BsrManager::GetInstance()
{
    if (s_instance == nullptr)
    {
        s_instance = CreateObject<BsrManager>();
    }
    return s_instance;
}

void
BsrManager::RecordBsr(Mac48Address staAddress, uint8_t tid, uint8_t queueSize)
{
    NS_LOG_FUNCTION(this << staAddress << +tid << +queueSize);
    
    uint32_t queueSizeBytes = queueSize * 256;
    
    // Update current state
    auto& state = m_bsrState[staAddress][tid];
    state.queueSizeUnits = queueSize;
    state.queueSizeBytes = queueSizeBytes;
    state.lastUpdateTime = Simulator::Now();
    
    // Fire traced callback IMMEDIATELY for runtime response
    m_bsrReceivedTrace(staAddress, tid, queueSize, queueSizeBytes);
    
    NS_LOG_DEBUG("BSR received: STA=" << staAddress << " TID=" << +tid 
                 << " Queue=" << queueSizeBytes << " bytes at t=" 
                 << Simulator::Now().GetSeconds() << "s");
}

uint32_t
BsrManager::GetCurrentQueueSize(Mac48Address staAddress, uint8_t tid) const
{
    NS_LOG_FUNCTION(this << staAddress << +tid);
    
    auto staIt = m_bsrState.find(staAddress);
    if (staIt != m_bsrState.end())
    {
        auto tidIt = staIt->second.find(tid);
        if (tidIt != staIt->second.end())
        {
            return tidIt->second.queueSizeBytes;
        }
    }
    return 0;
}

uint8_t
BsrManager::GetCurrentQueueSizeUnits(Mac48Address staAddress, uint8_t tid) const
{
    NS_LOG_FUNCTION(this << staAddress << +tid);
    
    auto staIt = m_bsrState.find(staAddress);
    if (staIt != m_bsrState.end())
    {
        auto tidIt = staIt->second.find(tid);
        if (tidIt != staIt->second.end())
        {
            return tidIt->second.queueSizeUnits;
        }
    }
    return 0;
}

bool
BsrManager::IsQueueAboveThreshold(Mac48Address staAddress, uint8_t tid, uint32_t thresholdBytes) const
{
    return GetCurrentQueueSize(staAddress, tid) > thresholdBytes;
}

Time
BsrManager::GetTimeSinceLastBsr(Mac48Address staAddress, uint8_t tid) const
{
    NS_LOG_FUNCTION(this << staAddress << +tid);
    
    auto staIt = m_bsrState.find(staAddress);
    if (staIt != m_bsrState.end())
    {
        auto tidIt = staIt->second.find(tid);
        if (tidIt != staIt->second.end())
        {
            return Simulator::Now() - tidIt->second.lastUpdateTime;
        }
    }
    return Seconds(0);
}

bool
BsrManager::HasBufferedData(Mac48Address staAddress) const
{
    NS_LOG_FUNCTION(this << staAddress);
    
    auto staIt = m_bsrState.find(staAddress);
    if (staIt != m_bsrState.end())
    {
        for (const auto& tidPair : staIt->second)
        {
            if (tidPair.second.queueSizeBytes > 0)
            {
                return true;
            }
        }
    }
    return false;
}

uint32_t
BsrManager::GetTotalBufferedData(Mac48Address staAddress) const
{
    NS_LOG_FUNCTION(this << staAddress);
    
    uint32_t total = 0;
    auto staIt = m_bsrState.find(staAddress);
    if (staIt != m_bsrState.end())
    {
        for (const auto& tidPair : staIt->second)
        {
            total += tidPair.second.queueSizeBytes;
        }
    }
    return total;
}

void
BsrManager::Clear()
{
    NS_LOG_FUNCTION(this);
    m_bsrState.clear();
}

} // namespace ns3
