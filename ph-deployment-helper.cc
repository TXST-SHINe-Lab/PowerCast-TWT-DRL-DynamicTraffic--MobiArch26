// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file ph-deployment-helper.cc
 * @brief PowercastEnergyHarvesterHelper implementation
 *
 * - Config-file parsing with per-line validation
 * - Deployment templates and assignment methods for nodes without an explicit entry
 * - Deployment statistics and configuration export
 */
#include "ph-deployment-helper.h"

#include "ns3/log.h"                    // Logging infrastructure
#include "ns3/random-variable-stream.h" // Random number generation
#include "ns3/simulator.h"              // Simulation environment

#include <algorithm> // STL algorithms
#include <fstream>   // File I/O operations
#include <iostream>  // Console I/O
#include <sstream>   // String stream processing

namespace ns3
{
namespace energy
{

// Enable logging for helper development and debugging
NS_LOG_COMPONENT_DEFINE("PowercastEnergyHarvesterHelper");

// Constructor with default settings
PowercastEnergyHarvesterHelper::PowercastEnergyHarvesterHelper()
      : m_defaultTemplate(MIXED_DEPLOYMENT)
      , m_assignmentMethod(SEQUENTIAL)
      , m_nodesDeployed(0)
      , m_configurationLoaded(false)
      , m_sequentialCounter(0)
{
    NS_LOG_FUNCTION(this);

    // Configure factory for PowercastEnergyHarvester creation
    m_factory.SetTypeId("ns3::energy::PowercastEnergyHarvester");

    // Apply default template configuration
    ApplyTemplate(m_defaultTemplate);

    NS_LOG_INFO("PowercastEnergyHarvesterHelper initialized with MIXED_DEPLOYMENT template");
}

// Load configuration from text file
bool
PowercastEnergyHarvesterHelper::LoadConfigurationFromFile(const std::string& filename)
{
    NS_LOG_FUNCTION(this << filename);

    std::ifstream configFile(filename);
    if (!configFile.is_open())
    {
        NS_LOG_ERROR("Failed to open configuration file: " << filename);
        return false;
    }

    // Clear existing configurations
    m_nodeConfigurations.clear();
    m_configurationFile = filename;

    std::string line;
    uint32_t lineNumber = 0;
    uint32_t nodesConfigured = 0;

    while (std::getline(configFile, line))
    {
        lineNumber++;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        // Parse line: NodeID CapacitorClass VoltageClass
        std::istringstream iss(line);
        uint32_t nodeId;
        std::string capClassStr, voltClassStr;

        if (!(iss >> nodeId >> capClassStr >> voltClassStr))
        {
            NS_LOG_WARN("Invalid line " << lineNumber << " in " << filename << ": " << line);
            continue;
        }

        // Parse capacitor and voltage classes
        PowercastEnergyHarvester::CapacitorClass capClass = ParseCapacitorClass(capClassStr);
        PowercastEnergyHarvester::VoltageClass voltClass = ParseVoltageClass(voltClassStr);

        // Store configuration
        m_nodeConfigurations[nodeId] = NodeConfig(capClass, voltClass);
        nodesConfigured++;

        NS_LOG_DEBUG("Configured Node " << nodeId << ": Cap=" << capClassStr
                                        << ", Volt=" << voltClassStr);
    }

    configFile.close();

    if (nodesConfigured > 0)
    {
        m_configurationLoaded = true;
        m_assignmentMethod = FILE_BASED;

        NS_LOG_INFO("Successfully loaded " << nodesConfigured << " node configurations from "
                                           << filename);
        return true;
    }
    else
    {
        NS_LOG_ERROR("No valid configurations found in " << filename);
        return false;
    }
}

// Set default deployment template
void
PowercastEnergyHarvesterHelper::SetDefaultTemplate(DeploymentTemplate templateType)
{
    NS_LOG_FUNCTION(this << templateType);

    m_defaultTemplate = templateType;
    ApplyTemplate(templateType);

    NS_LOG_INFO("Default template set to: " << templateType);
}

// Apply deployment template configuration
void
PowercastEnergyHarvesterHelper::ApplyTemplate(DeploymentTemplate templateType)
{
    NS_LOG_FUNCTION(this << templateType);

    // Clear existing template configurations
    m_templateCapClasses.clear();
    m_templateVoltClasses.clear();

    switch (templateType)
    {
    case UNIFORM_A1:
        // All devices: CLASS_A capacitor, 1.2V thresholds (value from config file)
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_A);
        m_templateVoltClasses.push_back(PowercastEnergyHarvester::CLASS_1);
        NS_LOG_INFO("Applied UNIFORM_A1 template: All CLASS_A/CLASS_1");
        break;

    case UNIFORM_B2:
        // All devices: CLASS_B capacitor, 0.9V thresholds (value from config file)
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_B);
        m_templateVoltClasses.push_back(PowercastEnergyHarvester::CLASS_2);
        NS_LOG_INFO("Applied UNIFORM_B2 template: All CLASS_B/CLASS_2");
        break;

    case MIXED_DEPLOYMENT:
        // Mixed configuration for diversity
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_A);
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_B);
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_C);
        m_templateVoltClasses.push_back(PowercastEnergyHarvester::CLASS_1);
        m_templateVoltClasses.push_back(PowercastEnergyHarvester::CLASS_2);
        m_templateVoltClasses.push_back(PowercastEnergyHarvester::CLASS_3);
        NS_LOG_INFO("Applied MIXED_DEPLOYMENT template: All classes available");
        break;

    case HIGH_CAPACITY:
        // Large capacitors only (CLASS_A and CLASS_C from config file)
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_A);
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_C);
        m_templateVoltClasses.push_back(PowercastEnergyHarvester::CLASS_1);
        m_templateVoltClasses.push_back(PowercastEnergyHarvester::CLASS_2);
        NS_LOG_INFO("Applied HIGH_CAPACITY template: CLASS_A and CLASS_C capacitors");
        break;

    case LOW_POWER:
        // All devices use 0.7V thresholds for low-power operation
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_A);
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_B);
        m_templateCapClasses.push_back(PowercastEnergyHarvester::CLASS_C);
        m_templateVoltClasses.push_back(PowercastEnergyHarvester::CLASS_3); // Only 0.7V
        NS_LOG_INFO("Applied LOW_POWER template: All capacitors with CLASS_3 voltage");
        break;

    default:
        NS_LOG_WARN("Unknown template type, using MIXED_DEPLOYMENT");
        ApplyTemplate(MIXED_DEPLOYMENT);
        break;
    }
}

// Set assignment method
void
PowercastEnergyHarvesterHelper::SetAssignmentMethod(AssignmentMethod method)
{
    NS_LOG_FUNCTION(this << method);

    m_assignmentMethod = method;
    m_sequentialCounter = 0; // Reset sequential counter

    NS_LOG_INFO("Assignment method set to: " << method);
}

// Configure specific node manually
void
PowercastEnergyHarvesterHelper::ConfigureNode(uint32_t nodeId,
                                              PowercastEnergyHarvester::CapacitorClass capClass,
                                              PowercastEnergyHarvester::VoltageClass voltClass)
{
    NS_LOG_FUNCTION(this << nodeId << capClass << voltClass);

    m_nodeConfigurations[nodeId] = NodeConfig(capClass, voltClass);

    NS_LOG_INFO("Node " << nodeId << " manually configured: Cap=" << capClass
                        << ", Volt=" << voltClass);
}

// Set attribute for all harvesters (simplified implementation)
void
PowercastEnergyHarvesterHelper::Set(std::string name, const AttributeValue& v)
{
    NS_LOG_FUNCTION(this << name);

    m_factory.Set(name, v);

    NS_LOG_INFO("Attribute set: " << name);
}

// Install harvester on energy source
Ptr<EnergyHarvester>
PowercastEnergyHarvesterHelper::DoInstall(Ptr<EnergySource> source) const
{
    NS_LOG_FUNCTION(this << source);

    if (source == nullptr)
    {
        NS_LOG_ERROR("Energy source is null");
        return nullptr;
    }

    // Get node ID from energy source
    uint32_t nodeId = 0;
    Ptr<Node> node = source->GetNode();
    if (node != nullptr)
    {
        nodeId = node->GetId();
    }

    // Get configuration for this node
    NodeConfig config = GetNodeConfiguration(nodeId);

    // Create harvester instance
    Ptr<EnergyHarvester> baseHarvester = m_factory.Create<EnergyHarvester>();
    Ptr<PowercastEnergyHarvester> harvester = DynamicCast<PowercastEnergyHarvester>(baseHarvester);

    if (harvester == nullptr)
    {
        NS_LOG_ERROR("Failed to create PowercastEnergyHarvester");
        return nullptr;
    }

    // Configure harvester with determined classes
    harvester->Configure(config.capacitorClass, config.voltageClass);

    // Establish energy source connection
    harvester->SetEnergySource(source);
    source->ConnectEnergyHarvester(harvester);

    // Update statistics
    m_nodesDeployed++;
    std::string configKey =
        CapacitorClassToString(config.capacitorClass) + VoltageClassToString(config.voltageClass);
    m_classStatistics[configKey]++;

    NS_LOG_INFO("Deployed harvester on node " << nodeId << ": " << configKey);

    return harvester;
}

// Batch installation
EnergyHarvesterContainer
PowercastEnergyHarvesterHelper::Install(EnergySourceContainer sources) const
{
    NS_LOG_FUNCTION(this << sources.GetN());

    EnergyHarvesterContainer harvesters;

    for (uint32_t i = 0; i < sources.GetN(); ++i)
    {
        Ptr<EnergySource> source = sources.Get(i);
        Ptr<EnergyHarvester> harvester = DoInstall(source);

        if (harvester != nullptr)
        {
            harvesters.Add(harvester);
        }
    }

    NS_LOG_INFO("Batch installation completed: " << harvesters.GetN() << " harvesters deployed");

    return harvesters;
}

// Get configuration for node (with fallback methods)
PowercastEnergyHarvesterHelper::NodeConfig
PowercastEnergyHarvesterHelper::GetNodeConfiguration(uint32_t nodeId) const
{
    NS_LOG_FUNCTION(this << nodeId);

    // Check if node has specific configuration
    auto configIt = m_nodeConfigurations.find(nodeId);
    if (configIt != m_nodeConfigurations.end())
    {
        NS_LOG_DEBUG("Using specific configuration for node " << nodeId);
        return configIt->second;
    }

    // Use fallback method based on assignment method
    switch (m_assignmentMethod)
    {
    case SEQUENTIAL:
        return GetSequentialConfiguration(nodeId);
    case RANDOM:
        return GetRandomConfiguration();
    case FILE_BASED:
        NS_LOG_WARN("Node " << nodeId << " not found in file, using sequential fallback");
        return GetSequentialConfiguration(nodeId);
    case MANUAL:
        NS_LOG_WARN("Node " << nodeId << " not manually configured, using sequential fallback");
        return GetSequentialConfiguration(nodeId);
    default:
        return GetSequentialConfiguration(nodeId);
    }
}

// Sequential configuration assignment
PowercastEnergyHarvesterHelper::NodeConfig
PowercastEnergyHarvesterHelper::GetSequentialConfiguration(uint32_t nodeId) const
{
    NS_LOG_FUNCTION(this << nodeId);

    if (m_templateCapClasses.empty() || m_templateVoltClasses.empty())
    {
        NS_LOG_WARN("No template classes available, using defaults");
        return NodeConfig(PowercastEnergyHarvester::CLASS_A, PowercastEnergyHarvester::CLASS_1);
    }

    // Round-robin assignment
    PowercastEnergyHarvester::CapacitorClass capClass =
        m_templateCapClasses[m_sequentialCounter % m_templateCapClasses.size()];
    PowercastEnergyHarvester::VoltageClass voltClass =
        m_templateVoltClasses[m_sequentialCounter % m_templateVoltClasses.size()];

    m_sequentialCounter++;

    return NodeConfig(capClass, voltClass);
}

// Random configuration assignment
PowercastEnergyHarvesterHelper::NodeConfig
PowercastEnergyHarvesterHelper::GetRandomConfiguration() const
{
    NS_LOG_FUNCTION(this);

    if (m_templateCapClasses.empty() || m_templateVoltClasses.empty())
    {
        return NodeConfig(PowercastEnergyHarvester::CLASS_A, PowercastEnergyHarvester::CLASS_1);
    }

    // Use NS-3 random number generator
    Ptr<UniformRandomVariable> uniformRv = CreateObject<UniformRandomVariable>();

    uint32_t capIndex = uniformRv->GetInteger(0, m_templateCapClasses.size() - 1);
    uint32_t voltIndex = uniformRv->GetInteger(0, m_templateVoltClasses.size() - 1);

    return NodeConfig(m_templateCapClasses[capIndex], m_templateVoltClasses[voltIndex]);
}

// Parse capacitor class from string
PowercastEnergyHarvester::CapacitorClass
PowercastEnergyHarvesterHelper::ParseCapacitorClass(const std::string& classStr) const
{
    if (classStr == "A" || classStr == "a")
    {
        return PowercastEnergyHarvester::CLASS_A;
    }
    else if (classStr == "B" || classStr == "b")
    {
        return PowercastEnergyHarvester::CLASS_B;
    }
    else if (classStr == "C" || classStr == "c")
    {
        return PowercastEnergyHarvester::CLASS_C;
    }
    else
    {
        NS_LOG_WARN("Unknown capacitor class: " << classStr << ", using CLASS_A");
        return PowercastEnergyHarvester::CLASS_A;
    }
}

// Parse voltage class from string
PowercastEnergyHarvester::VoltageClass
PowercastEnergyHarvesterHelper::ParseVoltageClass(const std::string& classStr) const
{
    if (classStr == "1")
    {
        return PowercastEnergyHarvester::CLASS_1;
    }
    else if (classStr == "2")
    {
        return PowercastEnergyHarvester::CLASS_2;
    }
    else if (classStr == "3")
    {
        return PowercastEnergyHarvester::CLASS_3;
    }
    else
    {
        NS_LOG_WARN("Unknown voltage class: " << classStr << ", using CLASS_1");
        return PowercastEnergyHarvester::CLASS_1;
    }
}

// Convert capacitor class to string
std::string
PowercastEnergyHarvesterHelper::CapacitorClassToString(
    PowercastEnergyHarvester::CapacitorClass capClass) const
{
    switch (capClass)
    {
    case PowercastEnergyHarvester::CLASS_A:
        return "A";
    case PowercastEnergyHarvester::CLASS_B:
        return "B";
    case PowercastEnergyHarvester::CLASS_C:
        return "C";
    default:
        return "A";
    }
}

// Convert voltage class to string
std::string
PowercastEnergyHarvesterHelper::VoltageClassToString(
    PowercastEnergyHarvester::VoltageClass voltClass) const
{
    switch (voltClass)
    {
    case PowercastEnergyHarvester::CLASS_1:
        return "1";
    case PowercastEnergyHarvester::CLASS_2:
        return "2";
    case PowercastEnergyHarvester::CLASS_3:
        return "3";
    default:
        return "1";
    }
}

// Export configuration to file
bool
PowercastEnergyHarvesterHelper::ExportConfigurationToFile(const std::string& filename) const
{
    NS_LOG_FUNCTION(this << filename);

    std::ofstream configFile(filename);
    if (!configFile.is_open())
    {
        NS_LOG_ERROR("Failed to create configuration file: " << filename);
        return false;
    }

    // Write header
    configFile << "# PowerCast Energy Harvester Configuration" << std::endl;
    configFile << "# Format: NodeID CapacitorClass VoltageClass" << std::endl;
    configFile << "# CapacitorClass: A(500μF), B(2200μF), C(20000μF) - values from config file"
               << std::endl;
    configFile << "# VoltageClass: 1(1.2V), 2(0.9V), 3(0.7V)" << std::endl;
    configFile << "#" << std::endl;

    // Write configurations
    for (const auto& config : m_nodeConfigurations)
    {
        configFile << config.first << " " << CapacitorClassToString(config.second.capacitorClass)
                   << " " << VoltageClassToString(config.second.voltageClass) << std::endl;
    }

    configFile.close();

    NS_LOG_INFO("Configuration exported to: " << filename);
    return true;
}

// Get configuration statistics
std::string
PowercastEnergyHarvesterHelper::GetConfigurationStatistics() const
{
    std::ostringstream stats;

    stats << "PowerCast Energy Harvester Deployment Statistics:" << std::endl;
    stats << "  Total nodes deployed: " << m_nodesDeployed << std::endl;
    stats << "  Assignment method: " << m_assignmentMethod << std::endl;

    if (m_configurationLoaded)
    {
        stats << "  Configuration file: " << m_configurationFile << std::endl;
        stats << "  Specific configurations: " << m_nodeConfigurations.size() << std::endl;
    }

    stats << "  Class distribution:" << std::endl;
    for (const auto& classStat : m_classStatistics)
    {
        stats << "    " << classStat.first << ": " << classStat.second << " nodes" << std::endl;
    }

    return stats.str();
}

// Validate configuration
bool
PowercastEnergyHarvesterHelper::ValidateConfiguration() const
{
    NS_LOG_FUNCTION(this);

    if (m_assignmentMethod == FILE_BASED && !m_configurationLoaded)
    {
        NS_LOG_ERROR("FILE_BASED assignment method requires loaded configuration");
        return false;
    }

    if (m_templateCapClasses.empty() || m_templateVoltClasses.empty())
    {
        NS_LOG_ERROR("No template classes configured");
        return false;
    }

    NS_LOG_INFO("Configuration validation passed");
    return true;
}

// Clear configuration
void
PowercastEnergyHarvesterHelper::ClearConfiguration()
{
    NS_LOG_FUNCTION(this);

    m_nodeConfigurations.clear();
    m_classStatistics.clear();
    m_configurationLoaded = false;
    m_configurationFile.clear();
    m_nodesDeployed = 0;
    m_sequentialCounter = 0;

    NS_LOG_INFO("Configuration cleared");
}

} // namespace energy
} // namespace ns3