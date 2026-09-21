// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file ph-deployment-helper.h
 * @brief PowercastEnergyHarvesterHelper: install PowerCast harvesters on many nodes from a config file or template
 *
 * SetupEnergyHarvesting() in twt-simulation-config.cc installs the REHD harvesters through it, overriding each node's classes with ConfigureNode() from REHD_TYPE_TEMPLATES.
 *
 * - Reads "NodeID CapacitorClass VoltageClass" lines (see ph-harvester-config.txt)
 * - Deployment templates: uniform, mixed, high-capacity, low-power
 * - Assignment methods for unlisted nodes: sequential, random, file-based, manual
 * - Per-node overrides, configuration validation, statistics and export
 */

#ifndef EHN_POWERCAST_ENERGY_HARVESTER_HELPER_H
#define EHN_POWERCAST_ENERGY_HARVESTER_HELPER_H

#include "ph-harvester-hardware.h" // PowercastEnergyHarvester class

#include "ns3/attribute.h"               // Attribute system
#include "ns3/energy-harvester-helper.h" // Base helper class
#include "ns3/object-factory.h"          // NS-3 factory pattern
#include "ns3/ptr.h"                     // Smart pointer system

#include <map>    // STL map for configurations
#include <string> // String handling
#include <vector> // STL vector for storage

namespace ns3
{
namespace energy
{

/**
 * @brief Helper for file-based PowercastEnergyHarvester deployment
 *
 * Installs PowercastEnergyHarvester instances with a file-based configuration system. Enables reading capacitor and
 * voltage class specifications from external configuration files for automated
 * deployment scenarios where node characteristics are predefined.
 *
 * Supported Configuration Methods:
 * - File-based: Read configurations from text files
 * - Template-based: Use predefined P21XXCSR-EVB templates
 * - Sequential: Assign classes in round-robin fashion
 * - Random: Random assignment from available classes
 * - Per-node: Individual node customization
 *
 * Configuration File Format (harvester_config.txt):
 * # Node configurations: NodeID CapacitorClass VoltageClass
 * 0 A 1
 * 1 B 2
 * 2 C 1
 * 3 A 3
 * ...
 *
 * Usage Examples:
 * @code
 * // File-based configuration
 * PowercastEnergyHarvesterHelper helper;
 * helper.LoadConfigurationFromFile("harvester_config.txt");
 * EnergyHarvesterContainer harvesters = helper.Install(energySources);
 *
 * // Template-based deployment
 * helper.SetDefaultTemplate(PowercastEnergyHarvesterHelper::MIXED_DEPLOYMENT);
 * EnergyHarvesterContainer harvesters = helper.Install(energySources);
 * @endcode
 */
class PowercastEnergyHarvesterHelper : public EnergyHarvesterHelper
{
  public:
    /**
     * @brief Deployment templates for common scenarios
     *
     * Predefined templates based on P21XXCSR-EVB configurations:
     * - UNIFORM_A1: All devices use CLASS_A capacitor (500μF), 1.2V thresholds
     * - UNIFORM_B2: All devices use CLASS_B capacitor (2200μF), 0.9V thresholds
     * - MIXED_DEPLOYMENT: Mixed capacitor and voltage classes
     * - HIGH_CAPACITY: All devices use large capacitors (A or C)
     * - LOW_POWER: All devices use 0.7V thresholds for low-power operation
     *
     * Note: Actual capacitor values are loaded from config file.
     */
    enum DeploymentTemplate
    {
        UNIFORM_A1,       ///< All CLASS_A capacitor, CLASS_1 voltage
        UNIFORM_B2,       ///< All CLASS_B capacitor, CLASS_2 voltage
        MIXED_DEPLOYMENT, ///< Mixed classes for diversity
        HIGH_CAPACITY,    ///< Large capacitors only (A and C)
        LOW_POWER         ///< All CLASS_3 voltage for low-power
    };

    /**
     * @brief Configuration assignment method
     *
     * Defines how configurations are assigned to nodes:
     * - SEQUENTIAL: Round-robin assignment through available classes
     * - RANDOM: Random selection from available classes
     * - FILE_BASED: Read from configuration file
     * - MANUAL: Manual assignment per node
     */
    enum AssignmentMethod
    {
        SEQUENTIAL, ///< Round-robin assignment
        RANDOM,     ///< Random assignment
        FILE_BASED, ///< File-based assignment
        MANUAL      ///< Manual per-node assignment
    };

    /**
     * @brief Default constructor
     *
     * Initializes helper with MIXED_DEPLOYMENT template and
     * SEQUENTIAL assignment method as defaults.
     */
    PowercastEnergyHarvesterHelper();

    /**
     * @brief Virtual destructor
     */
    virtual ~PowercastEnergyHarvesterHelper() = default;

    // --- CONFIGURATION METHODS ---

    /**
     * @brief Load configuration from file
     *
     * Reads node configurations from text file with format:
     * NodeID CapacitorClass VoltageClass
     *
     * CapacitorClass: A (500μF), B (2200μF), C (20000μF)
     * VoltageClass: 1 (1.2V), 2 (0.9V), 3 (0.7V)
     *
     * Note: Actual values loaded from config file header.
     *
     * @param filename Configuration file path
     * @return true if successful, false otherwise
     */
    bool LoadConfigurationFromFile(const std::string& filename);

    /**
     * @brief Set default deployment template
     * @param template Deployment template to use
     */
    void SetDefaultTemplate(DeploymentTemplate templateType);

    /**
     * @brief Set assignment method for unknown configurations
     * @param method Assignment method to use
     */
    void SetAssignmentMethod(AssignmentMethod method);

    /**
     * @brief Configure specific node
     * @param nodeId Node ID to configure
     * @param capClass Capacitor class
     * @param voltClass Voltage class
     */
    void ConfigureNode(uint32_t nodeId,
                       PowercastEnergyHarvester::CapacitorClass capClass,
                       PowercastEnergyHarvester::VoltageClass voltClass);

    /**
     * @brief Set attribute for all harvesters
     * @param name Attribute name
     * @param v Attribute value
     */
    void Set(std::string name, const AttributeValue& v);

    // --- INSTALLATION METHODS ---

    /**
     * @brief Install harvester on energy source
     *
     * Creates and configures harvester based on node ID and
     * loaded configuration. Uses fallback methods for unknown nodes.
     *
     * @param source Energy source for harvester connection
     * @return Smart pointer to created harvester
     */
    virtual Ptr<EnergyHarvester> DoInstall(Ptr<EnergySource> source) const override;

    /**
     * @brief Batch installation on energy source container
     * @param sources Container of energy sources
     * @return Container of created harvesters
     */
    EnergyHarvesterContainer Install(EnergySourceContainer sources) const;

    // --- UTILITY METHODS ---

    /**
     * @brief Export current configuration to file
     * @param filename Output file path
     * @return true if successful, false otherwise
     */
    bool ExportConfigurationToFile(const std::string& filename) const;

    /**
     * @brief Get configuration statistics
     * @return String with deployment statistics
     */
    std::string GetConfigurationStatistics() const;

    /**
     * @brief Validate loaded configuration
     * @return true if configuration is valid, false otherwise
     */
    bool ValidateConfiguration() const;

    /**
     * @brief Clear all configurations
     */
    void ClearConfiguration();

    /**
     * @brief Convert capacitor class to string
     * @param capClass Capacitor class
     * @return String representation
     */
    std::string CapacitorClassToString(PowercastEnergyHarvester::CapacitorClass capClass) const;

    /**
     * @brief Convert voltage class to string
     * @param voltClass Voltage class
     * @return String representation
     */
    std::string VoltageClassToString(PowercastEnergyHarvester::VoltageClass voltClass) const;

  private:
    /**
     * @brief Node configuration structure
     */
    struct NodeConfig
    {
        PowercastEnergyHarvester::CapacitorClass capacitorClass;
        PowercastEnergyHarvester::VoltageClass voltageClass;
        bool isConfigured;

        NodeConfig()
              : capacitorClass(PowercastEnergyHarvester::CLASS_A)
              , voltageClass(PowercastEnergyHarvester::CLASS_1)
              , isConfigured(false)
        {
        }

        NodeConfig(PowercastEnergyHarvester::CapacitorClass cap,
                   PowercastEnergyHarvester::VoltageClass volt)
              : capacitorClass(cap)
              , voltageClass(volt)
              , isConfigured(true)
        {
        }
    };

    /**
     * @brief Apply deployment template
     * @param templateType Template to apply
     */
    void ApplyTemplate(DeploymentTemplate templateType);

    /**
     * @brief Get configuration for node using fallback methods
     * @param nodeId Node ID
     * @return Node configuration
     */
    NodeConfig GetNodeConfiguration(uint32_t nodeId) const;

    /**
     * @brief Get sequential configuration
     * @param nodeId Node ID for sequential assignment
     * @return Node configuration
     */
    NodeConfig GetSequentialConfiguration(uint32_t nodeId) const;

    /**
     * @brief Get random configuration
     * @return Random node configuration
     */
    NodeConfig GetRandomConfiguration() const;

    /**
     * @brief Parse capacitor class from string
     * @param classStr String representation (A, B, C)
     * @return Capacitor class
     */
    PowercastEnergyHarvester::CapacitorClass ParseCapacitorClass(const std::string& classStr) const;

    /**
     * @brief Parse voltage class from string
     * @param classStr String representation (1, 2, 3)
     * @return Voltage class
     */
    PowercastEnergyHarvester::VoltageClass ParseVoltageClass(const std::string& classStr) const;

    // --- INTERNAL STATE ---
    ObjectFactory m_factory; ///< NS-3 object factory

    // Configuration management
    std::map<uint32_t, NodeConfig> m_nodeConfigurations; ///< Per-node configurations
    DeploymentTemplate m_defaultTemplate;                ///< Default deployment template
    AssignmentMethod m_assignmentMethod;                 ///< Assignment method for unknown nodes

    // Template configurations
    std::vector<PowercastEnergyHarvester::CapacitorClass>
        m_templateCapClasses; ///< Template capacitor classes
    std::vector<PowercastEnergyHarvester::VoltageClass>
        m_templateVoltClasses; ///< Template voltage classes

    // Statistics
    mutable uint32_t m_nodesDeployed;                          ///< Number of nodes deployed
    mutable std::map<std::string, uint32_t> m_classStatistics; ///< Class usage statistics

    // Configuration state
    bool m_configurationLoaded;           ///< Configuration loaded flag
    std::string m_configurationFile;      ///< Source configuration file
    mutable uint32_t m_sequentialCounter; ///< Counter for sequential assignment
};

} // namespace energy
} // namespace ns3

#endif // EHN_POWERCAST_ENERGY_HARVESTER_HELPER_H