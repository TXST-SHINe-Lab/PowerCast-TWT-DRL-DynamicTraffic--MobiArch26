// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file ph-harvester-hardware.cc
 * @brief PowerCast P21XXCSR-EVB RF energy harvester implementation
 *
 * - Band-6 efficiency curve digitized from the datasheet, linear interpolation between points
 * - 85% boost-converter efficiency on harvest, 75% PA / RF-chain efficiency on TX
 * - Strict two-threshold capacitor (DEEP_RATIO = SHOOT_RATIO = 1.0): Vcap stays in [Vmin, Vmax]
 * - An over-draw that slips past CanSustainTransmission() is clamped at the floor with a warning, not a fatal error
 * - NS_FATAL_ERROR if the harvester is used before Configure(), or with a capacitor class whose value was never loaded
 */

#include "ph-harvester-hardware.h"

#include "ns3/boolean.h"   // Boolean attribute support
#include "ns3/double.h"    // Attribute system for double-precision parameters
#include "ns3/log.h"       // Debugging and tracing infrastructure
#include "ns3/simulator.h" // Simulation time management
#include "ns3/uinteger.h"  // Integer attribute support

#include <cmath> // Mathematical functions for energy calculations

namespace ns3
{
namespace energy
{

// Enable logging for this component
NS_LOG_COMPONENT_DEFINE("PowercastEnergyHarvester");

// Register with NS-3 object system
NS_OBJECT_ENSURE_REGISTERED(PowercastEnergyHarvester);

// Fixed parameter definitions based on P21XXCSR-EVB specifications
const double PowercastEnergyHarvester::BOOST_EFFICIENCY =
    0.85; // 85% boost converter efficiency (P21XXCSR-EVB datasheet)
const double PowercastEnergyHarvester::PA_EFFICIENCY =
    0.75; // 75% PA / RF-chain efficiency (REHD TX; datasheet-informed sweet spot for an
          // ultra-low-power 2.4 GHz radio — models the PA/RF chain, not full baseband)
const double PowercastEnergyHarvester::OPERATING_FREQUENCY = 2450e6; // 2450 MHz center frequency
const double PowercastEnergyHarvester::DEEP_RATIO =
    1.0; // TX gate floor = Vmin (Vcap may not go below Vmin) — strict two-threshold model
const double PowercastEnergyHarvester::SHOOT_RATIO =
    1.0; // Harvest stops at Vmax (no overshoot) — strict two-threshold model

// Static capacitor value storage (MUST be loaded from config files - NO defaults)
double PowercastEnergyHarvester::s_capA_F = 0.0; // MUST be set from config file
double PowercastEnergyHarvester::s_capB_F = 0.0; // MUST be set from config file
double PowercastEnergyHarvester::s_capC_F = 0.0; // MUST be set from config file

// TypeId configuration with realistic 2.4GHz parameters
TypeId
PowercastEnergyHarvester::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::energy::PowercastEnergyHarvester")
            .SetParent<EnergyHarvester>()
            .SetGroupName("Energy")
            .AddConstructor<PowercastEnergyHarvester>()

            // --- CORE CONFIGURATION ATTRIBUTES ---

            .AddAttribute(
                "CapacitorClass",
                "Capacitor class: 0=Class A, 1=Class B, 2=Class C (values from config file)",
                UintegerValue(0),
                MakeUintegerAccessor(&PowercastEnergyHarvester::SetCapacitorClassAttribute,
                                     &PowercastEnergyHarvester::GetCapacitorClassAttribute),
                MakeUintegerChecker<uint32_t>(0, 2))

            .AddAttribute("VoltageClass",
                          "Voltage class: 0=1.2V(1), 1=0.9V(2), 2=0.7V(3)",
                          UintegerValue(0),
                          MakeUintegerAccessor(&PowercastEnergyHarvester::SetVoltageClassAttribute,
                                               &PowercastEnergyHarvester::GetVoltageClassAttribute),
                          MakeUintegerChecker<uint32_t>(0, 2))

            // --- TRACE SOURCES ---

            .AddTraceSource("Vcap",
                            "Capacitor voltage for monitoring",
                            MakeTraceSourceAccessor(&PowercastEnergyHarvester::m_vcapTrace),
                            "ns3::TracedValueCallback::Double")

            .AddTraceSource(
                "AvailableEnergy",
                "Available energy for monitoring",
                MakeTraceSourceAccessor(&PowercastEnergyHarvester::m_availableEnergyTrace),
                "ns3::TracedValueCallback::Double")

            .AddTraceSource("HarvestingEfficiency",
                            "Instantaneous harvesting efficiency",
                            MakeTraceSourceAccessor(&PowercastEnergyHarvester::m_efficiencyTrace),
                            "ns3::TracedValueCallback::Double");

    return tid;
}

// Constructor: empty capacitor, placeholder classes until Configure()
PowercastEnergyHarvester::PowercastEnergyHarvester()
      : m_vcap(0.0)
      , // Will be set after explicit configuration
      m_cStorage(0.0)
      , // Will be set after explicit configuration
      m_vmax(0.0)
      , // Will be set after explicit configuration
      m_vmin(0.0)
      , // Will be set after explicit configuration
      m_capacitorClass(static_cast<CapacitorClass>(-1))
      , // Invalid until configured
      m_voltageClass(static_cast<VoltageClass>(-1))
      , // Invalid until configured
      m_totalHarvestedEnergy(0.0)
      , // Zero cumulative harvested energy
      m_totalConsumedEnergy(0.0)
      , // Zero cumulative consumed energy
      m_onOffCycles(0)
      , // Zero power cycling events
      m_totalActiveTime_us(0)
      , // QoEH S_r accumulator (µs above vmin)
      m_lastUpdateTime(ns3::Seconds(0))
      , m_lastActiveState(false)
{
    NS_LOG_FUNCTION(this);

    // NO DEFAULT INITIALIZATION - Must be explicitly configured
    // Use Configure(capClass, voltClass) or SetCapacitorClass() + SetVoltageClass()
    NS_LOG_INFO(
        "PowerCast harvester created - MUST be explicitly configured before use (no defaults)");

    // Initialize trace variables
    m_vcapTrace = m_vcap;

    // Nominal/sunk/available energy split
    m_nominalEnergy = 0.5 * m_cStorage * m_vcap * m_vcap; // Total stored energy
    m_sunkEnergy = 0.5 * m_cStorage * (m_vmin * DEEP_RATIO) *
                   (m_vmin * DEEP_RATIO); // Unusable deep discharge energy
    m_availableEnergy =
        m_nominalEnergy - m_sunkEnergy; // Usable energy for operations above deep discharge level

    // Initialize NS-3 trace sources for monitoring
    m_vcapTrace = m_vcap;
    m_availableEnergyTrace = m_availableEnergy;
    m_efficiencyTrace = 0.0;

    NS_LOG_INFO("PowerCast energy harvester initialized with:");
    NS_LOG_INFO("  Operating Frequency: " << (OPERATING_FREQUENCY / 1e6) << " MHz");
    NS_LOG_INFO("  Capacitor Class: " << m_capacitorClass << " (" << (m_cStorage * 1e6) << " μF)");
    NS_LOG_INFO("  Voltage Class: " << m_voltageClass << " (Vmax=" << m_vmax << "V, Vmin=" << m_vmin
                                    << "V)");
    NS_LOG_INFO("  Boost Efficiency: " << (BOOST_EFFICIENCY * 100) << "%");
    NS_LOG_INFO("  PA Efficiency: " << (PA_EFFICIENCY * 100) << "%");
    NS_LOG_INFO("  Deep Discharge Protection: " << (DEEP_RATIO * 100) << "% of Vmin");
    NS_LOG_INFO("  Overshoot Protection: " << (SHOOT_RATIO * 100) << "% of Vmax");
    NS_LOG_INFO("  Initial Available Energy: " << m_availableEnergy << " J");
    NS_LOG_INFO("  Input Power Range: -12 to +15 dBm (datasheet efficiency curve)");
    NS_LOG_INFO("  TX gate: no hysteresis, TX allowed while Vcap stays >= DEEP_RATIO*Vmin ("
                << m_vmin * DEEP_RATIO << "V)");
}

// NS-3 lifecycle cleanup
void
PowercastEnergyHarvester::DoDispose()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("PowercastEnergyHarvester disposed safely");
}

// Event-driven QoEH active-time integration.
// Adds the elapsed interval since the previous vcap-changing event to m_totalActiveTime_us if vcap was above vmin during that interval.
// Called at the START of SetHarvestedEnergy / SetConsumedEnergy so the "before" state is what gets accounted for.
void
PowercastEnergyHarvester::UpdateActiveTime()
{
    ns3::Time now = ns3::Simulator::Now();
    if (m_lastActiveState)
    {
        m_totalActiveTime_us += static_cast<uint64_t>((now - m_lastUpdateTime).GetMicroSeconds());
    }
    m_lastActiveState = (m_vcap > m_vmin); // strict >: exactly Vmin counts as inactive
    m_lastUpdateTime = now;
}

// RF energy harvesting through the Band-6 efficiency curve
void
PowercastEnergyHarvester::SetHarvestedEnergy(double rxPowerDbm, Time duration)
{
    NS_LOG_FUNCTION(this << rxPowerDbm << duration.GetSeconds());
    UpdateActiveTime();

    // Validate configuration before use
    if (m_cStorage <= 0.0 || m_vmax <= 0.0 || m_vmin <= 0.0)
    {
        NS_FATAL_ERROR("PowerCast harvester not properly configured! Must call Configure(capClass, "
                       "voltClass) or set classes explicitly. Current values: C="
                       << m_cStorage << "F, Vmax=" << m_vmax << "V, Vmin=" << m_vmin << "V");
    }

    // Apply P21XXCSR-EVB input power specifications with proper clamping
    double clampedPowerDbm = rxPowerDbm;

    if (rxPowerDbm < -12.0)
    {
        NS_LOG_DEBUG("Input power below -12 dBm sensitivity floor: " << rxPowerDbm << " dBm");
        // Continue processing - efficiency curve returns 0 below the -12 dBm floor
    }

    if (rxPowerDbm > 15.0)
    {
        NS_LOG_DEBUG("Input power exceeds P21XXCSR-EVB maximum rating: "
                     << rxPowerDbm << " dBm, clamping to +15 dBm");
        clampedPowerDbm = 15.0; // Protect hardware from damage
    }

    // Convert dBm to watts for energy calculations: P(W) = 10^(P(dBm)/10) / 1000
    double powerMw = std::pow(10.0, rxPowerDbm / 10.0);

    // Calculate Band 6 harvesting efficiency using realistic curve
    double efficiency = CalculateRealistic24GHzEfficiency(clampedPowerDbm);

    // Calculate harvested energy: E = P × t × η_harvesting × η_boost
    double energyJ = (powerMw / 1000.0) * duration.GetSeconds() * efficiency * BOOST_EFFICIENCY;

    // Update cumulative energy accounting
    m_totalHarvestedEnergy += energyJ;
    m_nominalEnergy += energyJ;

    // Calculate theoretical voltage from energy: V = √(2E/C)
    double theoreticalVcap = std::sqrt(2 * m_nominalEnergy / m_cStorage);

    // Cap at SHOOT_RATIO*Vmax (= Vmax): harvest stops at a full capacitor
    m_vcap = std::min(theoreticalVcap, m_vmax * SHOOT_RATIO);

    // Recalculate energies based on actual clamped voltage
    m_nominalEnergy = 0.5 * m_cStorage * m_vcap * m_vcap;
    m_availableEnergy = m_nominalEnergy - m_sunkEnergy;

    // Update NS-3 trace sources for monitoring and visualization
    m_vcapTrace = m_vcap;
    m_availableEnergyTrace = m_availableEnergy;
    m_efficiencyTrace = efficiency;

    // Debug trace of the harvest step
    NS_LOG_DEBUG("RF energy harvesting operation completed:");
    NS_LOG_DEBUG("  Original RX Power: " << rxPowerDbm << " dBm");
    NS_LOG_DEBUG("  Clamped RX Power: " << clampedPowerDbm << " dBm (" << powerMw << " mW)");
    NS_LOG_DEBUG("  Duration: " << duration.GetSeconds() << " s");
    NS_LOG_DEBUG("  Band 6 Efficiency: " << (efficiency * 100) << "%");
    NS_LOG_DEBUG("  Energy Harvested: " << energyJ << " J");
    NS_LOG_DEBUG("  Capacitor Voltage: " << m_vcap << " V");
    NS_LOG_DEBUG("  Available Energy: " << m_availableEnergy << " J");
    NS_LOG_DEBUG("  Nominal Energy: " << m_nominalEnergy << " J");
}

// TX energy debit (PA / RF-chain efficiency included)
void
PowercastEnergyHarvester::SetConsumedEnergy(double txPowerDbm, Time duration)
{
    UpdateActiveTime();
    NS_LOG_FUNCTION(this << txPowerDbm << duration.GetSeconds());

    // Validate configuration before use
    if (m_cStorage <= 0.0 || m_vmax <= 0.0 || m_vmin <= 0.0)
    {
        NS_FATAL_ERROR("PowerCast harvester not properly configured! Must call Configure(capClass, "
                       "voltClass) or set classes explicitly. Current values: C="
                       << m_cStorage << "F, Vmax=" << m_vmax << "V, Vmin=" << m_vmin << "V");
    }

    // Convert transmission power to watts: P(W) = 10^(P(dBm)/10) / 1000
    double powerMw = std::pow(10.0, txPowerDbm / 10.0);

    // Calculate RF energy actually radiated (output power)
    double energyTx = (powerMw / 1000.0) * duration.GetSeconds();

    // Account for power amplifier efficiency (DC energy required)
    double energyDC = energyTx / PA_EFFICIENCY;

    // Update cumulative consumption tracking
    m_totalConsumedEnergy += energyDC;

    // Over-draw guard: the energy audit (CanSustainTransmission) should prevent a TX that can't be paid for, but if one slips through (e.g. an aggregate larger than the audit's representative packet), clamp gracefully at the hard floor instead of aborting the run.
    // Floor = DEEP_RATIO*Vmin (= Vmin).
    if (energyDC > m_availableEnergy)
    {
        NS_LOG_WARN("TX over-draw clamped: needed " << energyDC << "J, had " << m_availableEnergy
                                                    << "J — pinning at the DEEP_RATIO*Vmin floor");
        m_availableEnergy = 0.0;
        m_nominalEnergy = m_sunkEnergy;
    }
    else
    {
        m_availableEnergy -= energyDC;
        m_nominalEnergy -= energyDC;
    }

    // Update capacitor voltage based on remaining energy: V = √(2E/C)
    if (m_nominalEnergy > 0.0)
    {
        m_vcap = std::sqrt(2 * m_nominalEnergy / m_cStorage);
    }
    else
    {
        m_vcap = 0.0; // Completely discharged state
    }

    // Update NS-3 trace sources for monitoring
    m_vcapTrace = m_vcap;
    m_availableEnergyTrace = m_availableEnergy;

    // Debug trace of the TX debit
    NS_LOG_DEBUG("Energy consumption operation completed:");
    NS_LOG_DEBUG("  TX Power: " << txPowerDbm << " dBm (" << powerMw << " mW)");
    NS_LOG_DEBUG("  Duration: " << duration.GetSeconds() << " s");
    NS_LOG_DEBUG("  PA Efficiency: " << (PA_EFFICIENCY * 100) << "%");
    NS_LOG_DEBUG("  RF Energy (radiated): " << energyTx << " J");
    NS_LOG_DEBUG("  DC Energy (consumed): " << energyDC << " J");
    NS_LOG_DEBUG("  Remaining Vcap: " << m_vcap << " V");
    NS_LOG_DEBUG("  Remaining Available Energy: " << m_availableEnergy << " J");
    NS_LOG_DEBUG("  Total Nominal Energy: " << m_nominalEnergy << " J");
}

// Small awake-state (IDLE/RX/CCA) housekeeping drain. Raw DC, no PA/RF model.
void
PowercastEnergyHarvester::ConsumeDcEnergy(double energyJ)
{
    UpdateActiveTime();
    if (energyJ <= 0.0)
    {
        return;
    }
    if (m_cStorage <= 0.0 || m_vmin <= 0.0)
    {
        return; // not configured yet — nothing to drain
    }

    m_totalConsumedEnergy += energyJ;

    // Floor at the deep-discharge level: the capacitor cannot give up energy below 0.5*C*(Vmin*DEEP_RATIO)^2.
    // Clamp gracefully (no fatal error — unlike the TX path this is a tiny continuous draw and must never abort the run).
    const double floorNominal = 0.5 * m_cStorage * (m_vmin * DEEP_RATIO) * (m_vmin * DEEP_RATIO);
    m_nominalEnergy = std::max(floorNominal, m_nominalEnergy - energyJ);
    m_availableEnergy = m_nominalEnergy - m_sunkEnergy;

    m_vcap = (m_nominalEnergy > 0.0) ? std::sqrt(2 * m_nominalEnergy / m_cStorage) : 0.0;
    m_vcapTrace = m_vcap;
    m_availableEnergyTrace = m_availableEnergy;
}

// P21XXCSR-EVB Band 6 realistic efficiency curve implementation
double
PowercastEnergyHarvester::CalculateRealistic24GHzEfficiency(double rxPowerDbm) const
{
    NS_LOG_FUNCTION(this << rxPowerDbm);

    // PowerCast P21XXCSR-EVB Band-6 (2450 MHz, 0.7 V) RF-to-DC efficiency, digitized from the datasheet "Powerharvester Efficiency vs. RFIN" curve: zero below the -12 dBm sensitivity floor, rising to a ~46% peak at +8 dBm.
    // Above +8 dBm we hold the peak, since any higher input can be attenuated to the optimum (but never boosted up).
    static const double kP[] = {-12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0,
                                -1.0,  0.0,   1.0,   2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0};
    static const double kEta[] = {0.000, 0.020, 0.100, 0.210, 0.310, 0.370, 0.395,
                                  0.400, 0.402, 0.410, 0.420, 0.423, 0.426, 0.429,
                                  0.433, 0.438, 0.443, 0.449, 0.456, 0.460, 0.463};
    const int n = static_cast<int>(sizeof(kP) / sizeof(kP[0]));

    double efficiency;
    if (rxPowerDbm < kP[0])
    {
        efficiency = 0.0; // below the -12 dBm sensitivity floor: no conversion
    }
    else if (rxPowerDbm >= kP[n - 1])
    {
        efficiency = kEta[n - 1]; // held flat at the +8 dBm peak (~46%)
    }
    else
    {
        efficiency = kEta[n - 1];
        for (int i = 1; i < n; ++i)
        {
            if (rxPowerDbm <= kP[i])
            {
                const double t = (rxPowerDbm - kP[i - 1]) / (kP[i] - kP[i - 1]);
                efficiency = kEta[i - 1] + t * (kEta[i] - kEta[i - 1]);
                break;
            }
        }
    }

    NS_LOG_DEBUG("Band 6 efficiency calculated: " << (efficiency * 100) << "% for " << rxPowerDbm
                                                  << " dBm input");

    return efficiency;
}

// Apply P21XXCSR-EVB capacitor class specifications
void
PowercastEnergyHarvester::ApplyCapacitorClass(CapacitorClass capClass)
{
    NS_LOG_FUNCTION(this << capClass);

    switch (capClass)
    {
    case CLASS_A:
        m_cStorage = s_capA_F;
        if (m_cStorage <= 0.0)
        {
            NS_FATAL_ERROR("CLASS_A capacitor value not set! Must load CAP_A from config file "
                           "before using CLASS_A. No fallback values allowed!");
        }
        break;
    case CLASS_B:
        m_cStorage = s_capB_F;
        if (m_cStorage <= 0.0)
        {
            NS_FATAL_ERROR("CLASS_B capacitor value not set! Must load CAP_B from config file "
                           "before using CLASS_B. No fallback values allowed!");
        }
        break;
    case CLASS_C:
        m_cStorage = s_capC_F;
        if (m_cStorage <= 0.0)
        {
            NS_FATAL_ERROR("CLASS_C capacitor value not set! Must load CAP_C from config file "
                           "before using CLASS_C. No fallback values allowed!");
        }
        break;
    default:
        NS_FATAL_ERROR(
            "Unknown capacitor class specified: "
            << capClass
            << ". Valid classes are CLASS_A, CLASS_B, CLASS_C - no fallback values allowed!");
    }

    m_capacitorClass = capClass;

    NS_LOG_INFO("Capacitor class configured: " << capClass << " (" << (m_cStorage * 1e6) << " μF)");
}

// Static methods for configuring capacitor values from config files
void
PowercastEnergyHarvester::SetCapacitorValue(CapacitorClass capClass, double capacitanceF)
{
    switch (capClass)
    {
    case CLASS_A:
        s_capA_F = capacitanceF;
        NS_LOG_INFO("CLASS_A capacitor value set to " << (capacitanceF * 1e6) << " μF");
        break;
    case CLASS_B:
        s_capB_F = capacitanceF;
        NS_LOG_INFO("CLASS_B capacitor value set to " << (capacitanceF * 1e6) << " μF");
        break;
    case CLASS_C:
        s_capC_F = capacitanceF;
        NS_LOG_INFO("CLASS_C capacitor value set to " << (capacitanceF * 1e6) << " μF");
        break;
    default:
        NS_FATAL_ERROR("Unknown capacitor class: " << capClass);
    }
}

double
PowercastEnergyHarvester::GetCapacitorValue(CapacitorClass capClass)
{
    switch (capClass)
    {
    case CLASS_A:
        return s_capA_F;
    case CLASS_B:
        return s_capB_F;
    case CLASS_C:
        return s_capC_F;
    default:
        NS_FATAL_ERROR("Unknown capacitor class: " << capClass);
        return 0.0;
    }
}

// Apply P21XXCSR-EVB voltage threshold class specifications
void
PowercastEnergyHarvester::ApplyVoltageClass(VoltageClass voltClass)
{
    NS_LOG_FUNCTION(this << voltClass);

    switch (voltClass)
    {
    case CLASS_1:
        m_vmax = 1.25; // 1.2V class: Maximum operating voltage = 1.25V
        m_vmin = 1.02; // 1.2V class: Minimum operating voltage = 1.02V
        break;
    case CLASS_2:
        m_vmax = 0.945; // 0.9V class: Maximum operating voltage = 0.945V
        m_vmin = 0.9;   // 0.9V class: Minimum operating voltage = 0.9V
        break;
    case CLASS_3:
        m_vmax = 0.738; // 0.7V class: Maximum operating voltage = 0.738V
        m_vmin = 0.64;  // 0.7V class: Minimum operating voltage = 0.64V
        break;
    default:
        NS_FATAL_ERROR(
            "Unknown voltage class specified: "
            << voltClass
            << ". Valid classes are CLASS_1, CLASS_2, CLASS_3 - no fallback values allowed!");
    }

    m_voltageClass = voltClass;

    NS_LOG_INFO("Voltage class configured: " << voltClass << " (Vmax=" << m_vmax
                                             << "V, Vmin=" << m_vmin << "V)");
}

// Public capacitor class configuration with energy recalculation
void
PowercastEnergyHarvester::SetCapacitorClass(CapacitorClass capClass)
{
    NS_LOG_FUNCTION(this << capClass);
    ApplyCapacitorClass(capClass);

    // Recalculate energy state with new capacitance value
    // Maintain voltage but update energy based on E = ½CV²
    m_nominalEnergy = 0.5 * m_cStorage * m_vcap * m_vcap;
    m_sunkEnergy = 0.5 * m_cStorage * (m_vmin * DEEP_RATIO) * (m_vmin * DEEP_RATIO);
    m_availableEnergy = m_nominalEnergy - m_sunkEnergy;
    m_availableEnergyTrace = m_availableEnergy;
}

// Public voltage class configuration
void
PowercastEnergyHarvester::SetVoltageClass(VoltageClass voltClass)
{
    NS_LOG_FUNCTION(this << voltClass);
    ApplyVoltageClass(voltClass);

    // Start Vcap at Vmin (empty/depleted state): the REHD must HARVEST before its first UL TX.
    // This makes harvest (AP beacon + PDW) the binding throttle on UL from step 1.
    m_vcap = m_vmin;
    m_vcapTrace = m_vcap;
    NS_LOG_INFO("Reset capacitor to Vmin (start depleted): " << m_vcap << "V for voltage class "
                                                             << voltClass);

    // Recalculate energy state with new voltage and thresholds
    m_nominalEnergy = 0.5 * m_cStorage * m_vcap * m_vcap;
    m_sunkEnergy = 0.5 * m_cStorage * (m_vmin * DEEP_RATIO) * (m_vmin * DEEP_RATIO);
    m_availableEnergy = m_nominalEnergy - m_sunkEnergy;
    m_availableEnergyTrace = m_availableEnergy;
}

// Complete harvester configuration with both capacitor and voltage classes
void
PowercastEnergyHarvester::Configure(CapacitorClass capClass, VoltageClass voltClass)
{
    NS_LOG_FUNCTION(this << capClass << voltClass);

    ApplyCapacitorClass(capClass);
    ApplyVoltageClass(voltClass);

    // Initialize Vcap to Vmin (empty/depleted): REHD must harvest before its first UL TX, so harvest (AP beacon + PDW) is the binding throttle on UL from step 1.
    m_vcap = m_vmin;
    m_vcapTrace = m_vcap;
    NS_LOG_INFO("Configured capacitor to Vmin (start depleted): "
                << m_vcap << "V for cap class " << capClass << ", voltage class " << voltClass);

    // Recalculate complete energy state with new configuration
    m_nominalEnergy = 0.5 * m_cStorage * m_vcap * m_vcap;
    m_sunkEnergy = 0.5 * m_cStorage * (m_vmin * DEEP_RATIO) * (m_vmin * DEEP_RATIO);
    m_availableEnergy = m_nominalEnergy - m_sunkEnergy;
    m_availableEnergyTrace = m_availableEnergy;

    NS_LOG_INFO("Harvester fully configured: Capacitor=" << capClass << ", Voltage=" << voltClass);
}

// Get current available energy for transmission decisions
double
PowercastEnergyHarvester::GetAvailableEnergy() const
{
    // Validate configuration before use
    if (m_cStorage <= 0.0 || m_vmax <= 0.0 || m_vmin <= 0.0)
    {
        NS_FATAL_ERROR("PowerCast harvester not properly configured! Must call Configure(capClass, "
                       "voltClass) or set classes explicitly. Current values: C="
                       << m_cStorage << "F, Vmax=" << m_vmax << "V, Vmin=" << m_vmin << "V");
    }

    return m_availableEnergy;
}

// Get current harvesting efficiency for given input power level
double
PowercastEnergyHarvester::GetCurrentEfficiency(double rxPowerDbm) const
{
    return CalculateRealistic24GHzEfficiency(rxPowerDbm);
}

// Pre-transmission energy sustainability check with hysteresis control
bool
PowercastEnergyHarvester::CanSustainTransmission(double txPowerDbm, Time duration) const
{
    NS_LOG_FUNCTION(this << txPowerDbm << duration.GetSeconds());

    // Validate configuration before use
    if (m_cStorage <= 0.0 || m_vmax <= 0.0 || m_vmin <= 0.0)
    {
        NS_FATAL_ERROR("PowerCast harvester not properly configured! Must call Configure(capClass, "
                       "voltClass) or set classes explicitly. Current values: C="
                       << m_cStorage << "F, Vmax=" << m_vmax << "V, Vmin=" << m_vmin << "V");
    }

    // Simple energy audit (no hysteresis, no Vmin lock): can the capacitor pay for THIS transmission without dropping below the hard floor?
    // m_availableEnergy is the energy above the DEEP_RATIO*Vmin floor (DEEP_RATIO = 1.0), so "available >= required" is exactly "Vcap stays >= Vmin after the TX".
    // TX is allowed the instant enough energy is harvested — no waiting for a full recharge to Vmax.
    double powerMw = std::pow(10.0, txPowerDbm / 10.0);           // dBm → mW
    double energyTx = (powerMw / 1000.0) * duration.GetSeconds(); // RF energy (J)
    double requiredEnergyDC = energyTx / PA_EFFICIENCY;           // DC energy incl. PA loss

    if (m_availableEnergy < requiredEnergyDC)
    {
        NS_LOG_DEBUG("TX not sustainable: " << m_availableEnergy << "J available < "
                                            << requiredEnergyDC << "J required");
        return false;
    }
    return true;
}

// Output enable status check with safety margins
bool
PowercastEnergyHarvester::IsOutputEnabled() const
{
    // Apply dual safety criteria: energy availability and voltage safety margin.
    // Threshold = DEEP_RATIO*Vmin (the same hard floor used by the TX energy audit and the idle-drain clamp) so output-enable, the TX gate and ComputePhase all agree on where "depleted" is.
    double safetyThreshold = m_vmin * DEEP_RATIO; // hard floor (DEEP_RATIO*Vmin)
    bool hasEnergy = (m_availableEnergy > 0.0);
    bool hasSafeVoltage = (m_vcap >= safetyThreshold);

    NS_LOG_DEBUG("Output enable status check:");
    NS_LOG_DEBUG("  Available energy: " << m_availableEnergy << " J");
    NS_LOG_DEBUG("  Current Vcap: " << m_vcap << " V");
    NS_LOG_DEBUG("  Safety threshold (DEEP_RATIO*Vmin): " << safetyThreshold << " V");
    NS_LOG_DEBUG("  Output enabled: " << (hasEnergy && hasSafeVoltage));

    return hasEnergy && hasSafeVoltage;
}

// Attribute accessor methods for NS-3 TypeId system
void
PowercastEnergyHarvester::SetCapacitorClassAttribute(uint32_t capClass)
{
    SetCapacitorClass(static_cast<CapacitorClass>(capClass));
}

uint32_t
PowercastEnergyHarvester::GetCapacitorClassAttribute() const
{
    return static_cast<uint32_t>(m_capacitorClass);
}

void
PowercastEnergyHarvester::SetVoltageClassAttribute(uint32_t voltClass)
{
    SetVoltageClass(static_cast<VoltageClass>(voltClass));
}

uint32_t
PowercastEnergyHarvester::GetVoltageClassAttribute() const
{
    return static_cast<uint32_t>(m_voltageClass);
}

} // namespace energy
} // namespace ns3