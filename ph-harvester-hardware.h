// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file ph-harvester-hardware.h
 * @brief PowerCast P21XXCSR-EVB RF energy harvester model (rectenna + boost converter + storage capacitor)
 *
 * One REHD's harvester: the Band-6 (2450 MHz) RF-to-DC efficiency curve from the datasheet, a storage capacitor whose energy is split into nominal/sunk/available, and the TX energy audit that gates uplink.
 *
 * - Efficiency: 0 below -12 dBm, ~46% peak at +8 dBm, held flat above; input clamped at the +15 dBm rating
 * - Fixed 85% boost-converter efficiency (datasheet) on harvest, 75% PA / RF-chain efficiency on TX
 * - Strict two-threshold capacitor: TX may not take Vcap below Vmin (DEEP_RATIO = 1.0), harvest stops at Vmax (SHOOT_RATIO = 1.0)
 * - CanSustainTransmission() is the pre-TX audit; IsOutputEnabled() and ComputePhase() use the same floor
 * - Capacitor classes A/B/C (values from ph-harvester-config.txt, no hardcoded defaults) x voltage classes 1.2/0.9/0.7 V
 * - NS-3 trace sources for Vcap, available energy and instantaneous efficiency
 */

#ifndef PH_POWERCAST_ENERGY_HARVESTER_H
#define PH_POWERCAST_ENERGY_HARVESTER_H

#include "ns3/energy-harvester.h" // Base class for energy harvesting devices
#include "ns3/nstime.h"           // Time handling for simulation events
#include "ns3/traced-value.h"     // Monitoring and tracing system integration

namespace ns3
{
namespace energy
{

/**
 * @brief P21XXCSR-EVB energy harvester for the 2.4 GHz Wi-Fi band
 *
 * Hardware model: P21XXCSR-EVB Band 6
 * - Operating frequency: 2.4 GHz (2400-2500 MHz, center 2450 MHz)
 * - Input power range: -12 to +15 dBm (datasheet efficiency curve, clamped at +15 dBm)
 * - Boost efficiency: 85% (datasheet)
 * - PA / RF-chain efficiency on TX: 75%
 * - Energy state: nominal / sunk (below DEEP_RATIO*Vmin) / available
 * - Configuration: three capacitor classes x three voltage-threshold classes
 * - Pre-TX energy audit (CanSustainTransmission) and output-enable status
 */
class PowercastEnergyHarvester : public EnergyHarvester
{
  public:
    /**
     * @brief Capacitor class enumeration based on P21XXCSR-EVB specifications
     *
     * Capacitor values are loaded from config file (no hardcoded defaults):
     * - CLASS_A: Small electrolytic (default: 500μF from config)
     * - CLASS_B: Medium electrolytic (default: 2200μF from config)
     * - CLASS_C: Large electrolytic (default: 20000μF from config)
     */
    enum CapacitorClass
    {
        CLASS_A,
        CLASS_B,
        CLASS_C
    };

    /**
     * @brief Voltage threshold class enumeration based on P21XXCSR-EVB specifications
     *
     * Based on P21XXCSR-EVB voltage threshold settings (JP3, JP4, JP5 jumper configurations):
     * - CLASS_1: 1.2V setting (VMAX=1.25V, VMIN=1.02V) - High voltage operation
     * - CLASS_2: 0.9V setting (VMAX=0.945V, VMIN=0.9V) - Medium voltage operation
     * - CLASS_3: 0.7V setting (VMAX=0.738V, VMIN=0.64V) - Low voltage operation
     */
    enum VoltageClass
    {
        CLASS_1, ///< 1.2V threshold setting (JP3) for high voltage operation
        CLASS_2, ///< 0.9V threshold setting (JP4) for medium voltage operation
        CLASS_3  ///< 0.7V threshold setting (JP5) for low voltage operation
    };

    /**
     * @brief Get TypeId for NS-3 object system integration
     * @return TypeId with P21XXCSR-EVB configuration attributes and trace sources
     */
    static TypeId GetTypeId();

    /**
     * @brief Set custom capacitor value for a class (in Farads)
     * @param capClass The capacitor class to configure
     * @param capacitanceF Capacitance value in Farads
     *
     * Allows overriding default capacitor values from configuration files.
     * Must be called before creating harvester instances.
     */
    static void SetCapacitorValue(CapacitorClass capClass, double capacitanceF);

    /**
     * @brief Get capacitor value for a class (in Farads)
     * @param capClass The capacitor class to query
     * @return Capacitance value in Farads
     */
    static double GetCapacitorValue(CapacitorClass capClass);

    /**
     * @brief Default constructor
     *
     * Starts with CLASS_A / CLASS_1 (1.2 V) placeholders and an empty capacitor.
     * Capacitor values must be loaded from the config file and Configure() called before use; there is no fallback.
     */
    PowercastEnergyHarvester();

    /**
     * @brief Virtual destructor for proper inheritance cleanup
     */
    virtual ~PowercastEnergyHarvester() = default;

    // --- CORE ENERGY MANAGEMENT METHODS ---

    /**
     * @brief Credit harvested RF energy to the capacitor
     *
     * Converts received RF power at 2.4 GHz to stored energy through the datasheet Band-6 efficiency curve and the boost-converter efficiency, then recomputes nominal/sunk/available energy.
     *
     * @param rxPowerDbm Received RF power in dBm
     * @param duration Reception (harvest) duration
     *
     * Efficiency curve (P21XXCSR-EVB Band 6, 2450 MHz, 0.7 V datasheet trace):
     * - Below -12 dBm: 0% (below the sensitivity floor)
     * - -12 to -5 dBm: 0% to ~40% (steep sensitivity-threshold rise)
     * - -5 to +8 dBm: ~40% to ~46% (gradual climb to the peak)
     * - +8 dBm and above: held at the ~46% peak (excess input attenuated to optimum)
     *
     * Limits:
     * - Input power clamped at +15 dBm (hardware rating)
     * - Vcap capped at SHOOT_RATIO*Vmax (= Vmax)
     */
    void SetHarvestedEnergy(double rxPowerDbm, Time duration);

    /**
     * @brief Debit the DC energy of one transmission from the capacitor
     *
     * @param txPowerDbm Transmission power in dBm
     * @param duration Transmission duration
     *
     * Energy model:
     * - RF energy: E_RF = P * t
     * - DC energy: E_DC = E_RF / PA_EFFICIENCY (0.75)
     *
     * CanSustainTransmission() should have approved the TX first.
     * An over-draw that slips past it (e.g. an A-MPDU larger than the audited packet) is clamped at the DEEP_RATIO*Vmin floor with an NS_LOG_WARN rather than aborting the run.
     * NS_FATAL_ERROR only if the harvester was never configured.
     */
    void SetConsumedEnergy(double txPowerDbm, Time duration);

    /**
     * Drain a raw amount of DC energy from the capacitor (no RF/PA conversion).
     * Used for the small awake-state (IDLE/RX/CCA) housekeeping draw — being
     * awake costs a little capacitor energy even when not transmitting. Clamps
     * gracefully at the deep-discharge floor (DEEP_RATIO*Vmin; never
     * NS_FATAL_ERRORs, unlike the TX path).
     * @param energyJ DC energy to remove (Joules); non-positive is a no-op.
     */
    void ConsumeDcEnergy(double energyJ);

    // --- CONFIGURATION METHODS ---

    /**
     * @brief Set capacitor class with automatic energy state recalculation
     * @param capClass Capacitor class (CLASS_A, CLASS_B, CLASS_C - values from config file)
     */
    void SetCapacitorClass(CapacitorClass capClass);

    /**
     * @brief Set voltage threshold class with automatic energy state recalculation
     * @param voltClass Voltage class (CLASS_1: 1.2V, CLASS_2: 0.9V, CLASS_3: 0.7V)
     */
    void SetVoltageClass(VoltageClass voltClass);

    /**
     * @brief Complete harvester configuration with both capacitor and voltage classes
     * @param capClass Capacitor class for energy storage specification
     * @param voltClass Voltage threshold class for operational voltage range
     */
    void Configure(CapacitorClass capClass, VoltageClass voltClass);

    // --- GETTER METHODS ---

    /**
     * @brief Get current available energy for transmission decisions
     * @return Current usable energy in Joules (excludes sunk energy)
     */
    double GetAvailableEnergy() const;

    /**
     * @brief Get current capacitor voltage
     * @return Current capacitor voltage in Volts
     */
    double GetVcap() const
    {
        return m_vcap;
    }

    /**
     * @brief Get P21XXCSR-EVB Band 6 harvesting efficiency for given input power
     * @param rxPowerDbm Input power level for efficiency calculation
     * @return RF-to-DC efficiency (0.0 to ~0.46) at the given input power
     */
    double GetCurrentEfficiency(double rxPowerDbm) const;

    /**
     * @brief Pre-transmission energy sustainability validation
     *
     * Validates whether the harvester has sufficient available energy to sustain
     * a transmission without entering critical energy states. Should be called
     * before every transmission attempt to prevent unrealistic energy consumption.
     *
     * @param txPowerDbm Transmission power in dBm
     * @param duration Transmission duration
     * @return true if transmission can be sustained with current available energy
     *
     * Validation Process:
     * - Calculates total DC energy required (including PA efficiency)
     * - Compares against current available energy
     * - Returns false if insufficient energy available
     * - Logs the shortfall at NS_LOG_DEBUG
     */
    bool CanSustainTransmission(double txPowerDbm, Time duration) const;

    /**
     * @brief Check harvester output enable status
     *
     * @return true if output is enabled (energy available AND Vcap >= DEEP_RATIO*Vmin)
     *
     * Criteria:
     * - Available energy greater than 0 J
     * - Vcap at or above the DEEP_RATIO*Vmin floor (= Vmin), the same floor the TX audit and ComputePhase() use
     */
    bool IsOutputEnabled() const;

    /**
     * @brief Get current capacitor class
     * @return Current capacitor class
     */
    CapacitorClass GetCapacitorClass() const
    {
        return m_capacitorClass;
    }

    /**
     * @brief Get current voltage class
     * @return Current voltage threshold class
     */
    VoltageClass GetVoltageClass() const
    {
        return m_voltageClass;
    }

    // --- ENERGY ACCOUNTING AND STATISTICS ---

    /**
     * @brief Get total cumulative harvested energy
     * @return Total energy harvested since initialization (Joules)
     */
    double GetTotalHarvestedEnergy() const
    {
        return m_totalHarvestedEnergy;
    }

    /**
     * @brief Get total cumulative consumed energy
     * @return Total energy consumed since initialization (Joules)
     */
    double GetTotalConsumedEnergy() const
    {
        return m_totalConsumedEnergy;
    }

    /**
     * @brief Get number of power cycling events
     * @return Count of on/off cycles (maintained for compatibility)
     */
    uint32_t GetOnOffCycles() const
    {
        return m_onOffCycles;
    }

    // --- QoEH metric accessors (Phase 2 — for S_r / phase / capacitor-state oracle) ---

    /**
     * @brief Cumulative time (µs) the capacitor voltage has been above V_min.
     *        Updated event-driven at each SetHarvestedEnergy / SetConsumedEnergy
     *        call. NOTE: the latest interval (from last vcap-changing event to
     *        Simulator::Now()) is not counted until the next event. For BI-level
     *        metric collection this lag is negligible (PHY events fire many
     *        times per BI).
     */
    uint64_t GetTotalActiveTimeUs() const
    {
        return m_totalActiveTime_us;
    }

    double GetCapMaxVoltage() const
    {
        return m_vmax;
    }

    double GetCapMinVoltage() const
    {
        return m_vmin;
    }

    double GetNominalEnergy() const
    {
        return m_nominalEnergy;
    }

    double GetSunkEnergy() const
    {
        return m_sunkEnergy;
    }

    /**
     * @brief Classify the capacitor's current operating band (oracle metric).
     *
     * No-hysteresis model: phase is a pure function of Vcap relative to the hard
     * floor (DEEP_RATIO*Vmin), Vmin and Vmax — there is no longer a latch. Stays
     * consistent with IsOutputEnabled() (output is off only in the protection band)
     * and the TX energy audit (which floors at DEEP_RATIO*Vmin).
     *
     * @return  3 = protection   (Vcap <= DEEP_RATIO*Vmin — at the hard floor; TX gated)
     *          0 = critical-low (DEEP_RATIO*Vmin < Vcap < Vmin — alive, below nominal Vmin; empty while DEEP_RATIO = 1.0)
     *          2 = discharging  (Vmin <= Vcap < Vmax — healthy operating band)
     *          1 = active       (Vcap >= Vmax — fully charged / saturated)
     */
    uint8_t ComputePhase() const
    {
        const double floorV = m_vmin * DEEP_RATIO; // hard floor (DEEP_RATIO*Vmin)
        if (m_vcap <= floorV)
        {
            return 3; // protection (at/below hard floor)
        }
        if (m_vcap >= m_vmax)
        {
            return 1; // active (saturated)
        }
        if (m_vcap >= m_vmin)
        {
            return 2; // discharging (healthy band)
        }
        return 0; // critical-low (floor..Vmin, recovering)
    }

  protected:
    /**
     * @brief NS-3 lifecycle cleanup
     */
    virtual void DoDispose() override;

  private:
    /**
     * @brief P21XXCSR-EVB Band 6 realistic efficiency calculation with input validation
     *
     * Implements realistic efficiency curve based on P21XXCSR-EVB Band 6 datasheet
     * characterization. Provides accurate efficiency modeling for 2450 MHz center
     * frequency operation with proper sensitivity threshold and saturation behavior.
     *
     * @param rxPowerDbm Input power level in dBm (should be pre-clamped to +15 dBm max)
     * @return Datasheet RF-to-DC efficiency (0.0-0.46)
     *
     * Efficiency Regions (P21XXCSR-EVB Band 6, 2450 MHz, 0.7 V datasheet trace):
     * - Below -12 dBm: 0% (below the sensitivity floor)
     * - -12 to -5 dBm: 0% to ~40% (steep sensitivity-threshold rise)
     * - -5 to +8 dBm: ~40% to ~46% (gradual climb to the peak)
     * - +8 dBm and above: held at the ~46% peak (excess input attenuated to optimum)
     */
    double CalculateRealistic24GHzEfficiency(double rxPowerDbm) const;

    /**
     * @brief Apply P21XXCSR-EVB capacitor class specifications
     * @param capClass Capacitor class to configure
     */
    void ApplyCapacitorClass(CapacitorClass capClass);

    /**
     * @brief Apply P21XXCSR-EVB voltage threshold class specifications
     * @param voltClass Voltage class to configure
     */
    void ApplyVoltageClass(VoltageClass voltClass);

    // --- NS-3 ATTRIBUTE SYSTEM ACCESSORS ---

    void SetCapacitorClassAttribute(uint32_t capClass);
    uint32_t GetCapacitorClassAttribute() const;
    void SetVoltageClassAttribute(uint32_t voltClass);
    uint32_t GetVoltageClassAttribute() const;

    // --- DEVICE PARAMETERS ---

    double m_vcap;     ///< Current capacitor voltage (V)
    double m_cStorage; ///< Storage capacitance (F)
    double m_vmax;     ///< Maximum operating voltage threshold (V)
    double m_vmin;     ///< Minimum operating voltage threshold (V)

    // --- FIXED P21XXCSR-EVB PARAMETERS ---

    static const double BOOST_EFFICIENCY;    ///< Fixed 85% boost converter efficiency (datasheet)
    static const double PA_EFFICIENCY;       ///< Fixed 75% PA / RF-chain efficiency
    static const double OPERATING_FREQUENCY; ///< Fixed 2450 MHz center frequency
    static const double
        DEEP_RATIO; ///< Hard-floor ratio: Vcap may not drop below DEEP_RATIO*Vmin (1.0)
    static const double
        SHOOT_RATIO; ///< Harvest cap: Vcap may not rise above SHOOT_RATIO*Vmax (1.0)

    // --- DEVICE CONFIGURATION ---

    CapacitorClass m_capacitorClass; ///< Current capacitor class configuration
    VoltageClass m_voltageClass;     ///< Current voltage threshold class configuration

    // --- ENERGY STATE ---

    double m_availableEnergy;      ///< Usable energy for operations (J)
    double m_nominalEnergy;        ///< Total stored energy including sunk energy (J)
    double m_sunkEnergy;           ///< Unusable energy below deep discharge threshold (J)
    double m_totalHarvestedEnergy; ///< Cumulative harvested energy for statistics (J)
    double m_totalConsumedEnergy;  ///< Cumulative consumed energy for statistics (J)
    uint32_t m_onOffCycles;        ///< Power cycling events counter (for compatibility)

    // --- QoEH ACTIVE-TIME TRACKING (Phase 2) ---

    // m_totalActiveTime_us accumulates µs while vcap > vmin.
    // m_lastUpdateTime / m_lastActiveState capture the state at the previous vcap-changing event so each SetHarvested/SetConsumedEnergy call can integrate the elapsed interval into the active-time counter (event-driven; no extra simulator events scheduled).
    uint64_t m_totalActiveTime_us;
    ns3::Time m_lastUpdateTime;
    bool m_lastActiveState;

    /**
     * @brief Integrate the elapsed time since the previous vcap-changing event
     *        into m_totalActiveTime_us if vcap was above vmin during that
     *        interval. Called at the START of SetHarvestedEnergy /
     *        SetConsumedEnergy so the "before" state is what's accounted for.
     */
    void UpdateActiveTime();

    // --- NS-3 TRACE SOURCES ---

    TracedValue<double> m_vcapTrace;            ///< Capacitor voltage trace for monitoring
    TracedValue<double> m_availableEnergyTrace; ///< Available energy trace for monitoring
    TracedValue<double> m_efficiencyTrace;      ///< Instantaneous efficiency trace for analysis

    // --- STATIC CAPACITOR VALUE STORAGE (MUST be loaded from config files) ---

    static double s_capA_F; ///< CLASS_A capacitance in Farads (MUST be set from config, no default)
    static double s_capB_F; ///< CLASS_B capacitance in Farads (MUST be set from config, no default)
    static double s_capC_F; ///< CLASS_C capacitance in Farads (MUST be set from config, no default)
};

} // namespace energy
} // namespace ns3

#endif // PH_POWERCAST_ENERGY_HARVESTER_H