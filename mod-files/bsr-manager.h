// contrib/ai/examples/MobiCom/twt/bsr-manager.h

#ifndef BSR_MANAGER_H
#define BSR_MANAGER_H

#include "ns3/mac48-address.h"
#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/traced-callback.h"

#include <functional>
#include <map>

namespace ns3
{
/**
 * \brief Lightweight BSR Manager for real-time access
 *
 * Designed for RUNTIME parameter adaptation based on queue status.
 * No file I/O - pure in-memory with instant callbacks.
 */
class BsrManager : public Object
{
  public:
    /**
     * \brief Get the TypeId
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * \brief Constructor
     */
    BsrManager();

    /**
     * \brief Destructor
     */
    ~BsrManager() override;

    /**
     * \brief Get singleton instance
     * \return pointer to the singleton BsrManager
     */
    static Ptr<BsrManager> GetInstance();

    /**
     * \brief Record a new BSR entry
     * \param staAddress The STA MAC address
     * \param tid The Traffic Identifier
     * \param queueSize The queue size in units (x256 bytes)
     *
     * This fires traced callbacks immediately for runtime response
     */
    void RecordBsr(Mac48Address staAddress, uint8_t tid, uint8_t queueSize);

    /**
     * \brief Get the CURRENT queue size for a STA and TID
     * \param staAddress The STA MAC address
     * \param tid The Traffic Identifier
     * \return Current queue size in bytes (most recent BSR)
     */
    uint32_t GetCurrentQueueSize(Mac48Address staAddress, uint8_t tid) const;

    /**
     * \brief Get the CURRENT queue size in units for a STA and TID
     * \param staAddress The STA MAC address
     * \param tid The Traffic Identifier
     * \return Current queue size in units (x256 bytes)
     */
    uint8_t GetCurrentQueueSizeUnits(Mac48Address staAddress, uint8_t tid) const;

    /**
     * \brief Check if queue exceeds threshold
     * \param staAddress The STA MAC address
     * \param tid The Traffic Identifier
     * \param thresholdBytes Threshold in bytes
     * \return true if current queue size > threshold
     */
    bool IsQueueAboveThreshold(Mac48Address staAddress, uint8_t tid, uint32_t thresholdBytes) const;

    /**
     * \brief Get time since last BSR for a STA/TID
     * \param staAddress The STA MAC address
     * \param tid The Traffic Identifier
     * \return Time since last BSR report
     */
    Time GetTimeSinceLastBsr(Mac48Address staAddress, uint8_t tid) const;

    /**
     * \brief Check if any TID has data buffered
     * \param staAddress The STA MAC address
     * \return true if any TID has non-zero queue
     */
    bool HasBufferedData(Mac48Address staAddress) const;

    /**
     * \brief Get total buffered data across all TIDs
     * \param staAddress The STA MAC address
     * \return Total queue size in bytes
     */
    uint32_t GetTotalBufferedData(Mac48Address staAddress) const;

    /**
     * \brief Clear all BSR data
     */
    void Clear();

    /**
     * \brief TracedCallback signature for BSR reception
     * \param staAddress The STA MAC address
     * \param tid The Traffic Identifier
     * \param queueSize The queue size in units (x256 bytes)
     * \param queueSizeBytes The queue size in bytes
     */
    typedef void (*BsrReceivedCallback)(Mac48Address staAddress,
                                        uint8_t tid,
                                        uint8_t queueSize,
                                        uint32_t queueSizeBytes);

  private:
    /**
     * \brief Singleton instance
     */
    static Ptr<BsrManager> s_instance;

    /**
     * \brief Structure to hold current BSR state per STA/TID
     */
    struct BsrState
    {
        uint8_t queueSizeUnits{0};  ///< Current queue size in units
        uint32_t queueSizeBytes{0}; ///< Current queue size in bytes
        Time lastUpdateTime;        ///< Time of last BSR update
    };

    /**
     * \brief Map: STA Address -> TID -> BSR State
     */
    std::map<Mac48Address, std::map<uint8_t, BsrState>> m_bsrState;

    /**
     * \brief Traced callback for BSR reception
     */
    TracedCallback<Mac48Address, uint8_t, uint8_t, uint32_t> m_bsrReceivedTrace;
};

} // namespace ns3

#endif /* BSR_MANAGER_H */
