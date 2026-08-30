#!/bin/bash
# Script to install BsrManager into NS-3 WiFi module
# This adds real-time BSR access for adaptive TWT control
#
# Usage: ./install_bsr_manager.sh
#
# What this script does:
# 1. Copies bsr-manager.h and bsr-manager.cc to src/wifi/model/
# 2. Updates src/wifi/CMakeLists.txt to include BsrManager files
# 3. Patches src/wifi/model/qos-frame-exchange-manager.cc to integrate BsrManager
# 4. Backs up original files before modification

set -e # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  NS-3 BsrManager Installation Script${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
WIFI_MODEL_DIR="$NS3_ROOT/src/wifi/model"
WIFI_CMAKE="$NS3_ROOT/src/wifi/CMakeLists.txt"
MOD_FILES_DIR="$SCRIPT_DIR" # Script is now IN mod-files directory

echo -e "${YELLOW}NS-3 Root:${NC} $NS3_ROOT"
echo -e "${YELLOW}WiFi Model Dir:${NC} $WIFI_MODEL_DIR"
echo -e "${YELLOW}Mod Files Dir:${NC} $MOD_FILES_DIR\n"

# Check if we're in the right directory
if [ ! -d "$NS3_ROOT/src/wifi" ]; then
    echo -e "${RED}ERROR: NS-3 directory structure not found!${NC}"
    echo "Expected to find: $NS3_ROOT/src/wifi"
    echo "Current directory: $SCRIPT_DIR"
    echo "Please run this script from: contrib/ai/examples/MobiCom/twt/mod-files/"
    exit 1
fi

# Verify source files exist
if [ ! -f "$MOD_FILES_DIR/bsr-manager.h" ] || [ ! -f "$MOD_FILES_DIR/bsr-manager.cc" ]; then
    echo -e "${RED}ERROR: BsrManager source files not found in mod-files/!${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Copying BsrManager files to WiFi module${NC}"
echo "----------------------------------------"

# Copy bsr-manager.h
if [ -f "$WIFI_MODEL_DIR/bsr-manager.h" ]; then
    echo -e "${YELLOW}  ⚠ bsr-manager.h already exists, backing up...${NC}"
    cp "$WIFI_MODEL_DIR/bsr-manager.h" "$WIFI_MODEL_DIR/bsr-manager.h.backup.$(date +%Y%m%d_%H%M%S)"
fi
cp "$MOD_FILES_DIR/bsr-manager.h" "$WIFI_MODEL_DIR/bsr-manager.h"
echo -e "${GREEN}  ✓ Copied bsr-manager.h${NC}"

# Copy bsr-manager.cc
if [ -f "$WIFI_MODEL_DIR/bsr-manager.cc" ]; then
    echo -e "${YELLOW}  ⚠ bsr-manager.cc already exists, backing up...${NC}"
    cp "$WIFI_MODEL_DIR/bsr-manager.cc" "$WIFI_MODEL_DIR/bsr-manager.cc.backup.$(date +%Y%m%d_%H%M%S)"
fi
cp "$MOD_FILES_DIR/bsr-manager.cc" "$WIFI_MODEL_DIR/bsr-manager.cc"
echo -e "${GREEN}  ✓ Copied bsr-manager.cc${NC}\n"

echo -e "${GREEN}Step 2: Updating WiFi CMakeLists.txt${NC}"
echo "----------------------------------------"

# Backup CMakeLists.txt
if [ ! -f "$WIFI_CMAKE.backup" ]; then
    cp "$WIFI_CMAKE" "$WIFI_CMAKE.backup"
    echo -e "${GREEN}  ✓ Backed up original CMakeLists.txt${NC}"
else
    echo -e "${YELLOW}  ⚠ Backup already exists, skipping backup${NC}"
fi

# Check if bsr-manager is already in CMakeLists.txt
if grep -q "bsr-manager.cc" "$WIFI_CMAKE"; then
    echo -e "${YELLOW}  ⚠ bsr-manager.cc already in CMakeLists.txt, skipping...${NC}"
else
    # Add bsr-manager.cc after block-ack-manager.cc in source_files
    sed -i '/model\/block-ack-manager.cc/a \    model/bsr-manager.cc' "$WIFI_CMAKE"
    echo -e "${GREEN}  ✓ Added bsr-manager.cc to source_files${NC}"
fi

if grep -q "bsr-manager.h" "$WIFI_CMAKE"; then
    echo -e "${YELLOW}  ⚠ bsr-manager.h already in CMakeLists.txt, skipping...${NC}"
else
    # Add bsr-manager.h after block-ack-manager.h in header_files
    sed -i '/model\/block-ack-manager.h/a \    model/bsr-manager.h' "$WIFI_CMAKE"
    echo -e "${GREEN}  ✓ Added bsr-manager.h to header_files${NC}"
fi

echo ""

echo -e "${GREEN}Step 3: Patching qos-frame-exchange-manager.cc${NC}"
echo "----------------------------------------"

QOS_FEM_FILE="$WIFI_MODEL_DIR/qos-frame-exchange-manager.cc"

# Backup original file
if [ ! -f "$QOS_FEM_FILE.backup" ]; then
    cp "$QOS_FEM_FILE" "$QOS_FEM_FILE.backup"
    echo -e "${GREEN}  ✓ Backed up original qos-frame-exchange-manager.cc${NC}"
else
    echo -e "${YELLOW}  ⚠ Backup already exists, skipping backup${NC}"
fi

# Check if already patched
if grep -q "bsr-manager.h" "$QOS_FEM_FILE"; then
    echo -e "${YELLOW}  ⚠ Already patched (bsr-manager.h included), skipping...${NC}"
else
    # Add include for bsr-manager.h after wifi-mac-trailer.h
    sed -i '/#include "wifi-mac-trailer.h"/a #include "bsr-manager.h"' "$QOS_FEM_FILE"
    echo -e "${GREEN}  ✓ Added #include \"bsr-manager.h\"${NC}"

    # Now add the BsrManager integration code in PreProcessFrame method
    # We need to find the section where SetBufferStatus is called and add our code after it

    # Create a temporary file with the patch
    cat >/tmp/bsr_patch.txt <<'EOF'
                m_apMac->SetBufferStatus(hdr.GetQosTid(),
                                         mpdu->GetOriginal()->GetHeader().GetAddr2(),
                                         hdr.GetQosQueueSize());
                
                // ====== REAL-TIME BSR ACCESS - Runtime Variable Access ======
                auto bsrManager = BsrManager::GetInstance();
                if (bsrManager)
                {
                    bsrManager->RecordBsr(hdr.GetAddr2(),
                                         hdr.GetQosTid(),
                                         hdr.GetQosQueueSize());
                }
                // ====== END - TracedCallback fired immediately ======
EOF

    # Use awk to insert the code after SetBufferStatus call
    awk '
        /m_apMac->SetBufferStatus\(hdr\.GetQosTid\(\)/ {
            print
            getline; print  # Print the next line (address parameter)
            getline; print  # Print the next line (queue size parameter)
            # Now insert our BsrManager code
            print "                "
            print "                // ====== REAL-TIME BSR ACCESS - Runtime Variable Access ======"
            print "                auto bsrManager = BsrManager::GetInstance();"
            print "                if (bsrManager)"
            print "                {"
            print "                    bsrManager->RecordBsr(hdr.GetAddr2(),"
            print "                                         hdr.GetQosTid(),"
            print "                                         hdr.GetQosQueueSize());"
            print "                }"
            print "                // ====== END - TracedCallback fired immediately ======"
            next
        }
        { print }
    ' "$QOS_FEM_FILE" >"$QOS_FEM_FILE.tmp" && mv "$QOS_FEM_FILE.tmp" "$QOS_FEM_FILE"

    echo -e "${GREEN}  ✓ Added BsrManager integration to PreProcessFrame()${NC}"
fi

echo ""

echo -e "${GREEN}Step 4: Verifying installation${NC}"
echo "----------------------------------------"

# Verify files were copied
if [ -f "$WIFI_MODEL_DIR/bsr-manager.h" ] && [ -f "$WIFI_MODEL_DIR/bsr-manager.cc" ]; then
    echo -e "${GREEN}  ✓ BsrManager files present in WiFi model${NC}"
else
    echo -e "${RED}  ✗ BsrManager files NOT found in WiFi model${NC}"
    exit 1
fi

# Verify CMakeLists.txt was updated
if grep -q "bsr-manager.cc" "$WIFI_CMAKE" && grep -q "bsr-manager.h" "$WIFI_CMAKE"; then
    echo -e "${GREEN}  ✓ CMakeLists.txt updated${NC}"
else
    echo -e "${RED}  ✗ CMakeLists.txt NOT properly updated${NC}"
    exit 1
fi

# Verify qos-frame-exchange-manager.cc was patched
if grep -q "bsr-manager.h" "$QOS_FEM_FILE" && grep -q "BsrManager::GetInstance" "$QOS_FEM_FILE"; then
    echo -e "${GREEN}  ✓ qos-frame-exchange-manager.cc patched${NC}"
else
    echo -e "${RED}  ✗ qos-frame-exchange-manager.cc NOT properly patched${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete! ✓${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Rebuild NS-3: ${GREEN}./ns3 build${NC}"
echo "  2. Test with: ${GREEN}./ns3 run \"wns3-unilateral-twt --simId=10008\"${NC}"
echo ""
echo -e "${YELLOW}Backup files created:${NC}"
[ -f "$WIFI_CMAKE.backup" ] && echo "  - $WIFI_CMAKE.backup"
[ -f "$QOS_FEM_FILE.backup" ] && echo "  - $QOS_FEM_FILE.backup"
echo ""
echo -e "${YELLOW}To rollback:${NC}"
echo "  1. mv $WIFI_CMAKE.backup $WIFI_CMAKE"
echo "  2. mv $QOS_FEM_FILE.backup $QOS_FEM_FILE"
echo "  3. rm $WIFI_MODEL_DIR/bsr-manager.*"
echo "  4. Rebuild NS-3"
echo ""
