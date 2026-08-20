#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 2-12:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o slurm.tcia_download.%j.out
#SBATCH -e slurm.tcia_download.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# Download the raw DICOM collections from TCIA using the NBIA Data Retriever CLI
# driven by tools/manifest.tcia. This is the FIRST step of rebuilding /scratch
# from scratch (see README "Reproducing the data"). Paths from jobs/config.sh.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

# MANIFEST_TCIA defaults to the durable repo copy (metadata/manifest.tcia) via
# config.sh, so it survives a scratch purge. tools/ only holds the retriever jar.
# Java comes from Sol's module system, not a JDK extracted into scratch — the
# latter was purged (lib/ went missing) since scratch isn't exempt from the
# 90-day cleanup and nothing was reading it directly.
module load openjdk-17.0.3_7-gcc-12.1.0

# The NBIA retriever ships as an RPM. On Sol we extract it (no root) with
# rpm2cpio | cpio, which yields opt/nbia-data-retriever/lib/app/StandaloneDM.jar.
# VERIFY ON FIRST RUN: the exact jar path can differ between retriever versions.
JAR="${TOOLS_DIR}/opt/nbia-data-retriever/lib/app/StandaloneDM.jar"
if [ ! -f "${JAR}" ]; then
    echo "Retriever jar not found — extracting RPM into ${TOOLS_DIR} ..."
    RPM=$(ls "${TOOLS_DIR}"/nbia-data-retriever-*.rpm 2>/dev/null | head -1)
    if [ -z "${RPM}" ]; then
        echo "ERROR: no nbia-data-retriever RPM in ${TOOLS_DIR}."
        echo "Download it from https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images"
        exit 1
    fi
    ( cd "${TOOLS_DIR}" && rpm2cpio "${RPM}" | cpio -idmv )
fi

mkdir -p "${RAW_DIR}"

# Download every series listed in the manifest. -f overwrites partials, -v is verbose.
echo "Y" | java -jar "${JAR}" \
    --cli "${MANIFEST_TCIA}" \
    -d "${RAW_DIR}" \
    -v -f

# The retriever nests output under a subdir named after the manifest file
# (historically "manifest/"). Flatten it so collections sit directly in RAW_DIR,
# i.e. RAW_DIR/CT COLONOGRAPHY, RAW_DIR/Pediatric-CT-SEG, RAW_DIR/metadata.csv.
if [ -d "${RAW_DIR}/manifest" ]; then
    echo "Flattening ${RAW_DIR}/manifest/* into ${RAW_DIR}/"
    mv "${RAW_DIR}/manifest/"* "${RAW_DIR}/"
    rmdir "${RAW_DIR}/manifest"
fi

echo "Download complete. Next: sbatch jobs/prep_array.sh"
