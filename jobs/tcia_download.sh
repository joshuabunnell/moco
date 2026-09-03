#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 5-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -J tcia_download
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# Downloads raw DICOM from TCIA via the NBIA retriever CLI, driven by manifest.tcia.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

# Java comes from Sol's module system — a JDK extracted into scratch was purged before, so nothing persists here.
module load openjdk-17.0.3_7-gcc-12.1.0

# Extracted from the RPM via rpm2cpio; jar path may differ between retriever versions — verify on first run.
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
NEST="${RAW_DIR}/manifest"

# The retriever's own "download missing" mode is unusable here: it diffs the manifest
# against the metadata.csv it wrote last time, not against files on disk, so after a
# purge (csv survives, .dcm files don't) it concludes everything is present. Diff on
# disk instead and hand it a manifest of only what's genuinely absent. Rerunning resumes.
#
# The manifest must NOT live under a dot-prefixed directory. Retriever 4.4.3 derives
# its nest dir as manifestPath.substring(lastSlash+1, firstDot); a ".tcia_pending/"
# dir puts the first '.' before the last '/', the substring underflows, and it dies
# in performDownload with StringIndexOutOfBoundsException ("begin 37, end 23", every
# job through 62515887). A plain dir leaves the only '.' in the .tcia extension, so
# the nest resolves to RAW_DIR/manifest as intended.
PENDING_MANIFEST="${DATA_ROOT}/tcia_pending/manifest.tcia"
python3 "${PROJECT_DIR}/scripts/pending_series.py" \
    --manifest "${MANIFEST_TCIA}" \
    --catalog "${SERIES_CATALOG}" \
    --raw-dir "${RAW_DIR}" \
    --output "${PENDING_MANIFEST}"

# Series UIDs are the only lines starting with a digit; header keys are alphabetic.
if [ "$(grep -c '^[0-9]' "${PENDING_MANIFEST}")" -eq 0 ]; then
    echo "Nothing to download. Next: sbatch jobs/prep_array.sh"
    exit 0
fi

rm -rf "${NEST}"
rm -f "${RAW_DIR}"/NBIADataRetrieverCLI-*.log.lck "${RAW_DIR}"/NBIADataRetrieverCLI-*.log

# -f overwrites partials, -v is verbose. stdin answers up to two prompts: Y for the
# Data Usage Agreement, then A ("download all") for the resume prompt, which shouldn't
# fire now NEST is wiped but is harmless if it does. The retriever builds a fresh
# Scanner per prompt; the first one's buffered read-ahead drains up to ~8KB of the
# pipe and starves the second, crashing it on EOF (measured: 2KB fails, 10KB works).
# Pad with 50k A lines (~100KB) so input survives; extra lines are ignored.
{ echo Y; yes A | head -n 50000; } | java -jar "${JAR}" \
    --cli "${PENDING_MANIFEST}" \
    -d "${RAW_DIR}" \
    -v -f

# Retriever nests output under a subdir named after the manifest file — flatten it
# into RAW_DIR. rsync (not mv) so it merges over any stale top-level collection dirs.
# metadata.csv is excluded: the retriever's copy describes only this run's series, and
# overwriting the full listing would blind the next run's diff.
if [ -d "${NEST}" ]; then
    echo "Flattening ${NEST}/ into ${RAW_DIR}/"
    rsync -a --exclude=metadata.csv "${NEST}/" "${RAW_DIR}/"
    rm -rf "${NEST}"
fi

python3 "${PROJECT_DIR}/scripts/pending_series.py" \
    --manifest "${MANIFEST_TCIA}" \
    --catalog "${SERIES_CATALOG}" \
    --raw-dir "${RAW_DIR}" \
    --output "${PENDING_MANIFEST}"

echo "Download pass complete. Resubmit this job if any series are still pending,"
echo "otherwise: sbatch jobs/prep_array.sh"
