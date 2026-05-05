# nxcals-methods

**Liam J. O'Shaughnessy** (Princeton University '26, Physics)  
Under the supervision of **Amaury Beeckman** — CERN Beams Department, Operations Group, Proton Synchrotron  
2025 CERN Summer Student Programme

---

## Overview

This repository contains two related pieces of work from the 2025 CERN Summer Student Programme, both focused on longitudinal beam quality monitoring at the **Proton Synchrotron (PS)**:

1. **`nxcalsExtractors.py`** — A Python API for extracting, processing, and visualizing beam data from NXCALS, CERN's accelerator logging system based on Hadoop/Spark.

2. **`tomography/`** — Jupyter notebooks for benchmarking the **LBO** (Longitudinal Beam Observer) tomography system against the existing **Tomoscope**, supporting the operational case for LBO adoption by BE-OP.

The accompanying technical report is included as a PDF and available on the CERN repository:  
📄 [`Benchmarking_of_Tomographic_Reconstruction_Systems_for_Longitudinal_Beam_Quality_in_the_Proton_Synchrotron.pdf`](Benchmarking_of_Tomographic_Reconstruction_Systems_for_Longitudinal_Beam_Quality_in_the_Proton_Synchrotron.pdf)  
🔗 https://repository.cern/records/8sxe2-r0722

---

## Background

**NXCALS** (Next Generation Controls and Logging System) is CERN's Hadoop-based data logging infrastructure that stores accelerator telemetry — beam emittance, bunch length, intensity, and more — tagged by device, property, field, user (beam destination/program), and cyclestamp. Accessing this data requires a Spark session running on CERN's computing infrastructure (SWAN or lxplus).

The PS serves multiple users (beam programs) simultaneously, such as **TOF**, **SFTPRO1**, and others. Each user's data arrives on distinct machine cycles, identified by cyclestamps linked to the `CPS:NXCALS_FUNDAMENTAL` variable.

**Longitudinal tomography** reconstructs the phase-space distribution of a particle bunch from consecutive profile measurements, giving quantities like 90% emittance, bunch length, and momentum spread. Two systems exist at the PS: the legacy Tomoscope and the newer LBO. This work benchmarks them head-to-head.

---

## Repository Structure

```
nxcals-methods/
├── nxcalsExtractors.py       # NXCALS data extraction, statistics, and plotting API
├── tomography/
│   ├── Tomographics.ipynb            # Initial exploration: emittance & bunch length via NXCALS
│   ├── Tomographics_virtual.ipynb    # Adaptation for virtual machine / offline use
│   ├── SFT_analysis.ipynb            # SFTPRO1 user analysis using the extractor API
│   └── tomo_vs_lbo_analysis.ipynb    # Head-to-head Tomoscope vs LBO benchmarking
└── Benchmarking_of_..._Proton_Synchrotron.pdf  # Technical report
```

---

## `nxcalsExtractors.py` — API Reference

This module requires an active Spark session (see [Setup](#setup)). All functions assume timestamps in `'YYYY-MM-DD HH:MM:SS.mmm'` format (Europe/Berlin timezone as used by NXCALS).

### Data Extraction

#### `extractNxcalsData(spark, user, startTime, endTime, nxcalsDevice, nxcalsProperty, fields=[])`

The main entry point. Fetches data for a given user and time window, then automatically handles scalar, 1D vector, and 2D tensor fields and merges them into a single Pandas DataFrame.

```python
df = extractNxcalsData(
    spark,
    user        = "TOF",
    startTime   = "2025-05-11 16:00:00.000",
    endTime     = "2025-05-13 04:00:00.000",
    nxcalsDevice    = "PS.RING.PROC.BUNCH_PROFILES_BCW_OP",
    nxcalsProperty  = "BunchLengthData",
    fields      = ["meanEmitt90Perc", "bunchIntensityE10"],  # [] returns all fields
)
# Returns: Pandas DataFrame with columns [cyclestamp, field1, field2, ...]
```

**Field type dispatch (automatic):**

| NXCALS Type | Python type | Handler |
|-------------|-------------|---------|
| `Double`, `Integer`, `Long` | Scalar | `extractRawScalar` |
| `Struct` (dim=1) | 1D array | `extractRawVector` |
| `Struct` (dim=2) | 2D array | `extractRawTensor` |

#### Lower-level helpers

| Function | Input field type | Returns |
|----------|-----------------|---------|
| `extractRawScalar(data, field)` | Scalar | `cyclestamp`, scalar value |
| `extractRawVector(data, field)` | 1D array | `cyclestamp`, array (mode-length filtered) |
| `extractRawTensor(data, field)` | 2D array `[[value, indexer]]` | `cyclestamp`, reshaped array |
| `fetchData(spark, user, startTime, endTime, device, property)` | — | Raw user-filtered Spark DataFrame |
| `filterUser(spark, data, user, startTime, endTime)` | — | Spark DataFrame filtered to user's cyclestamps |

### Statistics

#### `getNxcalsStats(spark, user, startTime, endTime, nxcalsDevice, nxcalsProperty, field)`

Computes summary statistics for a single field, dispatching by field type.

```python
stats = getNxcalsStats(
    spark,
    user       = "SFTPRO1",
    startTime  = "2025-06-01 01:00:00.000",
    endTime    = "2025-07-21 04:00:00.000",
    nxcalsDevice   = "PS.RING.PROC.TOMO_BCW_OP_BURST_12",
    nxcalsProperty = "TomoResult",
    field      = "meanEmitt90Perc",
)
```

| Function | Field type | Returns |
|----------|-----------|---------|
| `getStatsScalar(data, field)` | Scalar | `pandas.describe()` summary |
| `getStatsVector(data, field)` | 1D array | Element-wise time-average vector |
| `getStatsTensor(data, field)` | 2D array | Per-indexer min/max/mean/stddev over time |

### Plotting

| Function | Input | Output |
|----------|-------|--------|
| `plotRawScalar(data, user, field)` | Output of `extractRawScalar` | Scatter plot: scalar vs. time (Europe/Berlin timezone) |
| `plotStatsVector(data, user)` | Output of `getStatsVector` | Line plot: averaged array value vs. vector index |
| `plotStatsTensor(data, user, time)` | Output of `getStatsTensor`, instant | Two plots: min/max/mean and stddev vs. indexer at that instant |

---

## Notebooks

### `Tomographics.ipynb`

Initial exploration notebook developed while building the extractor API. Pulls 90% emittance (`meanEmitt90Perc`) and bunch length data from the TOMO and bunch profile NXCALS streams for the TOF user, demonstrates raw Spark queries, user filtering via cyclestamp joins, and scatter plot visualization. This notebook contains the pre-API hand-written versions of the patterns later generalized in `nxcalsExtractors.py`.

**Key devices/properties accessed:**
- `PS.RING.PROC.TOMO_BCW_OP_BURST_12/TomoResult` — LBO tomography results
- `PS.RING.PROC.BUNCH_PROFILES_BCW_OP/BunchLengthData` — bunch length profiles

### `Tomographics_virtual.ipynb`

A version of `Tomographics.ipynb` adapted for offline or virtual machine use (e.g., without live Spark access). Useful for developing and testing analysis logic without a connected SWAN session.

### `SFT_analysis.ipynb`

Analysis notebook for the **SFTPRO1** user (Slow Extracted Fixed Target beam program), built using the finalized `nxcalsExtractors.py` API. Plots 90% emittance and bunch length distributions over an extended run period (June–July 2025), computing per-user statistics.

### `tomo_vs_lbo_analysis.ipynb`

The core benchmarking notebook supporting the technical report. Compares the legacy Tomoscope and the new LBO system directly on the same PS cycles using `pyda`/`pyda_japc`/`pyda_rda3` for real-time data acquisition. Key analyses include:

- Reconstructing phase-space distributions from both systems for the same bunch
- Comparing bunch profile traces and computed observables (emittance, bunch length, momentum spread)
- RF parameter extraction and bucket matching using the [solfege](https://gitlab.cern.ch/rf-br/solfege) library (`AcceleratorParameters`, `BeamParameters`, `RFParameters`)
- Potential well calculations and separatrix overlays
- Loss function convergence during tomographic reconstruction

**Key packages used:** `pyda`, `pyda_rda3`, `solfege`, `scipy`, `concurrent.futures`

---

## Setup

This code runs on **CERN's computing infrastructure**. It is not designed for use outside of this environment.

### Requirements

- Access to CERN SWAN (Service for Web-based ANalysis) or lxplus with a Spark session
- CERN account with access to NXCALS and the CMW control system
- Active PS beam program data in the NXCALS logging system

### Starting a Spark session (SWAN / lxplus)

The session configuration block is included (commented out) at the top of `nxcalsExtractors.py`:

```python
from nxcals.spark_session_builder import get_or_create, Flavor

conf = {
    'spark.driver.memory': '8g',
    'spark.executor.memory': '8g',
    'spark.executor.cores': 8,
    'spark.executor.instances': 8,
}
spark = get_or_create(app_name='my_analysis', flavor=Flavor.YARN_LARGE, conf=conf)
```

Uncomment this block and run it in your notebook before calling any `nxcalsExtractors` functions. The `spark` object is then passed as the first argument to all extraction and statistics functions.

### Importing the extractors

```python
import sys
sys.path.append('/path/to/nxcals-methods/')
from nxcalsExtractors import extractNxcalsData, getNxcalsStats, plotRawScalar
```

Or, if working from within the repo root:

```python
from nxcalsExtractors import *
```

---

## Minimal Example

```python
# 1. Start Spark session (see Setup above)

# 2. Extract bunch length data for TOF user over two days
from nxcalsExtractors import extractNxcalsData, plotRawScalar

df = extractNxcalsData(
    spark,
    user            = "TOF",
    startTime       = "2025-05-11 16:00:00.000",
    endTime         = "2025-05-13 04:00:00.000",
    nxcalsDevice    = "PS.RING.PROC.BUNCH_PROFILES_BCW_OP",
    nxcalsProperty  = "BunchLengthData",
    fields          = ["meanBunchLength"],
)

# 3. Plot
scalar_df = df[['cyclestamp', 'meanBunchLength']]
plotRawScalar(scalar_df, user="TOF", field="meanBunchLength")
```

---

## Notes on NXCALS Data

**Cyclestamps** are nanosecond-precision Unix timestamps that identify individual PS machine cycles. They are the primary join key across all NXCALS datasets.

**Users** (beam programs) are identified by filtering the `CPS:NXCALS_FUNDAMENTAL` variable. Calling `filterUser` before extracting fields ensures you only see data for the requested beam destination (e.g., `"TOF"` for the Time-of-Flight experiment, `"SFTPRO1"` for the Slow Fixed Target proton beam).

**Vector fields** in NXCALS are sometimes unnecessarily nested (a known NXCALS upload issue). `extractRawVector` and `getStatsVector` automatically detect and flatten this extra nesting before processing.

**Timezone**: All timestamps are stored in UTC internally. The plotting functions convert to `Europe/Berlin` for display, which is standard at CERN.

---

## Report

The technical report benchmarks the LBO tomographic reconstruction system against the deployed Tomoscope for longitudinal beam quality monitoring at the PS. It covers the tomographic reconstruction algorithm, a comparison of emittance and bunch length measurements from both systems across multiple users and fill periods, and the operational recommendation for LBO adoption by BE-OP-PS.

Full citation:

```
L. J. O'Shaughnessy, "Benchmarking of Tomographic Reconstruction Systems for 
Longitudinal Beam Quality in the Proton Synchrotron," CERN Summer Student Report, 
2025. https://repository.cern/records/8sxe2-r0722
```

---

## Acknowledgements

Thanks to Amaury Beeckman (BE-OP-PS) for supervision and project definition, and to the CERN BE-OP group for access to PS beam data and operational context during the 2025 Summer Student Programme.
