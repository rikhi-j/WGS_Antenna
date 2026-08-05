# ANSYS S11 add-on files

Add these files to the same folder as your current project files:

- `ansys_config.py`
- `ansys_s11_bridge.py`
- `example_ansys_s11_workflow.py`

## Setup

1. Open `ansys_config.py`.
2. Set:
   - `ANSYS_EXE`
   - `PROJECT_PATH`
   - `PROJECT_NAME`
   - `DESIGN_NAME`
   - `REPORT_NAME`
3. Make sure the report named by `REPORT_NAME` already exists in HFSS.
4. Run this first:

```bash
python ansys_s11_bridge.py
```

That only generates the HFSS script. It does not launch ANSYS.

## Run ANSYS and read S11

```python
from ansys_s11_bridge import run_ansys_export

result = run_ansys_export()

freq = result.frequency
s11 = result.s11

print(result.min_s11_db)
print(result.min_s11_frequency)
```

## Use with the current WGS files

Run:

```bash
python example_ansys_s11_workflow.py
```

By default this only validates a WGS particle and generates the HFSS script. After editing `ansys_config.py`, set:

```python
RUN_ANSYS = True
```

inside `example_ansys_s11_workflow.py`.

## Current limitation

These add-on files export S11/frequency from an existing HFSS project into Python. They do not yet create HFSS geometry from `WGS_Antenna` pixels. That would be the next add-on: converting `grid_or_wgs_to_polygons()` output into HFSS rectangles/polylines.
