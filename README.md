**PSO-Based Optimization of Optically Transparent Waveguide Slot Antennas
**

Project Purpose

This project automates the optimization of a pixelated Optically Transparent Waveguide Slot (WGS) antenna using Particle Swarm Optimization (PSO) and ANSYS HFSS. The framework automatically generates candidate designs, repairs invalid geometries, simulates them in HFSS, evaluates their performance, and continuously improves future designs.


Project Workflow

PSO 
->
Generate Geometry 
->
Repair
 ->
Validate
 ->
Generate HFSS Cutouts
 ->
Run HFSS
 ->
Export S11 & Gain
 ->
Score Antenna
 ->
Update PSO
 ->
Repeat

Repository Structure

- csv_optimizer.py -- Main optimization controller
- optimization_controller.py -- Coordinates workflow
- pso_optimizer.py -- Particle Swarm Optimization
- WGS_pixels.py -- WGS antenna representation
- wgs_illegal_handler.py -- Illegal geometry repair
- wgs_full_repair_handler.py -- Full repair pipeline
- validator_updated_only.py -- Geometry validation
- wgs_to_ansys_geometry.py -- HFSS cutout generation
- ansys_s11_bridge.py -- HFSS automation and CSV parsing
- design_evaluator.py -- Scoring system
- ansys_config.py -- Configuration

Software Requirements

- Python 3
- ANSYS Electronics Desktop / HFSS
- NumPy
- Matplotlib
- Shapely
- Initial Setup

Update the following values in ansys_config.py: 
- ANSYS_EXE 
- MASTER_PROJECT_PATH 
- WORKING_PROJECT_PATH 
- PROJECT_NAME 
- DESIGN_NAME 
- REPORT_NAME 
- GAIN_REPORT_NAME 
- EXPORT_CSV 
- EXPORT_GAIN_CSV

The HFSS master project must already contain: 
- Complete antenna model 
- Analysis setup 
- Existing S11 report 
- Existing Gain report 
- Conductor sheets used by the cutout script

Running the Project
- Run: python csv_optimizer.py

Output Files
- optimizer_state.pkl
- particle_history.jsonl
- particle_results/
- current_candidate.py
- generated_export_s11.py
