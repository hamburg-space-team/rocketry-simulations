# Requirements

- Python 3.12 with pip
- General packages: numpy, scipy, PyYAML, PySide6-Essentials, pyqtgraph
- Rocket specific packages: [RocketCEA](https://rocketcea.readthedocs.io/en/latest/index.html) and [CoolProp](https://coolprop.org/index.html)

### Setup

On Windows, run the script with your Python 3.12 executable. For a standard per-user installation, from this directory in PowerShell:

```powershell
$python312 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
& $python312 -m pip install numpy scipy PyYAML RocketCEA CoolProp PySide6-Essentials pyqtgraph
& $python312 main.py
```

# Simulation output and OpenRocket
Each run creates a folder under `output/` named `YYMMDD_<rse_filename>`, containing `input.yaml` (the exact input snapshot), `results.csv` with all plotted series, and `<rse_filename>.rse` for OpenRocket. Cold-flow RSE files have a distinct cold-flow motor designation. Repeated runs with the same title on one day get `_02`, `_03`, and so on. The CSV is created in GUI and `--no-gui` runs. The GUI's **Save data as CSV** action can save another copy elsewhere.

Keep OpenRocket pointed at this top-level `Engine` directory. Its motor loader searches configured directories recursively, so the RSE files in `output/` can be found without adding each run folder. OpenRocket loads external thrust curves at startup; restart it to see newly generated runs.

Edit [parameters.yaml](parameters.yaml) to change engine and operating settings. The script loads the YAML from its own directory, including when started elsewhere. Save changes before clicking **Run again** in the GUI. Use `rse_filename` as a short configuration identifier, for example `eos1_base_v01` or `eos1_n2press_v01`; the folder adds the run date, and the YAML snapshot records the exact settings.

Set `simulation_mode: cold_flow` to model an unlit injector test with CO₂ in the oxidizer tank. `cold_flow_injectors` selects `oxidizer`, `fuel`, or `both`; an unselected stream starts with zero mass. The selected injectors discharge to `air_pressure_zero` (1.01325 bar) without a chamber or nozzle calculation. In `fuel` mode, a separate regulated supply drives the fuel piston at `ops_pressure_target` (bar in YAML), less `tank_fuel_piston_loss`. In `both` mode, the piston follows the CO₂ tank pressure as it blows down; fuel displacement contributes to the CO₂ tank cooling. The cold-flow thrust estimate adds the reaction from co-directed axial injector jets, including exit pressure thrust for choked CO₂ flow. Real injector angles and impingement can change the measured axial force. A `both` run continues after fuel runs out while CO₂ still flows. If CO₂ is depleted, the fuel loses its pressure source. The run stops if CO₂ reaches its triple point (216.592 K), where this liquid/vapor model no longer applies. Use `simulation_duration_s` and `simulation_step_s` to set the test window and integration interval. Cold-flow output reports reaction, impulse, delivered mass, injector flow, tank pressure, and temperature. A `>` on flow time means the run reached its time or model limit while flow continued. Set `simulation_mode: hot_fire` for the N₂O combustion simulation. CO₂ and N₂O have different thermophysical properties, so calibrate injector discharge coefficients against measured cold-flow data. Ambient cold-flow pressure can differ substantially from firing chamber pressure; compare test data at the same downstream condition.

# GUI
The script opens the interactive results window with related summary values grouped into pills. A `>` prefix means the simulation hit its time or temperature limit before that phase ended. The **Load** plot shows thrust or injector reaction, the remaining propellant weight, and their difference. The load sensor is tared on the dry engine, so plotted load is `reaction - (remaining tank fluid + remaining fuel) * 9.81` N. This is the predicted sensor load, not the rocket's net flight force. The CSV includes `propellant_weight_N` and `load_N`; its `oxidizer_*` columns contain CO₂ in cold-flow mode and N₂O in hot-fire mode. Hot-fire mode also shows specific impulse and nozzle pressure match; cold-flow mode shows injector flow, discharged mass, and pressure. Tank temperature is on the Tanks tab. The toolbar can reset the view, save the plotted data as CSV, or run the simulation again in the same window. Run `& $python312 main.py --no-gui` when only terminal results and output files are needed.

# Engine model
- Blowdown simulation of a bi-liquid N₂O rocket engine and CO₂ cold-flow tests
- coolprop for fluid properties and state equations
- rocketcea for combustion dynamics
- Uses N₂O for hot fire, CO₂ for cold flow, and a suitable liquid fuel from CoolProp and RocketCEA
- HEM is used for two-phase tank-fluid injector flow above its triple point
- Nitrogen pressurization contributes to total tank pressure, while gas-phase fluid properties use the tank-fluid density
