# Requirements

- Python 3.12 with pip
- General packages: numpy, scipy, PySide6-Essentials, pyqtgraph
- Rocket specific packages: [RocketCEA](https://rocketcea.readthedocs.io/en/latest/index.html) and [CoolProp](https://coolprop.org/index.html)

### Setup

On Windows, run the script with your Python 3.12 executable. For a standard per-user installation, from this directory in PowerShell:

```powershell
$python312 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
& $python312 -m pip install numpy scipy RocketCEA CoolProp PySide6-Essentials pyqtgraph
& $python312 main.py
```

# RSE engine file generation
The script writes an `.rse` file, to be used in OpenRocket (Flight/)

# GUI
The script opens the interactive results window. Hover over a plot to see a time cursor and the values of its curves. Use the mouse wheel to zoom and drag to pan. The toolbar can reset the view, save the plotted data as CSV, or run the simulation again in the same window. Edit and save `main.py` before clicking **Run again** to use changed parameters. Run `& $python312 main.py --no-gui` when only the terminal summary and `.rse` file are needed.

# Engine model
- Blowdown simulation of bi-liquid nitrous oxide rocket engine
- coolprop for fluid properties and state equations
- rocketcea for combustion dynamics
- uses nitrous oxide as oxidizer, and for fuel a suitable liquid from coolprop and rocketcea
- HEM is used for two-phase nitrous injector flow
