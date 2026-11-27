"""Interactive results window for the engine simulation."""

import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


def show_results(*, time, state, rates, tank_pressure, chamber_pressure,
                 thrust, isp, expansion, summary, rse_path, script_path):
    try:
        from PySide6 import QtCore, QtWidgets
        import pyqtgraph as pg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The results window requires PySide6-Essentials and pyqtgraph. "
            "Install them with: python -m pip install PySide6-Essentials pyqtgraph "
            "(or use --no-gui)."
        ) from exc

    pg.setConfigOptions(antialias=True, background="#101827", foreground="#dce5f3")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rerun_dir = tempfile.TemporaryDirectory(prefix="engine-results-")
    result_file = str(Path(rerun_dir.name) / "latest.npz")

    class ResultsWindow(QtWidgets.QMainWindow):
        def closeEvent(self, event):
            if process.state() != QtCore.QProcess.ProcessState.NotRunning:
                process.kill()
                process.waitForFinished(3000)
            rerun_dir.cleanup()
            super().closeEvent(event)

    window = ResultsWindow()
    window.setWindowTitle("Engine simulation results")
    window.resize(1380, 900)
    window.setStyleSheet(
        "QMainWindow, QWidget { background: #0b1220; color: #dce5f3; }"
        "QTabWidget::pane { border: 1px solid #33445e; }"
        "QTabBar::tab { background: #1b2940; padding: 8px 20px; }"
        "QTabBar::tab:selected { background: #304a6c; }"
        "QToolBar { border: 0; spacing: 8px; }"
        "QToolButton { padding: 6px 10px; }"
    )

    current = dict(
        time=time, state=state, rates=rates, tank_pressure=tank_pressure,
        chamber_pressure=chamber_pressure, thrust=thrust, isp=isp,
        expansion=expansion, summary=summary, rse_path=rse_path,
    )

    root = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(root)
    layout.setContentsMargins(14, 12, 14, 10)
    layout.setSpacing(9)
    window.setCentralWidget(root)

    heading = QtWidgets.QLabel("Engine simulation results")
    heading.setStyleSheet("font-size: 20px; font-weight: 650;")
    layout.addWidget(heading)

    metrics = QtWidgets.QGridLayout()
    metrics.setHorizontalSpacing(8)
    metrics.setVerticalSpacing(6)
    metric_labels = {}
    for index, (label, value) in enumerate(summary.items()):
        card = QtWidgets.QFrame()
        card.setStyleSheet(
            "QFrame { background: #1b2940; border-radius: 6px; }"
            "QLabel { color: #dce5f3; }"
        )
        card_layout = QtWidgets.QHBoxLayout(card)
        card_layout.setContentsMargins(9, 5, 9, 5)
        card_layout.setSpacing(6)
        title = QtWidgets.QLabel(label)
        title.setStyleSheet("font-size: 12px; color: #a9bbd2;")
        number = QtWidgets.QLabel(value)
        number.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        number.setStyleSheet("font-size: 13px; font-weight: 600;")
        card_layout.addWidget(title, 1)
        card_layout.addWidget(number)
        metrics.addWidget(card, index // 4, index % 4)
        metric_labels[label] = number
    layout.addLayout(metrics)

    tabs = QtWidgets.QTabWidget()
    layout.addWidget(tabs, 1)
    models = []

    def make_tab(name):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)
        grid.setSpacing(12)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        tabs.addTab(tab, name)
        return grid

    class HoverPlotWidget(pg.PlotWidget):
        def eventFilter(self, watched, event):
            if (watched is self.viewport()
                    and event.type() == QtCore.QEvent.Type.Leave
                    and hasattr(self, "hide_hover")):
                self.hide_hover()
            return super().eventFilter(watched, event)

        def leaveEvent(self, event):
            if hasattr(self, "hide_hover"):
                self.hide_hover()
            super().leaveEvent(event)

    def add_plot(grid, row, column, title, ylabel, unit, series):
        plot = HoverPlotWidget(title=title)
        plot.setLabel("bottom", "Time", units="s")
        plot.setLabel("left", ylabel)
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.addLegend(offset=(8, 8))
        plot.setMinimumHeight(250)
        grid.addWidget(plot, row, column)

        cursor = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#b8c9df", width=1, style=QtCore.Qt.PenStyle.DashLine),
        )
        plot.addItem(cursor, ignoreBounds=True)
        cursor.hide()
        readout = QtWidgets.QLabel(plot)
        readout.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        readout.setStyleSheet(
            "background: #23344d; color: #f2f5fb; border: 1px solid #5b718e;"
            "border-radius: 5px; padding: 5px 8px;"
        )
        readout.hide()

        lines = []
        for name, color, values_for in series:
            curve = plot.plot(time, values_for(current), name=name,
                              pen=pg.mkPen(color, width=2.3))
            marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(color, width=2),
                                        brush=pg.mkBrush("#101827"))
            plot.addItem(marker, ignoreBounds=True)
            marker.hide()
            lines.append((name, color, values_for, curve, marker))

        def hide_hover():
            cursor.hide()
            readout.hide()
            for _, _, _, _, marker in lines:
                marker.hide()

        def mouse_moved(scene_pos):
            view = plot.getViewBox()
            if not view.sceneBoundingRect().contains(scene_pos):
                hide_hover()
                return
            times = current["time"]
            if len(times) == 0:
                return
            x = view.mapSceneToView(scene_pos).x()
            index = int(np.searchsorted(times, x))
            if index == len(times):
                index -= 1
            elif index and abs(times[index - 1] - x) < abs(times[index] - x):
                index -= 1
            sample_time = float(times[index])
            cursor.setPos(sample_time)
            cursor.show()
            rows = [f"<b>t = {sample_time:.3f} s</b>"]
            for name, color, values_for, _, marker in lines:
                value = float(values_for(current)[index])
                marker.setData([sample_time], [value])
                marker.show()
                rows.append(
                    f'<span style="color:{color}">●</span> '
                    f"{name}: {value:.4g} {unit}"
                )
            readout.setText("<br>".join(rows))
            readout.adjustSize()
            readout.move(max(0, plot.width() - readout.width() - 12), 12)
            readout.show()

        plot.hide_hover = hide_hover
        plot.viewport().installEventFilter(plot)
        plot.scene().sigMouseMoved.connect(mouse_moved)
        models.append((plot, lines))

    performance = make_tab("Performance")
    add_plot(performance, 0, 0, "Thrust", "Force (N)", "N", [
        ("Thrust", "#65d6c0", lambda d: d["thrust"]),
    ])
    add_plot(performance, 0, 1, "Propellant flow", "Mass flow (kg/s)", "kg/s", [
        ("N₂O", "#60a5fa", lambda d: -d["rates"][:, 0]),
        ("Fuel", "#f7b267", lambda d: -d["rates"][:, 1]),
    ])
    add_plot(performance, 1, 0, "Specific impulse", "Velocity (m/s)", "m/s", [
        ("Isp", "#d2a8ff", lambda d: d["isp"]),
    ])
    add_plot(performance, 1, 1, "Nozzle expansion", "Ratio", "", [
        ("Expansion", "#f48fb1", lambda d: d["expansion"]),
    ])

    tanks = make_tab("Tanks")
    add_plot(tanks, 0, 0, "Propellant mass", "Mass (kg)", "kg", [
        ("N₂O", "#60a5fa", lambda d: d["state"][:, 0]),
        ("Fuel", "#f7b267", lambda d: d["state"][:, 1]),
    ])
    add_plot(tanks, 0, 1, "Tank temperature", "Temperature (°C)", "°C", [
        ("Tank", "#f48fb1", lambda d: d["state"][:, 2] - 273.15),
    ])
    add_plot(tanks, 1, 0, "Pressure", "Pressure (bar)", "bar", [
        ("Tank", "#60a5fa", lambda d: d["tank_pressure"] / 1e5),
        ("Chamber", "#f7b267", lambda d: d["chamber_pressure"] / 1e5),
        ("Difference", "#65d6c0",
         lambda d: (d["tank_pressure"] - d["chamber_pressure"]) / 1e5),
    ])

    toolbar = window.addToolBar("Results")

    def refresh(new_data):
        nonlocal current
        current = new_data
        for label, widget in metric_labels.items():
            widget.setText(current["summary"].get(label, "—"))
        for plot, lines in models:
            plot.hide_hover()
            for _, _, values_for, curve, _ in lines:
                curve.setData(current["time"], values_for(current))
            plot.getViewBox().autoRange()
        window.statusBar().showMessage(f"RSE saved to {Path(current['rse_path']).resolve()}")

    reset_action = toolbar.addAction("Reset zoom")
    reset_action.triggered.connect(
        lambda: [plot.getViewBox().autoRange() for plot, _ in models]
    )

    def save_csv():
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            window, "Save simulation data",
            str(Path(current["rse_path"]).with_suffix(".csv")),
            "CSV files (*.csv)",
        )
        if not path:
            return
        d = current
        try:
            with open(path, "w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow([
                    "time_s", "oxidizer_kg", "fuel_kg", "tank_temperature_K",
                    "oxidizer_flow_kg_s", "fuel_flow_kg_s", "tank_pressure_Pa",
                    "chamber_pressure_Pa", "thrust_N", "isp_m_s", "expansion_ratio",
                ])
                writer.writerows(zip(
                    d["time"], d["state"][:, 0], d["state"][:, 1], d["state"][:, 2],
                    -d["rates"][:, 0], -d["rates"][:, 1], d["tank_pressure"],
                    d["chamber_pressure"], d["thrust"], d["isp"], d["expansion"],
                ))
        except OSError as exc:
            QtWidgets.QMessageBox.critical(window, "Could not save CSV", str(exc))
            return
        window.statusBar().showMessage(f"Saved {path}", 8000)

    csv_action = toolbar.addAction("Save data as CSV")
    csv_action.triggered.connect(save_csv)

    process = QtCore.QProcess(window)
    process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
    process.setWorkingDirectory(str(Path.cwd()))

    def run_again():
        rerun_action.setEnabled(False)
        window.statusBar().showMessage("Running simulation…")
        process.start(sys.executable, [script_path, "--no-gui", "--results-file", result_file])

    def finished(exit_code, exit_status):
        rerun_action.setEnabled(True)
        output = bytes(process.readAllStandardOutput()).decode(errors="replace")
        if exit_status != QtCore.QProcess.ExitStatus.NormalExit or exit_code != 0:
            QtWidgets.QMessageBox.critical(
                window, "Simulation failed", output[-5000:] or process.errorString(),
            )
            window.statusBar().showMessage("Simulation failed")
            return
        try:
            with np.load(result_file, allow_pickle=False) as data:
                new_data = {key: data[key].copy() for key in (
                    "time", "state", "rates", "tank_pressure", "chamber_pressure",
                    "thrust", "isp", "expansion",
                )}
                new_data["summary"] = json.loads(data["summary"].item())
                new_data["rse_path"] = data["rse_path"].item()
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.critical(window, "Could not load results", str(exc))
            window.statusBar().showMessage("Could not load simulation results")
            return
        refresh(new_data)

    rerun_action = toolbar.addAction("Run again")
    rerun_action.triggered.connect(run_again)
    process.finished.connect(finished)
    process.errorOccurred.connect(
        lambda error: finished(-1, QtCore.QProcess.ExitStatus.CrashExit)
        if error == QtCore.QProcess.ProcessError.FailedToStart else None
    )

    window.statusBar().showMessage(
        f"RSE saved to {Path(rse_path).resolve()}  ·  Hover for values  ·  Mouse wheel: zoom  ·  Drag: pan"
    )
    window.show()
    app.exec()
