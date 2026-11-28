"""Interactive results window for the engine simulation."""

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from result_export import write_results_csv


def show_results(*, time, state, rates, tank_pressure, chamber_pressure,
                 thrust, propellant_weight, load, fuel_inlet_pressure, isp, expansion,
                 summary, rse_path, run_dir, mode, script_path,
                 run_note=""):
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
        chamber_pressure=chamber_pressure, thrust=thrust,
        propellant_weight=propellant_weight, load=load,
        fuel_inlet_pressure=fuel_inlet_pressure, isp=isp,
        expansion=expansion, summary=summary, rse_path=rse_path,
        run_dir=run_dir, mode=mode,
        run_note=run_note,
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
    hot_metric_groups = [
        ("Burn", [("Liquid", "Liquid burn"), ("Powered", "Powered time")]),
        ("Impulse", [("", "Impulse")]),
        ("Thrust", [("Avg", "Thrust avg"), ("Max", "Thrust max")]),
        ("Isp", [("Avg", "Isp avg"), ("Max", "Isp max")]),
        ("TWR max", [("", "TWR max")]),
        ("Pressure max", [("Chamber", "P_chamber max"), ("Tank", "P_tank max")]),
        ("Mass flow avg", [("Ox", "Ox mass flow avg"),
                           ("Fuel", "Fuel mass flow avg")]),
        ("O/F ratio avg", [("", "O/F ratio avg")]),
    ]
    cold_metric_groups = [
        ("Flow time", [("", "Flow time")]),
        ("Thrust", [("Avg", "Thrust avg"), ("Max", "Thrust max")]),
        ("Impulse", [("", "Impulse")]),
        ("Isp avg", [("", "Isp avg")]),
        ("Delivered", [("CO₂", "CO2 delivered"), ("Fuel", "Fuel delivered")]),
        ("CO₂ flow", [("Avg", "CO2 flow avg"), ("Max", "CO2 flow max")]),
        ("Fuel flow", [("Avg", "Fuel flow avg"), ("Max", "Fuel flow max")]),
        ("Tank", [("P max", "P_tank max"), ("T min", "Tank temp min")]),
    ]
    mode_cards = {"hot_fire": [], "cold_flow": []}

    def add_metric_card(index, title_text, values):
        card = QtWidgets.QFrame()
        card.setStyleSheet(
            "QFrame { background: #1b2940; border-radius: 6px; }"
            "QLabel { color: #dce5f3; }"
        )
        card_layout = QtWidgets.QHBoxLayout(card)
        card_layout.setContentsMargins(9, 5, 9, 5)
        card_layout.setSpacing(8)
        title = QtWidgets.QLabel(title_text)
        title.setStyleSheet("font-size: 12px; color: #a9bbd2;")
        card_layout.addWidget(title, 1)
        for value_index, (caption, key) in enumerate(values):
            if value_index:
                divider = QtWidgets.QFrame()
                divider.setFixedWidth(1)
                divider.setFixedHeight(26)
                divider.setStyleSheet("background: #4b607e;")
                card_layout.addWidget(divider, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
            number = QtWidgets.QLabel(summary.get(key, "—"))
            number.setStyleSheet("font-size: 13px; font-weight: 600;")
            if caption:
                value_layout = QtWidgets.QVBoxLayout()
                value_layout.setContentsMargins(0, 0, 0, 0)
                value_layout.setSpacing(0)
                caption_label = QtWidgets.QLabel(caption)
                caption_label.setStyleSheet("font-size: 10px; color: #a9bbd2;")
                value_layout.addWidget(caption_label)
                value_layout.addWidget(number)
                card_layout.addLayout(value_layout)
            else:
                card_layout.addWidget(number)
            metric_labels.setdefault(key, []).append(number)
        metrics.addWidget(card, index // 4, index % 4)
        return card

    for card_mode, groups in (("hot_fire", hot_metric_groups),
                              ("cold_flow", cold_metric_groups)):
        for index, (title_text, values) in enumerate(groups):
            mode_cards[card_mode].append(add_metric_card(index, title_text, values))
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

    def add_plot(grid, row, column, title, ylabel, unit, series, reference_y=None):
        plot = HoverPlotWidget(title=title)
        plot.setLabel("bottom", "Time", units="s")
        plot.setLabel("left", ylabel)
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.addLegend(offset=(8, 8))
        plot.setMinimumHeight(250)
        grid.addWidget(plot, row, column)
        if reference_y is not None:
            plot.addItem(pg.InfiniteLine(
                pos=reference_y, angle=0,
                pen=pg.mkPen("#94a3b8", width=1,
                             style=QtCore.Qt.PenStyle.DashLine),
            ), ignoreBounds=True)

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
            curve = plot.plot(current["time"], values_for(current), name=name,
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

    hot_tab_index = tabs.count()
    performance = make_tab("Performance")
    add_plot(performance, 0, 0, "Load", "Force (N)", "N", [
        ("Thrust", "#65d6c0", lambda d: d["thrust"]),
        ("Propellant weight", "#f7b267", lambda d: d["propellant_weight"]),
        ("Tared load", "#d2a8ff", lambda d: d["load"]),
    ], reference_y=0.0)
    add_plot(performance, 0, 1, "Propellant flow", "Mass flow (kg/s)", "kg/s", [
        ("N₂O", "#60a5fa", lambda d: -d["rates"][:, 0]),
        ("Fuel", "#f7b267", lambda d: -d["rates"][:, 1]),
    ])
    add_plot(performance, 1, 0, "Specific impulse", "Velocity (m/s)", "m/s", [
        ("Isp", "#d2a8ff", lambda d: d["isp"]),
    ])
    add_plot(performance, 1, 1, "Nozzle pressure match", "Pamb / Pe", "", [
        ("Pamb / Pe", "#f48fb1", lambda d: d["expansion"]),
    ], reference_y=1.0)

    cold_tab_index = tabs.count()
    cold = make_tab("Cold flow")
    add_plot(cold, 0, 0, "Load", "Force (N)", "N", [
        ("Reaction", "#65d6c0", lambda d: d["thrust"]),
        ("Propellant weight", "#f7b267", lambda d: d["propellant_weight"]),
        ("Tared load", "#d2a8ff", lambda d: d["load"]),
    ], reference_y=0.0)
    add_plot(cold, 0, 1, "Injector flow", "Mass flow (kg/s)", "kg/s", [
        ("CO₂", "#60a5fa", lambda d: -d["rates"][:, 0]),
        ("Fuel", "#f7b267", lambda d: -d["rates"][:, 1]),
    ])
    add_plot(cold, 1, 0, "Discharged mass", "Mass (kg)", "kg", [
        ("CO₂", "#60a5fa", lambda d: d["state"][0, 0] - d["state"][:, 0]),
        ("Fuel", "#f7b267", lambda d: d["state"][0, 1] - d["state"][:, 1]),
    ])
    add_plot(cold, 1, 1, "Pressure", "Pressure (bar)", "bar", [
        ("CO₂ tank", "#60a5fa", lambda d: d["tank_pressure"] / 1e5),
        ("Fuel inlet", "#d2a8ff", lambda d: d["fuel_inlet_pressure"] / 1e5),
        ("Outlet", "#f7b267", lambda d: d["chamber_pressure"] / 1e5),
        ("CO₂ drop", "#65d6c0",
         lambda d: np.maximum(0.0, d["tank_pressure"] - d["chamber_pressure"]) / 1e5),
    ])

    tanks = make_tab("Tanks")
    add_plot(tanks, 0, 0, "Propellant mass", "Mass (kg)", "kg", [
        ("Tank fluid", "#60a5fa", lambda d: d["state"][:, 0]),
        ("Fuel", "#f7b267", lambda d: d["state"][:, 1]),
    ])
    add_plot(tanks, 0, 1, "Tank temperature", "Temperature (°C)", "°C", [
        ("Tank", "#f48fb1", lambda d: d["state"][:, 2] - 273.15),
    ])
    add_plot(tanks, 1, 0, "Pressure", "Pressure (bar)", "bar", [
        ("Tank", "#60a5fa", lambda d: d["tank_pressure"] / 1e5),
        ("Injector outlet", "#f7b267", lambda d: d["chamber_pressure"] / 1e5),
        ("Difference", "#65d6c0",
         lambda d: (d["tank_pressure"] - d["chamber_pressure"]) / 1e5),
    ])

    toolbar = window.addToolBar("Results")

    def result_status():
        saved = f"Results saved to {Path(current['run_dir']).resolve()}"
        return f"{current['run_note']}  ·  {saved}" if current["run_note"] else saved

    def show_mode():
        cold_flow = current["mode"] == "cold_flow"
        title = "Cold-flow results" if cold_flow else "Engine simulation results"
        heading.setText(title)
        window.setWindowTitle(title)
        for card_mode, cards in mode_cards.items():
            for card in cards:
                card.setVisible(card_mode == current["mode"])
        tabs.setTabVisible(hot_tab_index, not cold_flow)
        tabs.setTabVisible(cold_tab_index, cold_flow)
        tabs.setCurrentIndex(cold_tab_index if cold_flow else hot_tab_index)

    def refresh(new_data):
        nonlocal current
        current = new_data
        for label, widgets in metric_labels.items():
            for widget in widgets:
                widget.setText(current["summary"].get(label, "—"))
        show_mode()
        for plot, lines in models:
            plot.hide_hover()
            for _, _, values_for, curve, _ in lines:
                curve.setData(current["time"], values_for(current))
            plot.getViewBox().autoRange()
        window.statusBar().showMessage(result_status())

    reset_action = toolbar.addAction("Reset zoom")
    reset_action.triggered.connect(
        lambda: [plot.getViewBox().autoRange() for plot, _ in models]
    )

    def save_csv():
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            window, "Save simulation data",
            str(Path(current["run_dir"]) / "results-copy.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return
        d = current
        try:
            write_results_csv(path, d)
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
                    "thrust", "propellant_weight", "load", "fuel_inlet_pressure",
                    "isp", "expansion",
                )}
                new_data["summary"] = json.loads(data["summary"].item())
                new_data["rse_path"] = data["rse_path"].item()
                new_data["run_dir"] = data["run_dir"].item()
                new_data["mode"] = data["mode"].item()
                new_data["run_note"] = data["run_note"].item()
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

    show_mode()
    window.statusBar().showMessage(
        f"{result_status()}  ·  Hover for values  ·  Mouse wheel: zoom  ·  Drag: pan"
    )
    window.show()
    app.exec()
