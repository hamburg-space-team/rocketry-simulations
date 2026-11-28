"""Write the simulation series as a CSV file."""

import csv


def write_results_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "time_s", "oxidizer_kg", "fuel_kg", "tank_temperature_K",
            "oxidizer_flow_kg_s", "fuel_flow_kg_s", "tank_pressure_Pa",
            "injector_outlet_pressure_Pa", "fuel_inlet_pressure_Pa",
            "thrust_N", "propellant_weight_N", "load_N", "isp_m_s",
            "ambient_to_exit_pressure_ratio",
        ])
        writer.writerows(zip(
            data["time"], data["state"][:, 0], data["state"][:, 1],
            data["state"][:, 2], -data["rates"][:, 0],
            -data["rates"][:, 1], data["tank_pressure"],
            data["chamber_pressure"], data["fuel_inlet_pressure"],
            data["thrust"], data["propellant_weight"], data["load"],
            data["isp"],
            data["expansion"],
        ))
