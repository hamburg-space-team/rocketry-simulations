# -*- coding: utf-8 -*-
"""
Blowdown nitrous engine simulator for hot fire and cold flow
@author: Finn Breuer
Created on Sun Dec  7 19:20:42 2025
"""

"Requirements and parameters"
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import yaml
from result_export import write_results_csv
parser = argparse.ArgumentParser(description="Simulate the engine and display its results.")
parser.add_argument("--no-gui", action="store_true", help="Run without opening the Qt results window.")
parser.add_argument("--results-file", help=argparse.SUPPRESS)
args = parser.parse_args()

parameter_names = set("""
rse_filename rse_code simulation_mode cold_flow_injectors
simulation_duration_s simulation_step_s
ops_temp_celsius ops_fuel_name ops_pressurant_name
ops_pressurizing_enable ops_heating_enable ops_pressure_target
tank_ox_diam_out tank_ox_thick tank_ox_len tank_ox_ullage
tank_fuel_diam_out tank_fuel_thick tank_fuel_len tank_fuel_piston_loss
valve_ox_cv valve_fuel_cv
inj_ox_number inj_ox_diam_mm inj_ox_cd
inj_fuel_number inj_fuel_diam_mm inj_fuel_cd
inj_film_number inj_film_diam_mm inj_film_cd
chamber_diam chamber_length chamber_throat chamber_exit
chamber_cstar_efficiency chamber_nozzle_efficiency rocket_mass_dry
""".split())
parameter_file = Path(__file__).with_name("parameters.yaml")
parameter_bytes = parameter_file.read_bytes()
parameters = yaml.safe_load(parameter_bytes)
if not isinstance(parameters, dict):
    raise ValueError(f"{parameter_file} must contain a YAML mapping of parameters")
missing = parameter_names - parameters.keys()
unknown = parameters.keys() - parameter_names
if missing or unknown:
    raise ValueError(f"Invalid {parameter_file}: missing {sorted(missing)}; unknown {sorted(unknown)}")
globals().update(parameters)
if not isinstance(rse_filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", rse_filename):
    raise ValueError("rse_filename must be an identifier using letters, digits, underscores, or hyphens")
if simulation_mode not in ("hot_fire", "cold_flow"):
    raise ValueError("simulation_mode must be hot_fire or cold_flow")
if cold_flow_injectors not in ("oxidizer", "fuel", "both"):
    raise ValueError("cold_flow_injectors must be oxidizer, fuel, or both")
if simulation_duration_s <= 0 or simulation_step_s <= 0 or simulation_step_s >= simulation_duration_s:
    raise ValueError("simulation duration and step must be positive, with step shorter than duration")
# User-facing pressures are in bar; CoolProp and the model use pascals.
ops_pressure_target *= 1e5
tank_fuel_piston_loss *= 1e5
flow_ox_enabled = simulation_mode == "hot_fire" or cold_flow_injectors in ("oxidizer", "both")
flow_fuel_enabled = simulation_mode == "hot_fire" or cold_flow_injectors in ("fuel", "both")
fuel_only_cold_flow = simulation_mode == "cold_flow" and cold_flow_injectors == "fuel"
fuel_uses_tank_pressure = simulation_mode == "hot_fire" or cold_flow_injectors == "both"

import math
import numpy as np
np.seterr(all='ignore')
import scipy
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI as propsi
ox_fluid = "N2O" if simulation_mode == "hot_fire" else "CO2"
ox_triple_temp = propsi("Ttriple", ox_fluid)
ox_triple_pressure = propsi("ptriple", ox_fluid)
ox_critical_temp = propsi("Tcrit", ox_fluid)
if simulation_mode == "hot_fire":
    from rocketcea.cea_obj_w_units import CEA_Obj
    C = CEA_Obj(oxName='N2O', fuelName=ops_fuel_name, isp_units='m/s', cstar_units='m/s', pressure_units='Pa', temperature_units='K',
                sonic_velocity_units='m/s', enthalpy_units='J/kg', density_units='kg/m^3', specific_heat_units='J/kg-K', viscosity_units='poise',
                thermal_cond_units='W/cm-degC', fac_CR=None, make_debug_prints=False, useFastLookup=0, makeOutput=0)


"Setup script"
# geometry
tank_ox_diam = tank_ox_diam_out - 2*tank_ox_thick # m, oxidizer tank inner diameter
tank_fuel_diam = tank_fuel_diam_out - 2*tank_fuel_thick # m, fuel tank inner diameter
tank_fuel_vol = (tank_fuel_diam/2)**2*np.pi * tank_fuel_len # m^3, fuel tank volume
tank_ox_vol = ( (tank_ox_diam/2)**2 - (tank_fuel_diam/2)**2 ) * np.pi * tank_ox_len # m^3, oxidizer tank volume

inj_ox_diam = inj_ox_diam_mm/1000
inj_fuel_diam = inj_fuel_diam_mm/1000
inj_film_diam = inj_film_diam_mm/1000
inj_ox_area = (inj_ox_diam/2)**2 * np.pi * inj_ox_number # m^2, cross-section area of all oxidizer injector orifices
inj_fuel_area = (inj_fuel_diam/2)**2 * np.pi * inj_fuel_number + (inj_film_diam/2)**2 * np.pi * inj_film_number # m^2, cross-section area of all fuel injector orifices


chamber_throat_area = chamber_throat**2 / 4 * np.pi # m^2, nozzle throat area
chamber_exit_area = chamber_exit**2 / 4 * np.pi # m^2, nozzle exit area
chamber_expansion = chamber_exit_area / chamber_throat_area # nozzle expansion ratio

# initial conditions
gravity = 9.81 # m/s^2
air_R = 287.07 # surrounding air gas constant 
air_pressure_zero = 101325.0  # Pa, standard pressure at MSL in Pascal
ops_temp = 273.15 + ops_temp_celsius # K
ops_pressurant_mass = 0.0 # kg, mass of inert press gas. Set to zero in case pressurization is disabled


"Injectors"

def injector_ox_hem(T1, P1, P2): # K tank temperature, Pa tank pressure, Pa chamber pressure
    ### Two-phase Homogeneous Equilibrium Model
    dm = 0 # kg/s, mass flow rate
    if P1 <= P2:
        return 0.0, 0.0, 0.0
    ### Tank condition
    if P1 > propsi ("P", "T", T1, "Q", 0, ox_fluid) + 1e3 :
        h1 = propsi ("H", "T", T1, "P", P1, ox_fluid)
        s1 = propsi ("S", "T", T1, "P", P1, ox_fluid)
        rho1 = propsi ("D", "T", T1, "P", P1, ox_fluid)
    else: 
        h1 = propsi ("H", "T", T1, "Q", 0, ox_fluid)
        s1 = propsi ("S", "T", T1, "Q", 0, ox_fluid)
        rho1 = propsi ("D", "T", T1, "Q", 0, ox_fluid)
    ### Choked flow ( After Waxman 2014 - An Investigation of Injectors...)
    def flow(P2_func): 
        rho2 = propsi ("D", "P", max(P2_func, ox_triple_pressure), "S", s1, ox_fluid)
        h2 = propsi ("H", "P", max(P2_func, ox_triple_pressure), "S", s1, ox_fluid)
        if h2 < h1: 
            dm = - inj_ox_cd*valve_ox_cv*inj_ox_area * rho2 * math.sqrt(2*(h1 - h2))
        else: 
            dm = 0
        return dm 
    if P1 > ox_triple_pressure:
        P2_crit = scipy.optimize.fminbound(flow, ox_triple_pressure, P1, xtol=1e3, maxfun=10, disp=0)
    else:
        return 0.0, 0.0, 0.0
    exit_pressure = max(P2, P2_crit)
    dm = flow(exit_pressure)
    h_exit = propsi("H", "P", exit_pressure, "S", s1, ox_fluid)
    exit_velocity = math.sqrt(max(0.0, 2 * (h1 - h_exit)))
    reaction = (-dm * exit_velocity + (exit_pressure - P2) * inj_ox_area) if dm < 0 else 0.0
    dV = dm / rho1
    return dm, dV, reaction # kg/s, m^3/s, N (axial jet reaction)


def injector_ox_gas(T1, P1, P2, rho_ox): # K tank temperature, Pa total tank pressure, Pa outlet pressure, kg/m^3 tank-fluid density
    ### One-phase ideal gas model
    dm = 0 # kg/s, mass flow rate   
    ### Tank condition
    # The tank pressure can include N2 pressurant; it is not the fluid partial pressure.
    # Use fluid density from tank mass/volume for thermodynamic properties.
    rho = rho_ox
    cp = propsi ("Cpmass", "T", T1, "D", rho, ox_fluid)
    cv = propsi ("Cvmass", "T", T1, "D", rho, ox_fluid)
    gamma = cp/cv
    ### Adiabatic flow with choke condition
    critical_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    if P1 > P2 and P2 / P1 <= critical_ratio:
        exit_pressure = P1 * critical_ratio
        dm = - inj_ox_cd*valve_ox_cv*inj_ox_area*np.sqrt( gamma * rho * P1 * (2/(gamma+1))**( (gamma+1)/(gamma-1) ) )
    elif P1 > P2:
        exit_pressure = P2
        pressure_ratio = P2 / P1
        velocity_squared = 2 * cp * T1 * (pressure_ratio**(2/gamma) - pressure_ratio**((gamma+1)/gamma))
        dm = -inj_ox_cd * valve_ox_cv * inj_ox_area * rho * math.sqrt(max(0.0, velocity_squared))
    else:
        return 0.0, 0.0, 0.0
    exit_velocity = math.sqrt(max(0.0, 2 * cp * T1 *
                                  (1 - (exit_pressure / P1)**((gamma - 1) / gamma))))
    reaction = (-dm * exit_velocity + (exit_pressure - P2) * inj_ox_area) if dm < 0 else 0.0
    dV = dm / rho
    return dm, dV, reaction # kg/s, m^3/s, N (axial jet reaction)


def injector_fuel_spi(T1, P1, P2): # K tank temperature, Pa tank pressure, Pa chamber pressure
    ### One-phase ideal liquid model    
    dm = 0 # kg/s, mass flow rate
    P1 = P1 - tank_fuel_piston_loss
    if P1 <= P2:
        return 0.0, 0.0, 0.0
    rho = propsi ("D", "T", T1, "P", P1, ops_fuel_name)
    exit_velocity = math.sqrt(2 * (P1 - P2) / rho)
    dm = -inj_fuel_cd * valve_fuel_cv * inj_fuel_area * rho * exit_velocity
    dV = dm / rho 
    return dm, dV, -dm * exit_velocity # kg/s, m^3/s, N (axial jet reaction)


"Combustion chamber and nozzle"

def chamber(dm_ox, dm_fuel): # kg/s, mass flow rates
    dm_ox = abs(dm_ox)
    dm_fuel = abs(dm_fuel)
    dm = dm_ox + dm_fuel
    if dm_fuel > 0.001 and dm_ox > 0.001: # only if both props are still flowing
        dm_ratio = dm_ox / dm_fuel
        Pc = 1e5 # Pa, chamber pressure. Initial guess with arbitrary atmospheric
        for i in range(0, 10, 1): # run couple of times to converge chamber pressure
            c_star_opt = C.get_Cstar(Pc=Pc, MR=dm_ratio)       
            c_star = c_star_opt * chamber_cstar_efficiency
            Pc = c_star * dm / chamber_throat_area
        expansion = air_pressure_zero / Pc * C.get_PcOvPe(Pc=Pc, MR=dm_ratio, eps=chamber_expansion, frozen=0, frozenAtThroat=0) # Pamb/Pe; >1 overexpanded, <1 underexpanded
        isp_amb = chamber_nozzle_efficiency * chamber_cstar_efficiency * C.estimate_Ambient_Isp(Pc=Pc, MR=dm_ratio, eps=chamber_expansion, Pamb=air_pressure_zero)[0]
    else:
        Pc = 0.0; c_star = 0.0; isp_amb = 0.0; expansion=0.0
    F = dm * isp_amb # N, thrust force
    return F, Pc, c_star, isp_amb, expansion # N thrust force, Pa chamber pressure, m/s combustion efficiency, m/s specific impulse, 


"Tank fill and press"

def tank_init_eq(T0, Vtank, Vvap_fraction): # K initial temperature, m^3 tank volume, vapor volume fraction (dip tube length dependent)
    # finds initial state of ox tank  
    rho_liq = propsi ("D", "T", T0, "Q", 0, ox_fluid)
    rho_vap = propsi ("D", "T", T0, "Q", 1, ox_fluid)
    m_liq = rho_liq * Vtank * (1-Vvap_fraction)
    m_vap = rho_vap * Vtank * Vvap_fraction    
    m = m_vap + m_liq # kg, oxidizer mass
    x = m_vap / m  # ratio, vapor mass / total mass   
    P = propsi ("P", "T", T0, "Q", x, ox_fluid) # Pa, tank pressure
    rho_fuel = propsi ("D", "T", T0, "P", air_pressure_zero, ops_fuel_name)
    m_fuel = rho_fuel * tank_fuel_vol   
    y = np.array([m, m_fuel, T0])  
    return y, P, x


def tank_init_heating(y0, P_target): # initial state, Pa firing pressure target
    # heating to firing pressure after closing vent valve 
    m, mf, T = y0 # ox mass, fuel mass, tank temperature  
    rho = m/tank_ox_vol   
    T_heated = propsi ("T", "P", P_target, "D", rho, ox_fluid)
    y = np.array([m, mf, T_heated])  
    return y


def tank_init_pressurizing(y0, P_target): # initial state, Pa firing pressure target
    # inert gas pressurization to firing pressure  after closing vent valve
    # assumes press on pad only -> constant press mass
    m, mf, T = y0 # ox mass, fuel mass, tank temperature     
    
    rho = m/tank_ox_vol # ox density
    P = propsi ("P", "T", T, "D", rho, ox_fluid)  # oxidizer tank pressure
    P_press = P_target - P # Pa, inert gas partial pressure    
    if T > ox_critical_temp:
       m_press = 0 # do not press in this region
       return m_press
    rho_liq = propsi ("D", "T", T, "Q", 0, ox_fluid)
    rho_vap = propsi ("D", "T", T, "Q", 1, ox_fluid)
    x = ( rho_vap*rho_liq - rho_vap*rho ) / ( rho*(rho_liq-rho_vap) )
    if x < 0: x = 0 
    if x > 1: x = 1
    m_vap = m * x # kg, ox vapor mass
    
    if P_press > 1e3: # only do if significant difference
        V_press = m_vap / rho_vap # only in ox gaseous phase
        rho_press = propsi ("D", "T", T, "P", P_press, ops_pressurant_name) # kg/m^3, inert gas initial density    
        m_press = rho_press * V_press
    else:
        m_press = 0     
    return m_press



"Main"

### Main engine run function with equilibrium tank
def engine_tank_eq(y, ydot, P_ch, step):
    m, mf, T = y # ox mass, fuel mass, tank temperature
    
    ### Integration
    y = y + step * ydot # explicit Euler
    y[:2] = np.maximum(y[:2], 0.0)
    ydot = np.array([0, 0, 0])  # initialize 
    
    m, mf, T = y
    if fuel_only_cold_flow:
        # Fuel-only cold flow has no CO2 charge and uses a separate regulator.
        P_ch = air_pressure_zero
        if flow_fuel_enabled and mf > 0.01:
            dmf, _, fuel_reaction = injector_fuel_spi(T, ops_pressure_target, P_ch)
        else:
            dmf = 0.0
            fuel_reaction = 0.0
        fuel_exhaust_velocity = fuel_reaction / -dmf if dmf < -0.001 else 0.0
        return (y, np.array([0.0, dmf, 0.0]), 0.0, P_ch,
                fuel_reaction, fuel_exhaust_velocity, 0.0, 0.0)
    if m <= 0.01 or T <= ox_triple_temp:
        return y, ydot, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if simulation_mode == "cold_flow":
        P_ch = air_pressure_zero
    
    ### Ox tank 
    rho = m/tank_ox_vol # ox density
    P = propsi ("P", "T", T, "D", rho, ox_fluid)  # tank pressure
    if T > ox_critical_temp:
        rho_liq = propsi ("D", "T", T, "P", P, ox_fluid)
        rho_vap = rho_liq
        x = 1 # ratio, vapor quality: gas mass / total mass
        h_vaporization = 0
        Z_vap = propsi ("Z", "T", T, "P", P, ox_fluid)
        #Z_liq = Z_vap
    else: # in liquid/vapor regime
        rho_liq = propsi ("D", "T", T, "Q", 0, ox_fluid)
        rho_vap = propsi ("D", "T", T, "Q", 1, ox_fluid)
        x = ( rho_vap*rho_liq - rho_vap*rho ) / ( rho*(rho_liq-rho_vap) )
        h_liq = propsi ("H", "T", T, "Q", 0, ox_fluid)
        h_vap = propsi ("H", "T", T, "Q", 1, ox_fluid)
        h_vaporization = h_vap - h_liq
        Z_vap = propsi ("Z", "T", T, "Q", 1, ox_fluid)
    if x < 0: x = 0 
    if x > 1: x = 1
    m_liq = m * (1-x) # kg, ox liquid mass
    m_vap = m * x # kg, ox vapor mass
    
    ### Inert gas pressurization
    if ops_pressurizing_enable:
        V_pressurant = m_vap / rho_vap # only in ox gaseous phase
        rho_pressurant = ops_pressurant_mass / V_pressurant # assumes press on pad only -> constant press mass
        P_pressurant = propsi ("P", "T", T, "D", rho_pressurant, ops_pressurant_name) # Pa, inert gas partial pressure 
        P = P + P_pressurant # approximate total pressure
    
    ### Fuel injector
    if flow_fuel_enabled and mf > 0.01:
        fuel_supply_pressure = P
        dmf, dVf, fuel_reaction = injector_fuel_spi(T, fuel_supply_pressure, P_ch)
    else:
        dmf = 0.0; dVf = 0.0; fuel_reaction = 0.0
         
    ### Ox injector       
    if m > 0.01 and m_liq > 0.01:
        dm, dV, ox_reaction = (injector_ox_hem(T, P, P_ch) if flow_ox_enabled else (0.0, 0.0, 0.0))
        cp = propsi ("Cpmass", "T", T, "Q", 0, ox_fluid)
        tank_volume_change = dV + (dVf if fuel_uses_tank_pressure else 0.0)
        dT = h_vaporization/cp * tank_volume_change*rho_vap*(Z_vap)/m
    elif m > 0.01:
        dm, dV, ox_reaction = (injector_ox_gas(T, P, P_ch, rho) if flow_ox_enabled else (0.0, 0.0, 0.0))
        cp = propsi ("Cpmass", "T", T, "D", rho, ox_fluid)
        tank_volume_change = dV + (dVf if fuel_uses_tank_pressure else 0.0)
        dT = (P*tank_volume_change*Z_vap/m) / cp
    else:
        dm = 0.0; dT = 0.0; ox_reaction = 0.0
        
    ### Chamber combustion and nozzle
    if simulation_mode == "hot_fire":
        F_thrust, P_ch, c_star, isp, expansion = chamber(dm, dmf)
        isp_real = F_thrust / abs(dm+dmf) if abs(dm+dmf) > 0.001 else 0.0
    else:
        # Axial, co-directed free jets; no chamber or nozzle acceleration.
        F_thrust = ox_reaction + fuel_reaction
        isp_real = F_thrust / abs(dm + dmf) if abs(dm + dmf) > 0.001 else 0.0
        expansion = 0.0
    
    ### Return
    ydot = np.array([dm, dmf, dT]) # rates of: ox mass, fuel mass, tank temperature
    
    return y, ydot, P, P_ch, F_thrust, isp_real, expansion, m_liq


"Run simulation"
### Setup
T_0 = ops_temp
t_end = simulation_duration_s
step = simulation_step_s

if flow_ox_enabled:
    y0, P, x = tank_init_eq(T_0, tank_ox_vol, tank_ox_ullage)
    if ops_heating_enable:
        y0 = tank_init_heating(y0, ops_pressure_target)
    if ops_pressurizing_enable:
        ops_pressurant_mass = tank_init_pressurizing(y0, ops_pressure_target)
else:
    fuel_density = propsi("D", "T", T_0, "P", ops_pressure_target, ops_fuel_name)
    y0 = np.array([0.0, fuel_density * tank_fuel_vol, T_0])
    P = 0.0
if simulation_mode == "cold_flow" and not flow_fuel_enabled:
    y0[1] = 0.0
      
sol_t = np.arange(0, t_end, step)
sol_t = np.append(sol_t, t_end) if sol_t[-1] < t_end else sol_t
sol_y = np.array([y0])
sol_ydot = np.array([[0, 0, 0]])
sol_P = np.array([ops_pressure_target if flow_ox_enabled and (ops_heating_enable or ops_pressurizing_enable) else P])
sol_Pc = np.array([air_pressure_zero])
sol_Ft = np.array([0])
sol_isp = np.array([0])   
sol_exp = np.array([0])   
stop_note = ""
liquid_burn_end = None
end_reason = "time_limit"
        
### Main loop
for t in range(1, len(sol_t), 1):

    current_step = sol_t[t] - sol_t[t-1]
    y, ydot, P_tank, P_ch, F_thrust, isp, expans, m_liq = engine_tank_eq(sol_y[t-1], sol_ydot[t-1], sol_Pc[t-1], current_step)
    if flow_ox_enabled and y[0] > 0.01 and y[2] <= ox_triple_temp:
        end_reason = "model_limit"
        stop_note = f"Stopped at {sol_t[t-1]:.2f} s: {ox_fluid} reached its triple point ({ox_triple_temp:.2f} K)."
        print(stop_note)
        sol_t = sol_t[:t]
        break
    if liquid_burn_end is None and m_liq <= 0.01:
        liquid_burn_end = sol_t[t-1]
    ox_finished = not flow_ox_enabled or y[0] <= 0.01 or (simulation_mode == "cold_flow" and y[2] <= ox_triple_temp)
    fuel_finished = not flow_fuel_enabled or y[1] <= 0.01 or (fuel_uses_tank_pressure and y[0] <= 0.01)
    depleted = (ox_finished or fuel_finished) if simulation_mode == "hot_fire" else (ox_finished and fuel_finished)
    if depleted:
        end_reason = "propellant_depleted"
        sol_t = sol_t[:t]
        break
    # if y[0]+y[1] <= 0.1 or y[2] < 100: # empty tank or too cold for coolprop
    #     sol_t = sol_t[:t]
    #     break
    
    sol_y = np.append(sol_y, [y], axis=0)
    sol_ydot = np.append(sol_ydot, [ydot], axis=0)
    sol_P = np.append(sol_P, P_tank)
    sol_Pc = np.append(sol_Pc, P_ch)
    sol_Ft = np.append(sol_Ft, F_thrust)
    sol_isp = np.append(sol_isp, isp)
    sol_exp = np.append(sol_exp, expans)

active_flow_at_end = max(-sol_ydot[-1, 0] if flow_ox_enabled else 0.0,
                         -sol_ydot[-1, 1] if flow_fuel_enabled else 0.0)
if end_reason == "time_limit":
    if simulation_mode == "hot_fire" and sol_Ft[-1] > 0.001:
        limit_note = f"Simulation limit reached at {sol_t[-1]:.2f} s while thrust was still {sol_Ft[-1]:.0f} N."
    elif simulation_mode == "cold_flow" and active_flow_at_end > 0.001:
        limit_note = f"Simulation limit reached at {sol_t[-1]:.2f} s while injector flow was still {active_flow_at_end:.3f} kg/s."
    else:
        limit_note = ""
    if limit_note:
        stop_note = f"{stop_note} {limit_note}".strip()
        print(limit_note)
if liquid_burn_end is None and end_reason == "propellant_depleted":
    liquid_burn_end = sol_t[-1]

### Design numbers
des_impulse = scipy.integrate.trapezoid(sol_Ft, x=sol_t)
des_Ft_avg = des_impulse / sol_t[-1] if sol_t[-1] > 0 else 0.0
des_Ft_max = max(sol_Ft)
des_burntime = sol_t[-1]
liquid_burn_label = (f"{liquid_burn_end:.2f} s" if liquid_burn_end is not None
                     else f">{des_burntime:.2f} s")
powered_time_label = (f">{des_burntime:.2f} s" if end_reason != "propellant_depleted" and sol_Ft[-1] > 0.001
                      else f"{des_burntime:.2f} s")
propellant_used = (y0[0] + y0[1]) - (sol_y[-1, 0] + sol_y[-1, 1])
des_isp_avg = des_impulse / (gravity * propellant_used) if propellant_used > 0 else 0.0
des_isp_max = max(sol_isp)/9.81
des_twr = max(sol_Ft)/9.81/(rocket_mass_dry+sol_y[0,0]+sol_y[0,1])
des_Pc_max = max(sol_Pc)/1e5
des_P_max = max(sol_P)/1e5
des_massflow_ox = (y0[0] - sol_y[len(sol_t)-1, 0]) / des_burntime
des_massflow_fuel = (y0[1] - sol_y[len(sol_t)-1, 1]) / des_burntime
des_of_ratio = des_massflow_ox / des_massflow_fuel if des_massflow_fuel > 0 else 0.0

if simulation_mode == "hot_fire":
    print("Liquid burn:", liquid_burn_label)
    print("Powered time:", powered_time_label)
    print("Impulse:", round(des_impulse/1000, 3), "kNs")
    print("Thrust avg:", round(des_Ft_avg), "N")
    print("Thrust max:", round(des_Ft_max), "N")
    print("Isp avg:", round(des_isp_avg, 1), "s")
    print("Isp max:", round(des_isp_max, 1), "s")
    print("TWR max:", round(des_twr, 2))
    print("P_chamber max:", round(des_Pc_max, 2), "bar")
    print("P_tank max:", round(des_P_max, 2), "bar")
    print("Ox mass flow avg:", round(des_massflow_ox, 4), "kg/s")
    print("Fuel mass flow avg:", round(des_massflow_fuel, 4), "kg/s")
    print("O/F ratio avg:", round(des_of_ratio, 4))
else:
    ox_flow_series = np.maximum(0.0, -sol_ydot[:, 0])
    fuel_flow_series = np.maximum(0.0, -sol_ydot[:, 1])
    active_flow_series = np.maximum(ox_flow_series if flow_ox_enabled else 0.0,
                                    fuel_flow_series if flow_fuel_enabled else 0.0)
    flowing_samples = np.flatnonzero(active_flow_series > 0.001)
    last_flow_time = sol_t[flowing_samples[-1]] if len(flowing_samples) else 0.0
    if end_reason == "propellant_depleted":
        flow_time_label = f"{des_burntime:.2f} s"
    elif active_flow_at_end > 0.001:
        flow_time_label = f">{des_burntime:.2f} s"
    else:
        flow_time_label = f"{last_flow_time:.2f} s"
    ox_delivered = y0[0] - sol_y[-1, 0]
    fuel_delivered = y0[1] - sol_y[-1, 1]
    ox_flow_max = max(ox_flow_series)
    fuel_flow_max = max(fuel_flow_series)
    print(f"Cold flow ({ox_fluid}, {cold_flow_injectors}), outlet {air_pressure_zero / 1e5:.3f} bar")
    print("Flow time:", flow_time_label)
    print(f"Injector thrust avg/max: {des_Ft_avg:.1f}/{des_Ft_max:.1f} N")
    print(f"Impulse: {des_impulse:.1f} Ns; average Isp: {des_isp_avg:.1f} s")
    print(f"CO2 delivered: {ox_delivered:.3f} kg; avg/max flow: {des_massflow_ox:.4f}/{ox_flow_max:.4f} kg/s")
    print(f"Fuel delivered: {fuel_delivered:.3f} kg; avg/max flow: {des_massflow_fuel:.4f}/{fuel_flow_max:.4f} kg/s")
    print(f"Tank pressure max: {des_P_max:.2f} bar; tank temperature min: {min(sol_y[:, 2]) - 273.15:.2f} C")




"Export as RSE"
# Function to calculate CG (placeholder, replace with your actual calculation)
def calculate_cg(mass_ox, mass_fuel):
    # Example: linear interpolation based on mass
    mass_ox_0 = y0[0]; mass_fuel_0 = y0[1]
    return tank_ox_len * (1 - 0.5 * (mass_ox + mass_fuel) / (mass_ox_0 + mass_fuel_0))

motor_code = rse_code if simulation_mode == "hot_fire" else f"{rse_code} CO2 Cold Flow {cold_flow_injectors}"
# Generate XML content
xml_lines = [
    '<engine-database>',
    '\t<engine-list>',
    f'\t\t <engine mfg="Eos" Type="liquid" code="{motor_code}" auto-calc-cg="0" auto-calc-mass="0" ',
    f'\t\t len="{tank_ox_len*1000}" dia="{tank_ox_diam*1000}" initWt="{round((y0[0] + y0[1])*1000)}" propWt="{round((y0[0] + y0[1])*1000)}" burn-time="{des_burntime}" ', 
    f'\t\t Isp="{round(des_isp_avg)}" Itot="{round(des_impulse)}" avgThrust="{round(des_Ft_avg)}" peakThrust="{round(des_Ft_max)}" throatDia="{chamber_throat*1000}" exitDia="{chamber_exit*1000}">',
    '\n\t\t\t<comments>',
    f'{ox_fluid} and {ops_fuel_name}, flow ratio: {round(des_of_ratio, 4)}',
    f'Temperature: {ops_temp_celsius} C, Supercharge: {ops_pressurizing_enable}, Heating: {ops_heating_enable}',
    *(["Cold flow: injector jet reaction only; axial, co-directed jets assumed."] if simulation_mode == "cold_flow" else []),
    'Mission Eos of the Hamburg Space Team',
    *([f'Simulation note: {stop_note}'] if stop_note else []),
    '\t\t\t</comments>\n',
    '\t\t\t<data>'
]

for t, thrust, y in zip(sol_t, sol_Ft, sol_y):
    mass_ox, mass_fuel, temperature = y
    mass_total = mass_ox + mass_fuel
    cg = calculate_cg(mass_ox, mass_fuel)
    xml_lines.append(f'\t\t\t\t<eng-data t="{t:.10f}" f="{thrust:.10f}" m="{(mass_total*1000):.10f}" cg="{(cg*1000):.10f}"/>')
#xml_lines.append(f'\t\t\t\t<eng-data t="{t:.10f}" f="{thrust:.10f}" m="{total_mass:.10f}" cg="{cg:.10f}"/>')


xml_lines.extend([
    '\t\t\t</data>',
    '\t\t</engine>',
    '\t</engine-list>',
    '</engine-database>'
])

# Save the input and results together in a new folder for each run.
output_root = Path(__file__).resolve().parent / "output"
output_root.mkdir(exist_ok=True)
run_name = f"{datetime.now():%y%m%d}_{rse_filename}"
run_number = 1
while True:
    suffix = "" if run_number == 1 else f"_{run_number:02d}"
    run_dir = output_root / f"{run_name}{suffix}"
    try:
        run_dir.mkdir()
        break
    except FileExistsError:
        run_number += 1
(run_dir / "input.yaml").write_bytes(parameter_bytes)
rse_path = run_dir / f"{rse_filename}.rse"
with rse_path.open("w", encoding="utf-8") as f:
    f.write('\n'.join(xml_lines))
result_data = {
    "time": sol_t, "state": sol_y, "rates": sol_ydot,
    "tank_pressure": sol_P, "chamber_pressure": sol_Pc,
    "thrust": sol_Ft, "isp": sol_isp, "expansion": sol_exp,
    "propellant_weight": (sol_y[:, 0] + sol_y[:, 1]) * gravity,
    "fuel_inlet_pressure": (np.full(len(sol_t), max(0.0, ops_pressure_target - tank_fuel_piston_loss))
                            if fuel_only_cold_flow
                            else np.maximum(0.0, sol_P - tank_fuel_piston_loss)
                            if flow_fuel_enabled else np.zeros(len(sol_t))),
}
result_data["load"] = result_data["thrust"] - result_data["propellant_weight"]
write_results_csv(run_dir / "results.csv", result_data)
print(f"Results saved to {run_dir}")

if simulation_mode == "hot_fire":
    summary = {
        "Liquid burn": liquid_burn_label,
        "Powered time": powered_time_label,
        "Impulse": f"{des_impulse / 1000:.3f} kNs",
        "Thrust avg": f"{des_Ft_avg:.0f} N",
        "Thrust max": f"{des_Ft_max:.0f} N",
        "Isp avg": f"{des_isp_avg:.1f} s",
        "Isp max": f"{des_isp_max:.1f} s",
        "TWR max": f"{des_twr:.2f}",
        "P_chamber max": f"{des_Pc_max:.2f} bar",
        "P_tank max": f"{des_P_max:.2f} bar",
        "Ox mass flow avg": f"{des_massflow_ox:.4f} kg/s",
        "Fuel mass flow avg": f"{des_massflow_fuel:.4f} kg/s",
        "O/F ratio avg": f"{des_of_ratio:.4f}",
    }
else:
    summary = {
        "Flow time": flow_time_label,
        "Thrust avg": f"{des_Ft_avg:.1f} N",
        "Thrust max": f"{des_Ft_max:.1f} N",
        "Impulse": f"{des_impulse:.1f} Ns",
        "Isp avg": f"{des_isp_avg:.1f} s",
        "CO2 delivered": f"{ox_delivered:.3f} kg",
        "Fuel delivered": f"{fuel_delivered:.3f} kg",
        "CO2 flow avg": f"{des_massflow_ox:.4f} kg/s",
        "CO2 flow max": f"{ox_flow_max:.4f} kg/s",
        "Fuel flow avg": f"{des_massflow_fuel:.4f} kg/s",
        "Fuel flow max": f"{fuel_flow_max:.4f} kg/s",
        "P_tank max": f"{des_P_max:.2f} bar",
        "Tank temp min": f"{min(sol_y[:, 2]) - 273.15:.2f} C",
    }

if args.results_file:
    np.savez_compressed(
        args.results_file,
        **result_data,
        summary=json.dumps(summary), rse_path=str(rse_path),
        run_dir=str(run_dir), mode=simulation_mode,
        run_note=stop_note,
    )

if not args.no_gui:
    from plot_viewer import show_results
    show_results(
        **result_data,
        summary=summary,
        rse_path=str(rse_path),
        run_dir=str(run_dir), mode=simulation_mode,
        run_note=stop_note,
        script_path=str(Path(__file__).resolve()),
    )
