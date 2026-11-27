#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================================
 NITROUS OXIDE ROCKET-TANK FILLING SIMULATION
=====================================================================================
 A lumped control-volume (0-D network) transient simulation of filling a rocket
 oxidiser tank with nitrous oxide (N2O) from a reserve/supply tank through a line
 of pipes and on/off valves.

 Physics / methods
 -----------------
 * Every pipe and tank is a control volume (CV / "node") with two
   conserved quantities per species and one energy quantity:  m_N2O, m_air, U.
 * Fluid properties for N2O come from CoolProp (Helmholtz EOS, real fluid,
   two-phase capable).  The residual air that initially fills the line is treated
   as an ideal gas occupying the gas/ullage volume (Dalton partial pressures).
 * Two-phase N2O flow through the valve orifices uses the Homogeneous Equilibrium
   Model (HEM): isentropic expansion of the upstream stagnation state to the throat,
   mass flux  G = rho_t * sqrt(2 (h0 - h_t)),  maximised over throat pressure.
   The maximum of G identifies the CHOKED condition (critical flow); the code
   alerts you whenever a valve is choked.
 * Air-dominated nodes (before N2O arrives) use the compressible ideal-gas nozzle
   equation for the orifice, also with choking detection.
 * Valve orifices additionally respect the supplied Kv (flow-coefficient) limit
   (single-phase incompressible sizing); the actual flow is the more restrictive
   of the HEM/gas orifice result and the Kv result.
 * Pipe / valve-section pressure losses use Darcy-Weisbach friction (Haaland f).
 * The ODE system (m_N2O, m_air, U for every node) is advanced with the EXPLICIT
   (forward) EULER method.
 * SI units everywhere.  Temperatures in KELVIN.

 Flow path (as requested):
   reserve tank -> valve1 -> pipe1 -> valve2 -> pipe2 -> valve3 -> pipe3
                -> valve4 -> pipe4 -> rocket tank -> valve5
 Each valve is a zero-volume flow branch between adjacent control volumes.

 Outputs: time histories of pressure and temperature in every pipe, every valve
 and the rocket tank, plus the rising liquid level in the rocket tank, the position
 of every valve, and a report of any choked-flow events.
=====================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp import AbstractState
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent

# =====================================================================================
#  1.  USER INPUTS  --  edit everything in this block
# =====================================================================================

# ---- Fluids -------------------------------------------------------------------------
FLUID = "NitrousOxide"          # CoolProp name for N2O
P_ATM = 1.0e5                    # [Pa]  initial air pressure in line & rocket tank

# ---- Simulation control -------------------------------------------------------------
# STABILITY / PERFORMANCE NOTE:
#   Each step is advanced with an embedded Euler/Heun (RK1/RK2) pair; the gap between
#   the two estimates drives an adaptive step size, so DT_INIT only sets the very first
#   trial step.  If the controller still can't converge within MAX_RETRIES, or you see
#   "NUMERICAL INSTABILITY", tighten RTOL/ATOL_* or lower DT_MAX.
#   Runtime scales with accepted+rejected steps AND with how many valves pass two-phase
#   N2O (the HEM flash is the cost) -- most of that cost now falls only in transients.
T_TOTAL = 60.0                  # [s]   total simulated time (+ 10.0 s extension when STOP_ON_ROCKET_FILL enabled)

DT_INIT = 2.0e-4                # [s]   initial trial step size
DT_MAX  = 0.1                   # [s]   hard ceiling on the adaptive step size
DT_MIN  = 1.0e-6                # [s]   hard floor / stuck-controller guard
RTOL    = 1.0e-3                # [-]   relative error tolerance (shared)
ATOL_M  = 1.0e-4                # [kg]  absolute error tolerance for mN2O, mair
ATOL_U  = 1.0e-2               # [J]   absolute error tolerance for U
SAFETY  = 0.5                   # [-]   step-size controller safety factor
GROW_MAX, SHRINK_MIN = 50.0, 0.02 # [-]   per-step growth/shrink limits
MAX_RETRIES = 20                # max step-halvings before giving up on one step

PRINT_EVERY = 100      # print terminal time every N-th ACCEPTED step (set to 1 for every step)

# ---- Branch-flow numerical relaxation ----------------------------------------------
# First-order damping of jumps in the quasi-steady branch-flow solutions.  This is a
# numerical time constant, not a measured valve or pipe property; set to 0 to disable.
FLOW_TAU = 0.10               # [s]

# ---- Ambient / initial line temperature ---------------------------------------------
T_AMBIENT = 293.15             # [K]   initial temperature of line & rocket-tank air

# ---- Reserve (supply) tank ----------------------------------------------------------
RESERVE = dict(
    V         = 0.010,         # [m^3]  bottle internal volume (10 L)
    mass      = 8.0,           # [kg]   total N2O mass in the bottle
    T         = T_AMBIENT,     # [K]    initial bottle/N2O temperature
)

# ---- Rocket tank (vertical concentric annular tank) ------------------------------
ROCKET = dict(
    D_outer = 0.104,          # [m]  outer diameter of the concentric annulus
    D_inner = 0.050,          # [m]  inner diameter of the concentric annulus
    H = 1.000,                # [m]  height
    dip_tube_L = 0.150,       # [m]  dip-tube insertion length measured down from the top
    T = T_AMBIENT,            # [K]  initial air temperature
)

# ---- Main pipes 1..4  (circular) ----------------------------------------------------
#  length [m], inner diameter [m], Darcy equivalent sand roughness [m]
PIPES = [
    dict(L=1.50, D=0.010, roughness=1.5e-6),   # pipe1: smooth PTFE hose
    dict(L=0.50, D=0.010, roughness=4.0e-6),   # pipe2: GSE panel piping
    dict(L=5.00, D=0.010, roughness=1.5e-6),   # pipe3: smooth PTFE hose
    dict(L=0.20, D=0.010, roughness=50.0e-6),  # pipe4: nominal as-printed aluminium
]

# ---- Valves 1..5 --------------------------------------------------------------------
#  Each valve is a zero-volume orifice between adjacent control volumes.
#  Kv  = valve flow coefficient (metric units, m^3/h at 1 bar) -> used as a single-phase limit.
#  Cd  = orifice discharge coefficient (dimensionless, ~0.6-0.85).
#  t_actuat = time [s] for the valve to travel fully closed<->open (same rate both ways).
VALVES = [
    dict(Kv=3.00, d_orifice=0.0100, Cd=0.80, t_actuat=0.1),  # valve1
    dict(Kv=1.50, d_orifice=0.0100, Cd=0.65, t_actuat=0.1),  # valve2
    dict(Kv=2.00, d_orifice=0.0100, Cd=0.80, t_actuat=0.1),  # valve3
    dict(Kv=1.60, d_orifice=0.0100, Cd=0.80, t_actuat=0.1),  # valve4
    dict(Kv=0.50, d_orifice=0.0010, Cd=0.60, t_actuat=0.1),  # valve5
]

# ---- Valve schedules ----------------------------------------------------------------
#  For each valve, a list of (time_seconds, state) events.  state: 1 = fully open, 0 = closed.
#  The valve holds the last state whose time <= current sim time.
#  Example below: all valves open at t = 0.  Edit freely, e.g. staged opening.
#  NOTE: When STOP_ON_ROCKET_FILL is enabled, valves 1-4 close 0.5s after rocket fill target is reached.
VALVE_SCHEDULE = [
    [(0.0, 1)],                # valve1 (closes 0.5s after fill target reached if STOP_ON_ROCKET_FILL enabled)
    [(0.0, 1)],     # valve2 (closes 0.5s after fill target reached if STOP_ON_ROCKET_FILL enabled)
    [(0.0, 0), (1.0, 1)],     # valve3 (closes 0.5s after fill target reached if STOP_ON_ROCKET_FILL enabled)
    [(0.0, 1)],     # valve4 (closes 0.5s after fill target reached if STOP_ON_ROCKET_FILL enabled)
    [(0.0, 1)],                # valve5
]

# ---- Stopping criteria --------------------------------------------------------------
#  When STOP_ON_ROCKET_FILL is enabled, the simulation will continue until the rocket tank
#  liquid reaches the dip-tube tip.  The remaining ullage-height fraction is therefore
#  dip_tube_L / H.  Valves 1-4 then close after 0.5s and the simulation continues for
#  10s more to observe the pressure decay in the rocket tank.
#  NOTE: When T_TOTAL is reached, the simulation will stop regardless of the rocket fill level.

STOP_ON_ROCKET_FILL = False    # [bool] enable/disable rocket fill stopping condition
if not 0.0 <= ROCKET["dip_tube_L"] <= ROCKET["H"]:
    raise ValueError("ROCKET dip_tube_L must be between zero and the tank height H")
ROCKET_ULLAGE_TARGET = ROCKET["dip_tube_L"] / ROCKET["H"]
ROCKET_LIQUID_TARGET = 1.0 - ROCKET_ULLAGE_TARGET

# ---- Air (ideal gas) properties -----------------------------------------------------
R_AIR   = 287.05               # [J/(kg K)]
GAMMA_A = 1.400                # [-]
CV_AIR  = R_AIR / (GAMMA_A - 1.0)
CP_AIR  = GAMMA_A * CV_AIR

# ---- Numerical guards ---------------------------------------------------------------
M_FLOOR   = 1.0e-12            # [kg]  minimum species mass kept in a node
MAX_DRAIN = 0.20               # a branch may drain at most this fraction of upstream
                               #        species mass per step (outflow limiter)
HEM_NPTS  = 24                 # throat-pressure samples in the HEM mass-flux sweep
N2O_RICH  = 0.5                # mass fraction above which a node is "N2O dominated"

# =====================================================================================
#  2.  FLUID / SATURATION SET-UP
# =====================================================================================
_AS = AbstractState("HEOS", FLUID)          # reusable CoolProp state object
TCRIT = _AS.T_critical()
TTRIP = _AS.Ttriple()

# Build N2O saturation lookup tables so the inner state solve avoids CoolProp calls.
_T_TAB = np.linspace(TTRIP + 0.5, TCRIT - 0.15, 400)
_PS, _RL, _RV, _UL, _UV, _HL, _HV, _SL, _SV = (np.zeros_like(_T_TAB) for _ in range(9))
for _i, _T in enumerate(_T_TAB):
    _AS.update(CP.QT_INPUTS, 0.0, _T)
    _PS[_i], _RL[_i], _UL[_i], _HL[_i], _SL[_i] = _AS.p(), _AS.rhomass(), _AS.umass(), _AS.hmass(), _AS.smass()
    _AS.update(CP.QT_INPUTS, 1.0, _T)
    _RV[_i], _UV[_i], _HV[_i], _SV[_i] = _AS.rhomass(), _AS.umass(), _AS.hmass(), _AS.smass()


def sat(T):
    """Interpolated N2O saturation properties at temperature T [K]."""
    return (np.interp(T, _T_TAB, _PS), np.interp(T, _T_TAB, _RL), np.interp(T, _T_TAB, _RV),
            np.interp(T, _T_TAB, _UL), np.interp(T, _T_TAB, _UV))


def n2o_Psat(T):
    return float(np.interp(T, _T_TAB, _PS))


# =====================================================================================
#  3.  NETWORK TOPOLOGY  (nodes and branches)
# =====================================================================================
#  Node types: 'reservoir' (tanks) or 'pipe'.
#  Branch kinds: 'friction' (Darcy-Weisbach) or 'orifice' (valve, HEM/gas + Kv + choke).

nodes   = []     # list of dicts: name,type,V,L,D, (rocket: A_cross)
branches = []    # list of dicts describing each connection


def add_node(name, ntype, V, L=0.0, D=0.0, roughness=0.0, A_cross=None):
    nodes.append(dict(name=name, type=ntype, V=V, L=L, D=D,
                      roughness=roughness, A_cross=A_cross))
    return len(nodes) - 1


# --- reserve tank ---
i_reserve = add_node("reserve", "reservoir", RESERVE["V"])

# --- build the alternating chain of zero-volume valves and pipe control volumes ---
prev = i_reserve
for k in range(4):
    p = PIPES[k]
    Vp = np.pi / 4.0 * p["D"] ** 2 * p["L"]
    ip = add_node(f"pipe{k+1}", "pipe", Vp, L=p["L"], D=p["D"],
                  roughness=p["roughness"])
    branches.append(dict(kind="orifice", i=prev, j=ip, valve=k))
    prev = ip

# --- rocket tank ---
A_cross = np.pi / 4.0 * (ROCKET["D_outer"] ** 2 - ROCKET["D_inner"] ** 2)
V_rocket = A_cross * ROCKET["H"]
i_rocket = add_node("rocket", "reservoir", V_rocket, A_cross=A_cross)
branches.append(dict(kind="friction", i=prev, j=i_rocket))

# --- zero-volume vent valve and ambient boundary ---
i_vent = add_node("vent", "reservoir", 1.0e6)
branches.append(dict(kind="orifice", i=i_rocket, j=i_vent, valve=4))

N = len(nodes)

# Pre-compute the pipe resistance assigned to each branch.  A branch receives half
# the length of each adjacent pipe, so every pipe's complete length is represented
# across its two connections.  Reservoir connections add an entrance/exit loss.
for b in branches:
    ni, nj = nodes[b["i"]], nodes[b["j"]]
    # limiting diameter / area
    Ds = [d for d in (ni["D"], nj["D"]) if d > 0]
    D_b = min(Ds) if Ds else 0.01
    b["A"] = np.pi / 4.0 * D_b ** 2
    b["D"] = D_b
    # Each pipe contributes a separate half-length segment so its own diameter
    # and surface roughness are retained in the Darcy calculation.
    L_b = 0.0
    K_minor = 0.0
    segments = []
    for nd in (ni, nj):
        if nd["type"] == "pipe":
            seg_L = 0.5 * nd["L"]
            L_b += seg_L
            segments.append(dict(L=seg_L, D=nd["D"],
                                 A=np.pi / 4.0 * nd["D"] ** 2,
                                 roughness=nd["roughness"]))
        else:
            K_minor += 0.75          # tank entrance/exit loss
    b["L"] = L_b
    b["K_minor"] = K_minor
    b["segments"] = segments


# =====================================================================================
#  4.  STATE INVERSION:  (m_N2O, m_air, U, V)  ->  (T, P, ...)
# =====================================================================================

def brentq(f, a, b, tol=1e-6, maxit=40):
    """Minimal, dependency-light Brent root finder (kept local for portability)."""
    fa, fb = f(a), f(b)
    if fa == 0: return a
    if fb == 0: return b
    if fa * fb > 0:               # no sign change -> return closest end
        return a if abs(fa) < abs(fb) else b
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxit):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2 * np.finfo(float).eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, q = 2 * xm * s, 1 - s
            else:
                q, r = fa / fc, fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0: q = -q
            p = abs(p)
            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)
    return b


def node_state(mN2O, mair, U, V):
    """
    Return dict with T,P,rho,h,u,V_liq,quality,phase for a node holding mN2O kg of
    N2O and mair kg of air in volume V with total internal energy U.
    References: air ideal gas u=cv*T,h=cp*T (0 K ref); N2O from CoolProp.
    """
    mtot = mN2O + mair
    out = dict(mN2O=mN2O, mair=mair, mtot=mtot, V=V, V_liq=0.0, quality=1.0, phase="gas")

    # ---- pure air ----
    if mN2O <= M_FLOOR:
        T = U / (mair * CV_AIR)
        P = mair * R_AIR * T / V
        out.update(T=T, P=P, rho=mair / V, u=U / mtot, h=CP_AIR * T)
        return out

    # ---- pure N2O ----
    if mair <= M_FLOOR:
        rho = mN2O / V
        u = U / mN2O
        try:
            _AS.update(CP.DmassUmass_INPUTS, rho, u)
            T, P, Q, h = _AS.T(), _AS.p(), _AS.Q(), _AS.hmass()
        except Exception:
            T = brentq(lambda TT: _n2o_u_rho(rho, TT) - u, TTRIP + 1, TCRIT - 0.2)
            P, Q, h = _n2o_PQh(rho, T)
        if 0.0 <= Q <= 1.0:
            _, rl, rv, _, _ = sat(T)
            V_liq = mN2O * (1 - Q) / rl
            out.update(V_liq=V_liq, quality=Q, phase="two-phase")
        elif T < TCRIT:
            # CoolProp reports Q=-1 for both compressed liquid and gas.  Density
            # relative to the saturation envelope distinguishes those states.
            _, rl, rv, _, _ = sat(T)
            if rho >= rl:
                out.update(V_liq=V, quality=0.0, phase="liquid")
            elif rho <= rv:
                out.update(quality=1.0, phase="gas")
            else:
                # Defensive fallback for a state numerically just inside the dome.
                Q = (1.0 / rho - 1.0 / rl) / (1.0 / rv - 1.0 / rl)
                Q = float(np.clip(Q, 0.0, 1.0))
                out.update(V_liq=mN2O * (1.0 - Q) / rl,
                           quality=Q, phase="two-phase")
        else:
            out.update(quality=1.0, phase="supercritical")
        out.update(T=T, P=P, rho=rho, u=u, h=h)
        return out

    # ---- mixture: N2O (CoolProp) + air (ideal gas) sharing V, common T ----
    def energy_residual(T):
        return _mix_U(mN2O, mair, T, V)[0] - U

    Tlo, Thi = max(TTRIP + 1.0, 90.0), 1000.0
    try:
        T = brentq(energy_residual, Tlo, Thi, tol=1e-5)
    except Exception:
        T = T_AMBIENT
    Utot, P, V_liq, Q, phase, h = _mix_U(mN2O, mair, T, V, want_props=True)
    out.update(T=T, P=P, rho=mtot / V, u=U / mtot, h=h, V_liq=V_liq, quality=Q, phase=phase)
    return out


def _mix_U(mN2O, mair, T, V, want_props=False):
    """Internal energy of an N2O+air mixture at temperature T in volume V.
       Returns U (and, if want_props, P,V_liq,Q,phase,h_mix)."""
    Psat, rl, rv, ul, uv = sat(T)
    # candidate two-phase liquid mass (vapour fills gas space at rho_v)
    denom = (1.0 - rv / rl)
    mass_liq = (mN2O - rv * V) / denom if denom != 0 else -1.0

    if 0.0 < mass_liq < mN2O and T < TCRIT:
        # --- N2O two-phase, air in gas space ---
        mass_vap = mN2O - mass_liq
        V_liq = mass_liq / rl
        V_gas = max(V - V_liq, 1e-12)
        U_n2o = mass_liq * ul + mass_vap * uv
        P_air = mair * R_AIR * T / V_gas
        U = U_n2o + mair * CV_AIR * T
        if not want_props:
            return U, None
        P = Psat + P_air
        h_l = ul + Psat / rl
        h_v = uv + Psat / rv
        h_air = CP_AIR * T
        h_mix = (mass_liq * h_l + mass_vap * h_v + mair * h_air) / (mN2O + mair)
        Q = mass_vap / mN2O
        return U, P, V_liq, Q, "two-phase", h_mix

    elif mass_liq <= 0.0 or T >= TCRIT:
        # --- N2O all vapour/supercritical, shares full V with air ---
        rho_n = mN2O / V
        try:
            _AS.update(CP.DmassT_INPUTS, rho_n, T)
            Pn, un, hn = _AS.p(), _AS.umass(), _AS.hmass()
        except Exception:
            Pn = rho_n * 188.9 * T          # crude ideal-gas fallback (R_N2O~188.9)
            un = 0.75e3 * T
            hn = un + Pn / rho_n
        P_air = mair * R_AIR * T / V
        U = mN2O * un + mair * CV_AIR * T
        if not want_props:
            return U, None
        P = Pn + P_air
        h_mix = (mN2O * hn + mair * CP_AIR * T) / (mN2O + mair)
        return U, P, 0.0, 1.0, "gas", h_mix

    else:
        # --- N2O essentially all liquid; air squeezed into tiny bubble ---
        V_liq = min(mN2O / rl, V * (1 - 1e-6))
        V_gas = max(V - V_liq, 1e-9)
        U_n2o = mN2O * ul
        P_air = mair * R_AIR * T / V_gas
        U = U_n2o + mair * CV_AIR * T
        if not want_props:
            return U, None
        P = Psat + P_air
        h_l = ul + Psat / rl
        h_mix = (mN2O * h_l + mair * CP_AIR * T) / (mN2O + mair)
        return U, P, V_liq, 0.0, "liquid", h_mix


def _n2o_u_rho(rho, T):
    _AS.update(CP.DmassT_INPUTS, rho, T)
    return _AS.umass()


def _n2o_PQh(rho, T):
    _AS.update(CP.DmassT_INPUTS, rho, T)
    return _AS.p(), _AS.Q(), _AS.hmass()


# =====================================================================================
#  5.  BRANCH FLOW MODELS
# =====================================================================================

def phase_inventory(s):
    """Return the liquid and vapour N2O masses represented by a node state."""
    if s["phase"] == "two-phase":
        mvap = s["mN2O"] * float(np.clip(s["quality"], 0.0, 1.0))
        return s["mN2O"] - mvap, mvap
    if s["phase"] == "liquid":
        return s["mN2O"], 0.0
    return 0.0, s["mN2O"]


def branch_port_mode(node_idx, b):
    """Return the permanently configured outlet type at a tank connection."""
    if node_idx == i_reserve and b["kind"] == "orifice" and b.get("valve") == 0:
        return "bottom"
    if node_idx == i_rocket:
        if b["kind"] == "orifice" and b.get("valve") == 4:
            return "dip_tube"
        # The only other rocket connection is the bottom inlet from pipe 4.
        return "bottom"
    return "bulk"


def outlet_stream(s, node_idx, b):
    """Select the phase, composition and enthalpy withdrawn through a branch port.

    Returns (flow_state, fN2O, fair, h, available_N2O, available_air, label).
    The flow_state is used by the hydraulic model; the remaining values are used
    for conservative species and energy transport.
    """
    mode = branch_port_mode(node_idx, b)
    if mode == "bulk":
        fN = s["mN2O"] / max(s["mtot"], 1e-30)
        return s, fN, 1.0 - fN, s["h"], s["mN2O"], s["mair"], "bulk"

    mliq, mvap = phase_inventory(s)
    if mode == "dip_tube":
        liquid_at_tip = s["V_liq"] / s["V"] >= ROCKET_LIQUID_TARGET
        requested_phase = "liquid" if liquid_at_tip else "gas"
    else:
        requested_phase = "liquid" if mliq > M_FLOOR else "gas"

    Ps, rl, rv, ul, uv = sat(s["T"])
    hl = ul + Ps / rl
    hv = uv + Ps / rv

    if requested_phase == "liquid" and mliq > M_FLOOR:
        flow = dict(s)
        # A homogeneous compressed-liquid node already has the appropriate
        # enthalpy and density; a two-phase node withdraws saturated liquid.
        pure_compressed = s["phase"] == "liquid" and s["mair"] <= M_FLOOR
        h = s["h"] if pure_compressed else hl
        rho = s["mN2O"] / s["V"] if pure_compressed else rl
        flow.update(mN2O=mliq, mair=0.0, mtot=mliq, rho=rho, h=h,
                    phase="liquid", quality=0.0)
        return flow, 1.0, 0.0, h, mliq, 0.0, "liquid"

    # Ullage withdrawal contains all air and only the N2O vapour inventory.
    mgas = mvap + s["mair"]
    if mgas <= M_FLOOR and mliq > M_FLOOR:
        # Defensive fallback for a nominal gas port after the gas inventory has
        # numerically vanished.
        flow = dict(s)
        flow.update(mN2O=mliq, mair=0.0, mtot=mliq, rho=rl, h=hl,
                    phase="liquid", quality=0.0)
        return flow, 1.0, 0.0, hl, mliq, 0.0, "liquid"

    fN = mvap / max(mgas, 1e-30)
    Vgas = max(s["V"] - s["V_liq"], 1e-12)
    h_n2o_gas = hv
    if mvap > M_FLOOR and s["phase"] != "two-phase":
        try:
            _AS.update(CP.DmassT_INPUTS, mvap / Vgas, s["T"])
            h_n2o_gas = _AS.hmass()
        except Exception:
            pass
    h = (mvap * h_n2o_gas + s["mair"] * CP_AIR * s["T"]) / max(mgas, 1e-30)
    flow = dict(s)
    flow.update(mN2O=mvap, mair=s["mair"], mtot=mgas, V=Vgas,
                rho=mgas / Vgas, h=h, phase="gas", quality=1.0)
    return flow, fN, 1.0 - fN, h, mvap, s["mair"], "gas"

def darcy_f(Re, D, roughness):
    """Darcy friction factor via Haaland (turbulent) or laminar 64/Re."""
    if Re < 1.0:
        return 0.0
    if Re < 2300.0:
        return 64.0 / Re
    rr = roughness / D
    return (-1.8 * np.log10((rr / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def friction_flow(b, si, sj):
    """Quasi-steady Darcy-Weisbach mass flow between two nodes (signed, i->j positive)."""
    Pi, Pj = si["P"], sj["P"]
    dP = Pi - Pj
    if abs(dP) < 1.0:
        return 0.0
    up_idx = b["i"] if dP > 0 else b["j"]
    up_bulk = si if dP > 0 else sj
    up = outlet_stream(up_bulk, up_idx, b)[0]
    rho = max(up["rho"], 1e-4)
    A_ref = b["A"]
    segments = b["segments"]
    # Iterate friction factors using the common mass flow.  Segment velocities
    # differ when adjacent pipes have different diameters; their losses are
    # expressed relative to the limiting branch area A_ref.
    mu = 2.0e-5
    fs = [0.02] * len(segments)
    v_ref = 0.0
    for _ in range(3):
        K = b["K_minor"] + 1e-6
        for f, seg in zip(fs, segments):
            area_ratio = A_ref / seg["A"]
            K += f * seg["L"] / max(seg["D"], 1e-6) * area_ratio ** 2
        v_ref = np.sqrt(2.0 * abs(dP) / (rho * K))
        mdot_guess = rho * A_ref * v_ref
        fs = [darcy_f(mdot_guess * seg["D"] / (mu * seg["A"]),
                      seg["D"], seg["roughness"])
              for seg in segments]
    mdot = rho * A_ref * v_ref
    return mdot if dP > 0 else -mdot


def orifice_flow(b, si, sj, valve_pos):
    """
    Valve orifice flow (i->j positive).  Returns (mdot, choked_flag).
    Uses HEM for N2O-dominated upstream, ideal-gas nozzle for air-dominated upstream,
    limited by the valve Kv.  Handles both flow directions.  valve_pos in [0,1]
    scales the effective throat area and Kv (0 = closed, 1 = fully open).
    """
    if valve_pos <= 0.0:
        return 0.0, False
    v = VALVES[b["valve"]]
    A = np.pi / 4.0 * v["d_orifice"] ** 2 * valve_pos
    Cd = v["Cd"]

    Pi, Pj = si["P"], sj["P"]
    if abs(Pi - Pj) < 1.0:
        return 0.0, False
    forward = Pi > Pj
    up_idx = b["i"] if forward else b["j"]
    up_bulk = si if forward else sj
    up = outlet_stream(up_bulk, up_idx, b)[0]
    P_up = up["P"]
    P_dn = sj["P"] if forward else si["P"]
    xN2O = up["mN2O"] / max(up["mtot"], 1e-30)

    if xN2O >= N2O_RICH:
        G, choked = hem_mass_flux(up, P_dn)
    else:
        G, choked = gas_mass_flux(up, P_dn)

    mdot = Cd * A * G
    kv = v.get("Kv", v.get("Cv", 0.0)) * valve_pos
    mdot = min(mdot, kv_limit(kv, abs(P_up - P_dn), up["rho"]))
    # The valve branch also carries the resistance of the adjacent pipe halves.
    # Keep that resistance after removing the artificial valve-section volumes.
    pipe_mdot = abs(friction_flow(b, si, sj))
    if pipe_mdot > 0.0:
        mdot = min(mdot, pipe_mdot)
    mdot = max(mdot, 0.0)
    return (mdot, choked) if forward else (-mdot, choked)


def hem_mass_flux(up, P_dn):
    """Homogeneous Equilibrium Model mass flux [kg/(m^2 s)] with choking detection."""
    T = up["T"]
    # upstream N2O stagnation entropy & enthalpy
    try:
        if up["phase"] == "two-phase":
            _AS.update(CP.QT_INPUTS, min(max(up["quality"], 0.0), 1.0), T)
        elif up["phase"] == "liquid":
            _AS.update(CP.QT_INPUTS, 0.0, T)
        else:
            _AS.update(CP.DmassT_INPUTS, max(up["mN2O"] / up["V"], 1e-6), T)
        s0, h0 = _AS.smass(), _AS.hmass()
        Psat0 = n2o_Psat(T)
        P_up_n2o = Psat0 if up["phase"] in ("two-phase", "liquid") else _AS.p()
    except Exception:
        return 0.0, False

    P_dn = min(P_dn, P_up_n2o - 1.0)
    if P_dn <= 0:
        P_dn = 1.0
    Ps = np.linspace(P_up_n2o, max(P_dn, 1.0e3), HEM_NPTS)
    G = np.zeros_like(Ps)
    for k, Pt in enumerate(Ps):
        try:
            _AS.update(CP.PSmass_INPUTS, Pt, s0)
            rt, ht = _AS.rhomass(), _AS.hmass()
            dh = h0 - ht
            G[k] = rt * np.sqrt(2.0 * dh) if dh > 0 else 0.0
        except Exception:
            G[k] = 0.0
    kmax = int(np.argmax(G))
    # choked = mass flux peaks at a throat pressure strictly above the downstream
    # pressure (critical/back-pressure-independent flow), with a small margin so the
    # flag does not chatter right at the critical point.
    choked = (kmax <= HEM_NPTS - 2) and (G[kmax] > 1.002 * G[-1])
    return float(G[kmax]), bool(choked)


def gas_mass_flux(up, P_dn):
    """Compressible ideal-gas (air) nozzle mass flux per unit area, with choking."""
    P_up = up["P"]
    T = up["T"]
    R = R_AIR
    g = GAMMA_A
    r_crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    ratio = max(P_dn / P_up, 1e-6)
    if ratio <= r_crit:                      # choked
        G = P_up * np.sqrt(g / (R * T) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))
        return G, True
    term = ratio ** (2.0 / g) - ratio ** ((g + 1.0) / g)
    G = P_up * np.sqrt(2.0 * g / ((g - 1.0) * R * T) * max(term, 0.0))
    return G, False


def kv_limit(Kv, dP, rho):
    """Single-phase incompressible mass-flow limit from valve Kv (SI conversion)."""
    if Kv <= 0 or dP <= 0:
        return 0.0
    SG = max(rho / 1000.0, 1e-6)
    Q_m3h = Kv * np.sqrt((dP / 1.0e5) / SG)  # dP in bar
    return rho * Q_m3h / 3600.0


# =====================================================================================
#  6.  VALVE SCHEDULE
# =====================================================================================

def _slew(pos, target, dt, t_actuat):
    """Move pos toward target at constant rate 1/t_actuat over dt seconds."""
    if dt <= 0.0:
        return pos
    if t_actuat <= 0.0:
        return target
    max_delta = dt / t_actuat
    delta = target - pos
    if abs(delta) <= max_delta:
        return target
    return pos + max_delta * (1.0 if delta > 0.0 else -1.0)


def valve_position(vidx, t):
    """Rate-limited valve position in [0,1].  Tracks the commanded (0/1) state
       from VALVE_SCHEDULE, moving at constant rate 1/t_actuat so both
       closed->open and open->closed transitions take t_actuat seconds -- a
       command that reverses direction mid-travel ramps from wherever the
       valve currently is, rather than resetting."""
    t_actuat = VALVES[vidx].get("t_actuat", 0.0)
    sched = VALVE_SCHEDULE[vidx]
    pos = 0.0
    target = 0.0
    t_prev = sched[0][0] if sched else 0.0
    for (ts, s) in sched:
        if ts > t:
            break
        pos = _slew(pos, target, ts - t_prev, t_actuat)
        t_prev = ts
        target = float(s)
    pos = _slew(pos, target, t - t_prev, t_actuat)
    return pos


def valve_is_open(vidx, t):
    return valve_position(vidx, t) > 0.0


# =====================================================================================
#  7.  INITIAL CONDITIONS
# =====================================================================================

mN2O = np.zeros(N)
mair = np.zeros(N)
U    = np.zeros(N)

# Reserve bottle: its measured volume, N2O mass and temperature uniquely define
# density and specific internal energy.  Pressure and phase are EOS outputs, not
# independent inputs.
Tr = float(RESERVE["T"])
Vr = float(RESERVE["V"])
mr = float(RESERVE["mass"])
if Vr <= 0.0 or mr <= 0.0 or Tr <= 0.0:
    raise ValueError("RESERVE V, mass and T must all be positive")
try:
    _AS.update(CP.DmassT_INPUTS, mr / Vr, Tr)
    ur = _AS.umass()
except Exception as exc:
    raise ValueError(
        f"Invalid reserve state: V={Vr:g} m^3, mass={mr:g} kg, T={Tr:g} K"
    ) from exc
mN2O[i_reserve] = mr
mair[i_reserve] = 0.0
U[i_reserve]    = mr * ur

# every other node: air at P_ATM, ambient temperature
for idx, nd in enumerate(nodes):
    if idx == i_reserve:
        continue
    T0 = ROCKET["T"] if idx == i_rocket else T_AMBIENT
    rho_a = P_ATM / (R_AIR * T0)
    mair[idx] = rho_a * nd["V"]
    mN2O[idx] = 0.0
    U[idx]    = mair[idx] * CV_AIR * T0

reserve_initial = node_state(mN2O[i_reserve], mair[i_reserve], U[i_reserve], Vr)
print("Initial reserve state: "
      f"{reserve_initial['P']/1e5:.3f} bar, {reserve_initial['T']:.2f} K, "
      f"rho={mr/Vr:.2f} kg/m^3, phase={reserve_initial['phase']}, "
      f"liquid volume={100.0*reserve_initial['V_liq']/Vr:.2f}%")

# =====================================================================================
#  8.  TIME INTEGRATION  (adaptive embedded Euler/Heun, explicit)
# =====================================================================================

def compute_hydraulics(mN2O_v, mair_v, U_v, t):
    """Evaluate node states and per-branch target flows.  Depends only on the state
       and t, NOT on the trial step size, so it's safe/cheap to reuse across the
       step-size controller's retries and across accepted steps (it's the expensive,
       CoolProp/HEM-flash-bearing half of the derivative evaluation)."""
    states = [node_state(mN2O_v[k], mair_v[k], U_v[k], nodes[k]["V"]) for k in range(N)]
    binfo = []
    for b in branches:
        i, j = b["i"], b["j"]
        si, sj = states[i], states[j]
        if b["kind"] == "friction":
            mdot_target, choked, valve_open = friction_flow(b, si, sj), False, True
        else:
            valve_pos = valve_position(b["valve"], t)
            valve_open = valve_pos > 0.0
            mdot_target, choked = orifice_flow(b, si, sj, valve_pos)

        if b["kind"] == "orifice" and not valve_open:
            binfo.append(dict(mdot_target=0.0, choked=False, valve_open=False, fwd=None, bwd=None))
            continue

        # Pre-compute the outlet composition for BOTH possible flow directions so the
        # FLOW_TAU-filtered flow (only known once dt is chosen, in assemble_derivative)
        # can pick whichever port its own sign indicates -- same as the single-pass
        # original code, which always resolved the port after filtering.
        _, fNf, fAf, hf, availNf, availAf, _ = outlet_stream(si, i, b)
        _, fNb, fAb, hb, availNb, availAb, _ = outlet_stream(sj, j, b)
        binfo.append(dict(
            mdot_target=mdot_target, choked=choked, valve_open=valve_open,
            fwd=dict(fN=fNf, fA=fAf, h=hf, availN=availNf, availA=availAf),
            bwd=dict(fN=fNb, fA=fAb, h=hb, availN=availNb, availA=availAb),
        ))
    return states, binfo


def assemble_derivative(binfo, mdot_prev, dt):
    """Apply the FLOW_TAU relaxation to each branch's target flow at trial step dt and
       assemble the state derivative.  Cheap arithmetic only -- safe to redo per retry."""
    dmN2O = np.zeros(N)
    dmair = np.zeros(N)
    dU    = np.zeros(N)
    valve_mdot = [0.0] * 5
    mdot_new = np.zeros(len(branches))

    for bi, b in enumerate(branches):
        info = binfo[bi]
        if b["kind"] == "orifice" and not info["valve_open"]:
            mdot_new[bi] = 0.0
            valve_mdot[b["valve"]] = 0.0
            continue

        mdot_target = info["mdot_target"]
        if FLOW_TAU > 0.0:
            alpha = dt / (FLOW_TAU + dt)
            mdot = mdot_prev[bi] + alpha * (mdot_target - mdot_prev[bi])
        else:
            mdot = mdot_target
        mdot_new[bi] = mdot

        if mdot == 0.0:
            if b["kind"] == "orifice":
                valve_mdot[b["valve"]] = 0.0
            continue

        port = info["fwd"] if mdot > 0.0 else info["bwd"]
        sgn = 1.0 if mdot > 0.0 else -1.0
        m_up = abs(mdot)
        mN = m_up * port["fN"]
        mA = m_up * port["fA"]
        i, j = b["i"], b["j"]
        dmN2O[i] -= sgn * mN
        dmN2O[j] += sgn * mN
        dmair[i] -= sgn * mA
        dmair[j] += sgn * mA
        dU[i]    -= sgn * m_up * port["h"]
        dU[j]    += sgn * m_up * port["h"]
        if b["kind"] == "orifice":
            valve_mdot[b["valve"]] = sgn * m_up

    return dmN2O, dmair, dU, mdot_new, valve_mdot


def maxdrain_dt_cap(binfo):
    """Largest dt such that no branch would drain more than MAX_DRAIN of its upstream
       port's available species mass, using the (dt-independent) target flow rate as
       the rate estimate.  A step-size bound, not a post-hoc flow clamp -- clamping the
       flow after the fact would silently invalidate the Euler/Heun error estimate."""
    cap = np.inf
    for info in binfo:
        mdot_target = info["mdot_target"]
        if mdot_target == 0.0 or info["fwd"] is None:
            continue
        port = info["fwd"] if mdot_target > 0.0 else info["bwd"]
        rate = abs(mdot_target)
        if port["fN"] > 0.0 and port["availN"] > 0.0:
            cap = min(cap, MAX_DRAIN * port["availN"] / (port["fN"] * rate))
        if port["fA"] > 0.0 and port["availA"] > 0.0:
            cap = min(cap, MAX_DRAIN * port["availA"] / (port["fA"] * rate))
    return cap


def next_event_time(t, fill_reach_time):
    """Nearest future instant a step must not step over: valve-schedule changes
       (including ones appended dynamically by STOP_ON_ROCKET_FILL), the instant
       each resulting position ramp saturates (exact when a ramp starts from the
       opposite extreme; approximate otherwise, which the error controller still
       handles correctly), the post-fill stop, and the run's final time."""
    candidates = [T_TOTAL]
    for vidx, sched in enumerate(VALVE_SCHEDULE):
        t_actuat = VALVES[vidx].get("t_actuat", 0.0)
        for ts, _ in sched:
            if ts > t + 1e-12:
                candidates.append(ts)
            if t_actuat > 0.0 and ts + t_actuat > t + 1e-12:
                candidates.append(ts + t_actuat)
    if fill_reach_time is not None:
        candidates.append(fill_reach_time + 10.0)
    return min(candidates)


def error_norm(mN2O1, mair1, U1, mN2O2, mair2, U2):
    """Weighted-RMS local error between the Euler (1) and Heun (2) estimates."""
    y1 = np.concatenate([mN2O1, mair1, U1])
    y2 = np.concatenate([mN2O2, mair2, U2])
    atol = np.concatenate([np.full(N, ATOL_M), np.full(N, ATOL_M), np.full(N, ATOL_U)])
    scale = atol + RTOL * np.maximum(np.abs(y1), np.abs(y2))
    return float(np.sqrt(np.mean(((y2 - y1) / scale) ** 2)))


def _step_factor(err, lo, hi):
    return float(np.clip(SAFETY * max(err, 1e-12) ** -0.5, lo, hi))


print(f"Nodes: {N} | Branches: {len(branches)}")
print("Running adaptive Euler/Heun integration ...")

# recording arrays
rec_t = []
rec_P = []          # [step][node]
rec_T = []
rec_level = []
rec_reserve_mass = []
rec_rocket_mass = []
rec_mdot_valve = [[] for _ in range(5)]
choke_events = []            # (t, valve_idx) onset events
choke_active = [False] * 5   # current choke state per valve (for edge detection)
choke_time   = [0.0] * 5     # accumulated simulated time each valve was choked
choke_first  = [None] * 5    # first time each valve choked
branch_mdot_prev = np.zeros(len(branches))  # FLOW_TAU-filtered flow, carried between steps


def record(t, states, valve_mdot):
    rec_t.append(t)
    rec_P.append([s["P"] for s in states])
    rec_T.append([s["T"] for s in states])
    rec_level.append(states[i_rocket]["V_liq"] / nodes[i_rocket]["A_cross"])
    rec_reserve_mass.append(mN2O[i_reserve])
    rec_rocket_mass.append(mN2O[i_rocket])
    for vi in range(5):
        rec_mdot_valve[vi].append(valve_mdot[vi])


t = 0.0
dt_next = DT_INIT
fill_reach_time = None
_sustain_time = 0.0
accepted_steps = 0
rejected_steps = 0

states0, binfo0 = compute_hydraulics(mN2O, mair, U, t)
record(t, states0, [0.0] * 5)

while t < T_TOTAL - 1e-9:
    t_event = next_event_time(t, fill_reach_time)
    remaining = max(t_event - t, 1e-12)
    dt_floor = min(DT_MIN, remaining)
    dt_trial = min(dt_next, DT_MAX, maxdrain_dt_cap(binfo0), remaining)
    dt_trial = max(dt_trial, dt_floor)

    attempt = 0
    err = 0.0
    while True:
        dmN2O1, dmair1, dU1, mdotfilt1, valve_mdot1 = assemble_derivative(binfo0, branch_mdot_prev, dt_trial)
        mN2O_e = np.clip(mN2O + dt_trial * dmN2O1, M_FLOOR, None)
        mair_e = np.clip(mair + dt_trial * dmair1, M_FLOOR, None)
        U_e    = U + dt_trial * dU1

        states1, binfo1 = compute_hydraulics(mN2O_e, mair_e, U_e, t + dt_trial)
        dmN2O2, dmair2, dU2, mdotfilt2, valve_mdot2 = assemble_derivative(binfo1, branch_mdot_prev, dt_trial)

        mN2O_h = mN2O + 0.5 * dt_trial * (dmN2O1 + dmN2O2)
        mair_h = mair + 0.5 * dt_trial * (dmair1 + dmair2)
        U_h    = U + 0.5 * dt_trial * (dU1 + dU2)

        err = error_norm(mN2O_e, mair_e, U_e, mN2O_h, mair_h, U_h)
        converged = err <= 1.0 or dt_trial <= dt_floor * (1.0 + 1e-9)
        if converged or attempt >= MAX_RETRIES:
            if not converged:
                print(f"\n*** step-size controller could not converge at t={t:.6f}s after "
                      f"{MAX_RETRIES} retries (err={err:.3g}). Continuing with the last trial step. ***")
            break
        rejected_steps += 1
        attempt += 1
        dt_trial = max(dt_trial * _step_factor(err, SHRINK_MIN, 0.9), dt_floor)

    mN2O = np.clip(mN2O_h, M_FLOOR, None)
    mair = np.clip(mair_h, M_FLOOR, None)
    U    = U_h

    if not (np.all(np.isfinite(mN2O)) and np.all(np.isfinite(mair)) and np.all(np.isfinite(U))):
        print(f"\n*** NUMERICAL INSTABILITY at t={t:.5f}s. Tighten RTOL/ATOL_* or lower DT_MAX. ***")
        break

    branch_mdot_prev = mdotfilt1
    accepted_steps += 1

    # ---- choke bookkeeping, sampled once from k1 (the state at the START of this step) ----
    for bi, b in enumerate(branches):
        if b["kind"] != "orifice":
            continue
        vidx = b["valve"]
        info = binfo0[bi]
        if info["choked"]:
            choke_time[vidx] += dt_trial
            if choke_first[vidx] is None:
                choke_first[vidx] = t
                si, sj = states0[b["i"]], states0[b["j"]]
                print(f"  [CHOKED]  valve{vidx+1} first choked at t = {t:8.4f} s "
                      f"(P_up={si['P']/1e5:6.2f} bar -> P_dn={sj['P']/1e5:6.2f} bar)")
            if not choke_active[vidx]:
                choke_events.append((t, vidx))
                choke_active[vidx] = True
        else:
            choke_active[vidx] = False

    t += dt_trial
    # committed-state hydraulics: recorded now, and reused as next iteration's k1
    states0, binfo0 = compute_hydraulics(mN2O, mair, U, t)
    record(t, states0, valve_mdot1)

    if PRINT_EVERY and (accepted_steps % max(1, PRINT_EVERY) == 0):
        print(f"\rSim time: {t:8.4f} s  (dt={dt_trial:.2e} s, accepted={accepted_steps}, "
              f"rejected={rejected_steps})", end='', flush=True)

    dt_next = np.clip(dt_trial * _step_factor(err, SHRINK_MIN, GROW_MAX), DT_MIN, DT_MAX)

    # Check whether the liquid has reached the dip-tube tip for 0.5 s.  The
    # annular tank has constant cross-sectional area, so height and volume
    # fractions are identical.
    if STOP_ON_ROCKET_FILL:
        rocket_liq_frac = states0[i_rocket]["V_liq"] / nodes[i_rocket]["V"]
        if rocket_liq_frac >= ROCKET_LIQUID_TARGET:
            _sustain_time += dt_trial
        else:
            _sustain_time = 0.0
        if _sustain_time >= 0.5 and fill_reach_time is None:
            # Close valves 1-4 after the dip-tube level is sustained for 0.5s.
            print(f"\nRocket liquid reached the dip-tube tip "
                  f"({ROCKET_LIQUID_TARGET*100:.1f}% liquid, "
                  f"{ROCKET_ULLAGE_TARGET*100:.1f}% ullage) for "
                  f"{_sustain_time:.3f} s at t={t:.4f}s")
            print("Closing valves 1-4, continuing simulation for 10 more seconds...")
            # Dynamically add close events for valves 1-4 at current time
            for vi in range(4):
                VALVE_SCHEDULE[vi].append((t, 0))
            fill_reach_time = t
            _sustain_time = 0.0

    # Stop 10 seconds after rocket fill target is reached
    if fill_reach_time is not None and t >= fill_reach_time + 10.0 - 1e-9:
        print(f"Stopping: 10 seconds elapsed after fill target reached at t={t:.4f}s")
        break

print()
print(f"Integration finished. Accepted steps: {accepted_steps}, rejected attempts: {rejected_steps}.")

# =====================================================================================
#  9.  RESULTS  &  PLOTS
# =====================================================================================

rec_t = np.array(rec_t)
rec_P = np.array(rec_P)          # [nt, N]
rec_T = np.array(rec_T)
rec_level = np.array(rec_level)
rec_reserve_mass = np.array(rec_reserve_mass)
rec_rocket_mass = np.array(rec_rocket_mass)

name = [nd["name"] for nd in nodes]
plot_nodes = ["reserve"] + [f"pipe{k+1}" for k in range(4)] + ["rocket"]
plot_idx = [name.index(n) for n in plot_nodes]

# ---- choke summary ----
print("\n================ CHOKED-FLOW SUMMARY ================")
total_time = rec_t[-1] if len(rec_t) else 0.0
if not any(choke_first):
    print("No choked flow detected at any valve.")
else:
    for vi in range(5):
        if choke_first[vi] is not None:
            frac = 100.0 * choke_time[vi] / max(total_time, 1e-9)
            print(f"  valve{vi+1}: first choked at t={choke_first[vi]:.4f} s, "
                  f"choked ~{frac:.1f}% of the run (by time)")
        else:
            print(f"  valve{vi+1}: never choked")
print("====================================================\n")

# ---- final-state table ----
print("Final node states (t = %.3f s):" % rec_t[-1])
print(f"{'node':>9} {'P [bar]':>10} {'T [K]':>9} {'phase':>11}")
final_states = [node_state(mN2O[k], mair[k], U[k], nodes[k]["V"]) for k in range(N)]
for k in plot_idx:
    s = final_states[k]
    print(f"{name[k]:>9} {s['P']/1e5:10.3f} {s['T']:9.2f} {s['phase']:>11}")
print(f"\nRocket-tank liquid level: {rec_level[-1]*1000:.1f} mm "
      f"({rec_level[-1]/ROCKET['H']*100:.1f} % of height)")
print(f"N2O delivered to rocket tank: {mN2O[i_rocket]:.4f} kg")

# ---- Figure 1: pressure dashboard ----
fig1, axs1 = plt.subplots(len(plot_idx), 1, figsize=(10, 13), sharex=True)
for ax, k in zip(axs1, plot_idx):
    ax.plot(rec_t, rec_P[:, k] / 1e5, color=f"C{plot_idx.index(k)}")
    ax.set_ylabel("pressure [bar]")
    ax.set_title(name[k])
    ax.grid(True, alpha=0.3)
axs1[-1].set_xlabel("time [s]")
fig1.suptitle("Pressure in pipes and rocket tank")
fig1.tight_layout(rect=[0, 0, 1, 0.98])
fig1.savefig(OUTPUT_DIR / "fill_pressures.png", dpi=130)

# ---- Figure 2: temperature dashboard ----
fig2, axs2 = plt.subplots(len(plot_idx), 1, figsize=(10, 13), sharex=True)
for ax, k in zip(axs2, plot_idx):
    ax.plot(rec_t, rec_T[:, k], color=f"C{plot_idx.index(k)}")
    ax.set_ylabel("temperature [K]")
    ax.set_title(name[k])
    ax.grid(True, alpha=0.3)
axs2[-1].set_xlabel("time [s]")
fig2.suptitle("Temperature in pipes and rocket tank")
fig2.tight_layout(rect=[0, 0, 1, 0.98])
fig2.savefig(OUTPUT_DIR / "fill_temperatures.png", dpi=130)

# ---- Figure 3: liquid-level and N2O-inventory dashboard ----
fig3, axs3 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
ax3_level, ax3_reserve, ax3_rocket = axs3
ax3_level.plot(rec_t, rec_level * 1000.0, color="tab:blue")
target_level = ROCKET_LIQUID_TARGET * ROCKET["H"]
ax3_level.axhline(target_level * 1000.0, color="tab:red", linestyle="--",
                  label=f"dip-tube tip ({ROCKET_ULLAGE_TARGET*100:.1f}% ullage)")
ax3_level.set_ylabel("liquid level [mm]")
ax3_level.set_title("Rocket-tank N2O liquid level")
ax3_level.grid(True, alpha=0.3)
ax3_level.legend()

ax3_reserve.plot(rec_t, rec_reserve_mass, color="tab:orange")
ax3_reserve.set_ylabel("N2O mass [kg]")
ax3_reserve.set_title("Supply-bottle N2O mass")
ax3_reserve.grid(True, alpha=0.3)

ax3_rocket.plot(rec_t, rec_rocket_mass, color="tab:green")
ax3_rocket.set_xlabel("time [s]")
ax3_rocket.set_ylabel("N2O mass [kg]")
ax3_rocket.set_title("Rocket-tank N2O mass")
ax3_rocket.grid(True, alpha=0.3)

fig3.suptitle("N2O fill level and tank inventories")
fig3.tight_layout(rect=[0, 0, 1, 0.97])
fig3.savefig(OUTPUT_DIR / "fill_level.png", dpi=130)

# ---- Figure 4: valve mass flows ----
rec_mdot_valve = [np.asarray(mdot_series, dtype=float) for mdot_series in rec_mdot_valve]
fig4, axs4 = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
for vi, ax in enumerate(axs4):
    mdot_arr = rec_mdot_valve[vi] * 1000.0
    if mdot_arr.size == 0:
        continue
    ax.plot(rec_t[:mdot_arr.size], mdot_arr, color=f"C{vi}")
    ax.set_ylabel("mass flow [g/s]")
    ax.set_title(f"Valve {vi+1} mass flow")
    ax.grid(True, alpha=0.3)
axs4[-1].set_xlabel("time [s]")
fig4.tight_layout(rect=[0, 0, 1, 0.96])
fig4.savefig(OUTPUT_DIR / "fill_valveflow.png", dpi=130)

# ---- Figure 5: valve-position dashboard (0=closed, 1=open) ----
rec_valve_pos = np.array([[valve_position(vi, t) for t in rec_t] for vi in range(5)])
fig5, axs5 = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
for vi, ax in enumerate(axs5):
    ax.plot(rec_t, rec_valve_pos[vi, :], color=f"C{vi}")
    ax.set_ylabel("position")
    ax.set_title(f"Valve {vi+1}")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
axs5[-1].set_xlabel("time [s]")
fig5.suptitle("Valve open/closed schedule (1=open, 0=closed)")
fig5.tight_layout(rect=[0, 0, 1, 0.97])
fig5.savefig(OUTPUT_DIR / "fill_valvepos.png", dpi=130)

plt.show()
print(f"\nSaved plots in: {OUTPUT_DIR}")
