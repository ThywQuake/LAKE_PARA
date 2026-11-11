"""Build a probe-instrumented copy of the lakebatch Fortran source.

The copy goes to `lakebatch/scratch/probe-build/` (or --out).  It contains
`source/model/lm_probe.f90` (from tools/ref/lm_probe.f90) and probes at the
entry and exit of `SUBROUTINE LAKE`:

  LAKE_entry : after UPDATE_CURRENT_TIMESTEP (the state the step starts from)
  LAKE_exit  : before UPDATE_NEXT_TIMESTEP   (the state the step produced)

Run with LM_PROBE_STEPS=2,180,240 (comma separated) to write
`probe/<step>/<tag>.bin` in the run directory.  With LM_PROBE_STEPS unset the
probe build must produce byte-identical output to the check build; that is the
instrumentation self-check.

usage: python3 tools/ref/make_probe_build.py [--out DIR] [--build] [--parallel N]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAKEBATCH = ROOT / "lakebatch"
SRC = LAKEBATCH / "core" / "source"
TEMPLATE = Path(__file__).with_name("lm_probe.f90")

ENTRY_VARS = [
    "Tw1", "Ti1", "Tis1", "Tsoil1", "Sal1", "h1", "l1", "hs1", "ls1", "snowmass",
    "ddz", "ddz05", "dzeta_int", "z_full", "zsoil", "zsoilcols(1:nsoilcols+1,ix,iy)",
    "tempair", "humair", "pressure", "uwind", "vwind", "precip",
    "longwave", "shortwave", "cloud", "netrad", "time", "nstep",
    "lamw", "E1", "eps1", "k2", "kt", "dhwfsoil", "dhw", "dhw0",
]


def call_end(lines, start):
    """Index of the last non-blank line of the statement starting at `start`.

    A trailing `&` continues the statement over blank and comment lines (the
    UPDATE_CURRENT_TIMESTEP call has a blank line inside its continuation).
    """
    i = start
    last = start
    while i < len(lines):
        cur = lines[i].rstrip()
        if not cur.strip() or cur.strip().startswith("!"):
            i += 1
            continue
        last = i
        if not cur.endswith("&"):
            break
        i += 1
    return last


def patch_lake(path):
    lines = path.read_text().splitlines()
    out = []
    used = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if not used and line.lstrip().lower().startswith("use lb_point_ctx"):
            end = call_end(lines, i)
            out.extend(lines[i:end + 1])
            out.append("use LM_PROBE_MOD, only : LM_PROBE")
            used = True
            i = end + 1
            continue
        if "call UPDATE_CURRENT_TIMESTEP" in line:
            end = call_end(lines, i)
            out.append('  call LM_PROBE("PRE_LOAD", "Tw1", Tw1)')
            out.append('  call LM_PROBE("PRE_LOAD", "z_full", z_full)')
            out.append('  call LM_PROBE("PRE_LOAD", "ip_zT", ip%zTinitprof)')
            out.append('  call LM_PROBE("PRE_LOAD", "ip_T", ip%Tinitprof)')
            out.extend(lines[i:end + 1])
            out.extend(f'  call LM_PROBE("LAKE_entry", "{v}", {v})' for v in ENTRY_VARS)
            out.append('  call LM_PROBE("POST_LOAD", "Tw1", Tw1)')
            i = end + 1
            continue
        if "call UPDATE_NEXT_TIMESTEP" in line:
            out.extend(f'  call LM_PROBE("LAKE_exit", "{v}", {v})' for v in ENTRY_VARS)
            out.append(line)
            i += 1
            continue
        out.append(line)
        i += 1
    if not used:
        raise SystemExit("could not find 'use LB_POINT_CTX' in lake.f90")
    path.write_text("\n".join(out) + "\n")




def patch_init_var(path):
    s = path.read_text()
    if "LM_PROBE_MOD" not in s:
        s = s.replace("  use GRID_OPERATIONS_LAKE, only : LININTERPOL",
                      "  use GRID_OPERATIONS_LAKE, only : LININTERPOL\n  use LM_PROBE_MOD, only : LM_PROBE")
    s = s.replace("& z_full,Tw1,M+1,flag,.true.)",
                  "& z_full,Tw1,M+1,flag,.true.)\n  call LM_PROBE(\"INITVAR\", \"Tw1\", Tw1)", 1)
    s = s.replace("& z_full,Sal1,M+1,flag,.true.)",
                  "& z_full,Sal1,M+1,flag,.true.)\n  call LM_PROBE(\"INITVAR\", \"Sal1\", Sal1)", 1)
    s = s.replace("  deallocate(pressoil)",
                  "  call LM_PROBE(\"INITVAR\", \"Tsoil1\", Tsoil1)\n"
                  "  call LM_PROBE(\"INITVAR\", \"h1\", h1)\n"
                  "  call LM_PROBE(\"INITVAR\", \"l1\", l1)\n"
                  "  call LM_PROBE(\"INITVAR\", \"hs1\", hs1)\n"
                  "  call LM_PROBE(\"INITVAR\", \"ls1\", ls1)\n"
                  "  call LM_PROBE(\"INITVAR\", \"snowmass\", snowmass)\n"
                  "  deallocate(pressoil)", 1)
    path.write_text(s)



def patch_density(path):
    lines = path.read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        if "call DENSITY_W" in lines[i]:
            tag = f"DENS_{i + 1}"
            out.append(f'  call LM_PROBE("{tag}_in", "Tw1", Tw1)')
            out.append(f'  call LM_PROBE("{tag}_in", "Sal1", Sal1)')
            out.append(f'  call LM_PROBE("{tag}_in", "preswat", preswat)')
            out.append(lines[i])
            out.append(f'  call LM_PROBE("{tag}_out", "row", row)')
            i += 1
            continue
        out.append(lines[i])
        i += 1
    path.write_text("\n".join(out) + "\n")



def patch_bathym(path):
    lines = path.read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        if "call BATHYMETRY" in lines[i]:
            tag = f"BATHY_{i + 1}"
            end = call_end(lines, i)
            out.extend(lines[i:end + 1])
            for name in ("z_full", "preswat", "area_int", "area_half", "vol", "botar"):
                out.append(f'  call LM_PROBE("{tag}_out", "{name}", {name})')
            out.append(f'  call LM_PROBE("{tag}_out", "Lx", bathymwater(1:M+1)%Lx)')
            out.append(f'  call LM_PROBE("{tag}_out", "Ly", bathymwater(1:M+1)%Ly)')
            out.append(f'  call LM_PROBE("{tag}_out", "rad_int", bathymwater(1:M+1)%rad_int)')
            i = end + 1
            continue
        out.append(lines[i])
        i += 1
    path.write_text("\n".join(out) + "\n")



def patch_sfcflx(path):
    s = path.read_text()
    start = s.index("SUBROUTINE SfcFlx_momsenlat")
    imp = s.index("IMPLICIT NONE", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    anchor = "wvpres_s = SfcFlx_satwvpres"
    probes = "".join(f'  call LM_PROBE("SFC_in", "{n}", {n})\n' for n in
                     ("height_u", "height_tq", "fetch", "U_a", "T_a", "q_a",
                      "T_s", "P_a", "h_ice"))
    s = s.replace(anchor, probes + anchor, 1)
    end = s.index("END SUBROUTINE SfcFlx_momsenlat")
    outs = "".join(f'  call LM_PROBE("SFC_out", "{n}", {n})\n' for n in
                   ("Q_momentum", "Q_sensible", "Q_latent", "Q_watvap"))
    s = s[:end] + outs + s[end:]
    path.write_text(s)



def patch_heatbalance(path):
    s = path.read_text()
    start = s.index("FUNCTION HEATBALANCE")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    first = s.index("if     (skin%par == 1", start)
    s = s[:first] + ('  call LM_PROBE("HB_in", "Tsurf", Tsurf)\n'
                     '  call LM_PROBE("HB_in", "surftyp", surftyp)\n') + s[first:]
    end = s.index("END FUNCTION HEATBALANCE")
    ret = s.rindex("RETURN", 0, end)
    probes = "".join(f'  call LM_PROBE("HB_out", "{n}", {n})\n' for n in
                     ("hflux", "Elatent", "Radbal", "Radbal_surf", "eflux",
                      "dHdt", "convect_flux", "eflux0", "eflux0_kinem", "HEATBALANCE"))
    s = s[:ret] + probes + s[ret:]
    path.write_text(s)


def patch_senslatmom(path):
    s = path.read_text()
    start = s.index("SUBROUTINE SENSLATMOM_FLUXES")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    first = s.index("if (relwind%par == 1)", start)
    ins = "".join(f'  call LM_PROBE("SLM_in", "{n}", {n})\n' for n in
                  ("Tsurf", "fetch", "uwind", "vwind", "u", "v", "tempair",
                   "pressure", "humair", "zref", "roughness", "relhums",
                   "aM", "bM", "l1", "h1", "relwind%par", "PBLpar%par"))
    s = s[:first] + ins + s[first:]
    end = s.index("END SUBROUTINE SENSLATMOM_FLUXES")
    ret = s.rindex("RETURN", 0, end)
    probes = "".join(f'  call LM_PROBE("SLM_out", "{n}", {n})\n' for n in
                     ("hflux", "Elatent", "tau", "cdmw", "xlew", "velfrict",
                      "wind10", "ro", "humsurf", "wr", "TET2", "TET1"))
    s = s[:ret] + probes + s[ret:]
    path.write_text(s)



def patch_heatfluxsurf(path):
    s = path.read_text()
    start = s.index("SUBROUTINE HEATFLUXSURF")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    first = s.index("SELECT CASE (surftyp)", start)
    ins = "".join(f'  call LM_PROBE("HFS_in", "{n}", {n})\n' for n in
                  ("Tw1", "Tw2", "lamw", "h1", "ddz", "dhw0", "dt",
                   "bathymwater(1)%area_int", "bathymwater(1)%area_half",
                   "bathymwater(M+1)%area_int", "bathymwater(M)%area_half"))
    s = s[:first] + ins + s[first:]
    end = s.index("END SUBROUTINE HEATFLUXSURF")
    ret = s.rindex("RETURN", 0, end)
    outs = "".join(f'  call LM_PROBE("HFS_out", "{n}", {n})\n' for n in
                   ("eflux", "dHdt", "convect_flux", "botflux"))
    s = s[:ret] + outs + s[ret:]
    path.write_text(s)


def patch_radbalance(path):
    s = path.read_text()
    start = s.index("SUBROUTINE RADBALANCE")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    first = s.index("if     (ifrad%par == 1)", start)
    ins = "".join(f'  call LM_PROBE("RB_in", "{n}", {n})\n' for n in
                  ("Tsurf", "surftyp", "albedo", "albedo_lw", "emissivity",
                   "longwave", "shortwave", "netrad", "cloud", "humair",
                   "pressure", "tempair"))
    s = s[:first] + ins + ('  call LM_PROBE("RB_in", "RadWater_integr", RadWater%integr(0:1))\n'
                           '  call LM_PROBE("RB_in", "RadIce_integr", RadIce%integr(0:1))\n') + s[first:]
    end = s.index("END SUBROUTINE RADBALANCE")
    ret = s.rindex("RETURN", 0, end)
    outs = "".join(f'  call LM_PROBE("RB_out", "{n}", {n})\n' for n in
                   ("Radbal", "Radbal_surf", "ShortBal_toplayer", "surfrad", "swup"))
    s = s[:ret] + outs + s[ret:]
    path.write_text(s)



def patch_surf_char(path):
    s = path.read_text()
    start = s.index("SUBROUTINE SURF_CHAR")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    anchor = "select case (surftyp)"
    first = s.index(anchor, start)
    ins = "".join(f'  call LM_PROBE("SC_in", "{n}", {n})\n' for n in
                  ("surftyp", "year", "month", "day", "hour", "phi", "fetch",
                   "wind", "velfrict_prev", "varalb%par"))
    s = s[:first] + ins + s[first:]
    end = s.index("END SUBROUTINE SURF_CHAR")
    outs = "".join(f'  call LM_PROBE("SC_out", "{n}", {n})\n' for n in
                   ("albedo", "albedo_lw", "roughness", "aM", "bM",
                    "emissivity", "relhums", "velfrict_prev"))
    s = s[:end] + outs + s[end:]
    path.write_text(s)



def patch_tdiff(path):
    s = path.read_text()
    start = s.index("SUBROUTINE T_DIFF")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    anchor = "  elseif (water == 1) then  ! There is no snow and ice, but water exists"
    pos = s.index(anchor) + len(anchor)
    ins = "".join(f'  call LM_PROBE("TD_in", "{n}", {n})\n' for n in
                  ("Tsurf", "dt", "Tw1", "Tsoil1", "lamw", "area_int",
                   "area_half", "h1", "ddz", "dhw", "dhw0", "PEMF",
                   "pt_down_f", "lamsoil", "csoil", "rosoil", "dzs"))
    s = s[:pos] + ('\n  call LM_PROBE("TD_in", "RadWater_integr", RadWater%integr)\n'
                   '  call LM_PROBE("TD_in", "lsh_water", lsh%water(1:M+1))\n') + ins + s[pos:]
    # coefficient probe at the water+soil solve
    key = "call PROGONKA (nveclen,a,b,c,d,Temp,1,M+ns)"
    pos = s.index(key) + len(key)
    s = s[:pos] + ('\n  call LM_PROBE("TD2", "a", a(1:M+ns))\n'
                   '  call LM_PROBE("TD2", "b", b(1:M+ns))\n'
                   '  call LM_PROBE("TD2", "c", c(1:M+ns))\n'
                   '  call LM_PROBE("TD2", "d", d(1:M+ns))\n'
                   '  call LM_PROBE("TD2", "Temp", Temp(1:M+ns))\n') + s[pos:]
    anchor2 = "  else ! There is no snow, ice and water: only soil exists"
    pos2 = s.index(anchor2)
    outs = ('  call LM_PROBE("TD_out", "Tw2", Tw2)\n'
            '  call LM_PROBE("TD_out", "Tsoil2", Tsoil2)\n')
    s = s[:pos2] + outs + s[pos2:]
    path.write_text(s)



def patch_t_solver(path):
    s = path.read_text()
    start = s.index("SUBROUTINE T_SOLVER")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    end = s.index("END SUBROUTINE T_SOLVER")
    outs = "".join(f'  call LM_PROBE("TS_out", "{n}", {n})\n' for n in
                   ("Tsurf3", "iter", "Bal3"))
    outs += '  call LM_PROBE("TS_out", "surftyp", surftyp(ix,iy))\n'
    s = s[:end] + outs + s[end:]
    path.write_text(s)



def patch_turb_scales(path):
    s = path.read_text()
    start = s.index("SUBROUTINE TURB_SCALES")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    anchor = "  call MIXED_LAYER_CALC"
    pos = s.index(anchor, start)
    ins = "".join(f'  call LM_PROBE("TSC_in", "{n}", {n})\n' for n in
                  ("M", "row", "k_turb_T_flux", "T_massflux", "eflux0_kinem",
                   "turb_density_flux", "Buoyancy0", "tau", "kor"))
    ins += ('  call LM_PROBE("TSC_in", "ddz", gsp%ddz)\n'
            '  call LM_PROBE("TSC_in", "ddz2", gsp%ddz2)\n'
            '  call LM_PROBE("TSC_in", "dzeta_int", gsp%dzeta_int)\n'
            '  call LM_PROBE("TSC_in", "dzeta_05int", gsp%dzeta_05int)\n'
            '  call LM_PROBE("TSC_in", "ddz05", gsp%ddz05)\n'
            '  call LM_PROBE("TSC_in", "z_full", gsp%z_full)\n'
            '  call LM_PROBE("TSC_in", "h1", ls%h1)\n'
            '  call LM_PROBE("TSC_in", "vol", ls%vol)\n'
            '  call LM_PROBE("TSC_in", "Tw2", wst%Tw2)\n'
            '  call LM_PROBE("TSC_in", "Sal2", wst%Sal2)\n'
            '  call LM_PROBE("TSC_in", "u2", wst%u2)\n'
            '  call LM_PROBE("TSC_in", "v2", wst%v2)\n'
            '  call LM_PROBE("TSC_in", "lamw", wst%lamw)\n'
            '  call LM_PROBE("TSC_in", "lamsal", wst%lamsal)\n'
            '  call LM_PROBE("TSC_in", "area_int", bathymwater(1:M+1)%area_int)\n'
            '  call LM_PROBE("TSC_in", "Lx", bathymwater(1:M+1)%Lx)\n'
            '  call LM_PROBE("TSC_in", "Ly", bathymwater(1:M+1)%Ly)\n'
            '  call LM_PROBE("TSC_in", "RadWater_integr", RadWater%integr)\n')
    s = s[:pos] + ins + s[pos:]
    end = s.index("END SUBROUTINE TURB_SCALES")
    outs = "".join(f'  call LM_PROBE("TSC_out", "{n}", {n})\n' for n in
                   ("i_maxN", "H_mixed_layer", "maxN", "w_conv_scale",
                    "T_conv_scale", "Wedderburn", "LakeNumber", "Rossby_rad",
                    "ThermThick", "ReTherm", "RiTherm"))
    outs += ('  call LM_PROBE("TSC_out", "Rp", trb%Rp)\n'
             '  call LM_PROBE("TSC_out", "Rpdens", trb%Rpdens)\n')
    s = s[:end] + outs + s[end:]
    path.write_text(s)



def patch_momentum_lake(path):
    lines = path.read_text().splitlines(keepends=True)
    inserts = []
    for i, line in enumerate(lines):
        if 'call MOMENTUM_SOLVER' not in line:
            continue
        end = i
        for j in range(i, min(i + 25, len(lines))):
            if lines[j].rstrip().endswith(')'):
                end = j
                break
        tag = f"MOM_{i + 1}"
        pre = "".join(f'  call LM_PROBE("{tag}_in", "{n}", {n})\n' for n in
                      ("u1", "v1", "E1", "eps1", "k2", "row", "Tw1", "Sal1",
                       "Ri", "uv", "knum", "veg_sw", "u_star", "h1", "l1",
                       "ls1", "dhw", "dhw0", "alphax", "alphay",
                       "velfrict_bot", "velfrict_ice"))
        post = "".join(f'  call LM_PROBE("{tag}_out", "{n}", {n})\n' for n in
                       ("u2", "v2", "k2", "Ri",
                        "velfrict_bot", "velfrict_ice"))
        inserts.append((i, pre))
        inserts.append((end + 1, post))
    for pos, text in sorted(inserts, reverse=True):
        lines.insert(pos, text)
    path.write_text("".join(lines))



def patch_bathym2(path):
    lines = path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if 'call BATHYMETRY' not in line:
            continue
        end = i
        for j in range(i, min(i + 25, len(lines))):
            if lines[j].rstrip().endswith(')'):
                end = j
                break
        text = "".join(f'  call LM_PROBE("BATHY2", "{n}", {n})\n' for n in
                       ("zsoilcols(1:nsoilcols+1,ix,iy)",
                        "bathymsoil(1:nsoilcols+1,ix,iy)%area_int",
                        "bathymsoil(1:nsoilcols,ix,iy)%area_half",
                        "bathymsoil(1:nsoilcols+1,ix,iy)%itop",
                        "bathymsoil(1:nsoilcols+1,ix,iy)%dzSL",
                        "bathymsoil(1:nsoilcols+1,ix,iy)%ibot",
                        "bathymsoil(1:nsoilcols+1,ix,iy)%icent",
                        "bathymsoil(1:nsoilcols,ix,iy)%dzSLc"))
        lines.insert(end + 1, text)
        path.write_text("".join(lines))
        return



def patch_kepsilon_lake(path):
    lines = path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if 'call K_EPSILON(' not in line or 'dnx' not in line:
            continue
        # only the case-1 call has dnx,dny on the first line
        end = i
        for j in range(i, min(i + 12, len(lines))):
            if lines[j].rstrip().endswith(')'):
                end = j
                break
        pre = "".join(f'  call LM_PROBE("KE_in", "{n}", {n})\n' for n in
                      ("E1", "eps1", "k2", "k2t", "k5", "k2_mid", "k3_mid",
                       "k5_mid", "l", "tf", "kt", "gamt", "gamu", "gamv",
                       "row", "row2", "z_half", "u1", "v1",
                       "u2", "v2", "um", "vm", "Tw1", "Tw2", "Sal1", "Sal2",
                       "knum", "veg_sw", "lamw", "lamsal", "h1", "l1",
                       "dhw", "dhw0", "vol", "botar", "b0", "tau_air",
                       "tau_wav", "tau_i", "tau_gr", "uwind", "vwind",
                       "fetch", "roughness"))
        post = "".join(f'  call LM_PROBE("KE_out", "{n}", {n})\n' for n in
                       ("E2", "eps2", "k2", "kt", "H_entrainment",
                        "Buoyancy0", "Ri_bulk", "Eseiches"))
        lines.insert(end + 1, post)
        lines.insert(i, pre)
        path.write_text("".join(lines))
        return



def patch_kepsilon_internal(path):
    s = path.read_text()
    start = s.index("SUBROUTINE K_EPSILON")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    # after the surface dissipation flux (open-water kepsbc < 4)
    anchor2 = s.index("xx(1) = 0.5*(area_int(2)/area_half(1)*k5_mid(2)*dt", start)
    s = s[:anchor2] + ('  call LM_PROBE("KE6", "ksurf1", ksurf1(1))\n'
                       '  call LM_PROBE("KE6", "Esurf1", Esurf1(1))\n'
                       '  call LM_PROBE("KE6", "CE1", CE(1))\n'
                       '  call LM_PROBE("KE6", "k21", k2(1))\n'
                       '  call LM_PROBE("KE6", "tau_air", tau_air)\n') + s[anchor2:]
    # after the two PROGONKA solves
    key_e = "call PROGONKA (nveclen,a,b,c,d,E_it2,1,M)"
    pos_e = s.index(key_e, start)
    coef = "".join(f'  call LM_PROBE("KE7", "{n}", {n}(1:M))\n' for n in ("a", "b", "c", "d"))
    s = s[:pos_e] + coef + s[pos_e:]
    for call in ("E_it2", "eps_it2"):
        key = f"call PROGONKA (nveclen,a,b,c,d,{call},1,M)"
        pos = s.index(key, start) + len(key)
        s = s[:pos] + f'\n  call LM_PROBE("KE5", "{call}", {call}(1:M))' + s[pos:]
    anchor = s.index("lam_E  (i) = CE(i)/sigmaE", start)
    enddo = s.index("enddo", anchor)
    ins = ('  call LM_PROBE("KE3", "rhotemp", rhotemp(1:M))\n'
           '  call LM_PROBE("KE3", "rhosal", rhosal(1:M))\n'
           '  call LM_PROBE("KE3", "work", work(1:M,1:2))\n'
           '  call LM_PROBE("KE3", "E12", E12(1:M))\n'
           '  call LM_PROBE("KE3", "eps12", eps12(1:M))\n')
    s = s[:enddo + len("enddo")] + "\n" + ins + s[enddo + len("enddo"):]
    end = s.index("END SUBROUTINE K_EPSILON")
    outs = "".join(f'  call LM_PROBE("KE2_out", "{n}", {n})\n' for n in
                   ("CE", "CEt", "s", "gen", "f", "c_eps1", "c_eps2",
                    "c_eps3", "gamt", "gamu", "gamv", "FTKES", "FTKEB",
                    "FS", "FB", "TKES", "TKEB"))
    s = s[:end] + outs + s[end:]
    path.write_text(s)



def patch_soilforlake_lake(path):
    lines = path.read_text().splitlines(keepends=True)
    inserts = []
    for i, line in enumerate(lines):
        if 'call SOILFORLAKE' not in line:
            continue
        end = i
        for j in range(i, min(i + 6, len(lines))):
            if lines[j].rstrip().endswith(')'):
                end = j
                break
        pre = "".join(f'  call LM_PROBE("SOILF_in", "{n}", {n})\n' for n in
                      ("Tsoil1", "Tsoil2", "Tsoil3", "wl1", "wi1", "wl4",
                       "wi2", "Sals1", "csoil", "rosoil", "lamsoil",
                       "dhwfsoil", "h1", "l1", "ls1", "Tw1"))
        post = "".join(f'  call LM_PROBE("SOILF_out", "{n}", {n})\n' for n in
                       ("Tsoil1", "Tsoil3", "wl1", "wi1", "wl4", "wi2",
                        "dhwfsoil"))
        inserts.append((i, end, pre, post))
    for i, end, pre, post in reversed(inserts):
        lines.insert(end + 1, post)
        lines.insert(i, pre)
    path.write_text("".join(lines))



def patch_initvars(path):
    lines = path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if 'call INIT_VAR' not in line:
            continue
        end = i
        for j in range(i, min(i + 40, len(lines))):
            if lines[j].rstrip().endswith(')'):
                end = j
                break
        text = "".join(f'  call LM_PROBE("INIT_IN", "{n}", {n})\n' for n in
                       ("Tsoil1", "wi1", "wl1", "Sals1", "Tw1", "h1",
                        "l1", "hs1", "ls1"))
        lines.insert(end + 1, text)
        path.write_text("".join(lines))
        return



def patch_soilcolstemp(path):
    s = path.read_text()
    start = s.index("SUBROUTINE SOILCOLSTEMP")
    imp = s.index("implicit none", start)
    s = s[:imp] + "use LM_PROBE_MOD, only : LM_PROBE\n" + s[imp:]
    key = "call TSURFSOILCOL("
    pos = s.index(key, start)
    # find the end of this call
    end = s.index("\n", s.index("Tsoilsurf)", pos)) + 1
    s = s[:end] + '  call LM_PROBE("SCOL", "Tsoilsurf", Tsoilsurf(1:gs%nsoilcols))\n' + s[end:]
    end2 = s.index("END SUBROUTINE SOILCOLSTEMP")
    s = s[:end2] + ('  call LM_PROBE("SCOL", "Tsoil1", Tsoil1)\n'
                    '  call LM_PROBE("SCOL", "soilflux", soilflux(1:gs%nsoilcols))\n') + s[end2:]
    path.write_text(s)




def patch_case1_checkpoints(path):
    lines = path.read_text().splitlines(keepends=True)
    inserts = []
    for i, line in enumerate(lines):
        if line.strip() == "dhw0 = dhwp + dhwe":
            tag = f"CK_DHW_{i + 1}"
            text = "".join(f'  call LM_PROBE("{tag}", "{n}", {n})\n' for n in
                           ("dhw", "dhw0", "dhwfsoil", "Elatent", "h1"))
            inserts.append((i + 1, text))
            continue
        for call in ("call SOILCOLSTEMP", "call SOILFORLAKE"):
            if call not in line:
                continue
            end = i
            for j in range(i, min(i + 12, len(lines))):
                if lines[j].rstrip().endswith(')'):
                    end = j
                    break
            tag = f"CK_{call.split()[1]}_{i + 1}"
            text = "".join(f'  call LM_PROBE("{tag}", "{n}", {n})\n' for n in
                           ("Tsoil1", "Tsoil2", "Tsoil3", "dhwfsoil", "h1",
                            "dhw", "dhw0", "Tw1", "Elatent"))
            inserts.append((end + 1, text))
    for pos, text in sorted(inserts, reverse=True):
        lines.insert(pos, text)
    path.write_text("".join(lines))

def patch_cmake(path):
    text = path.read_text()
    needle = "${path}/lake_modules.f90"
    if needle not in text:
        raise SystemExit("lake_modules.f90 not in CMakeLists")
    text = text.replace(needle, needle + "\n\t${path}/lm_probe.f90", 1)
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LAKEBATCH / "scratch" / "probe-build"))
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; use a new --out directory (no rm -rf)")
    (out / "build").mkdir(parents=True)

    shutil.copytree(SRC, out / "source")
    shutil.copy(TEMPLATE, out / "source" / "model" / "lm_probe.f90")

    cmake = (LAKEBATCH / "core" / "build" / "CMakeLists.txt").read_text()
    cmake = cmake.replace('RUNTIME_OUTPUT_DIRECTORY "../"', 'RUNTIME_OUTPUT_DIRECTORY "."')
    old = 'list(APPEND FLAGS_ "-O3")'
    if old not in cmake:
        raise SystemExit("CMakeLists: -O3 flag line not found")
    cmake = cmake.replace(old, old + '\nlist(APPEND FLAGS_ "-ffp-contract=off")')
    (out / "build" / "CMakeLists.txt").write_text(cmake)
    patch_cmake(out / "build" / "CMakeLists.txt")

    patch_lake(out / "source" / "model" / "lake.f90")
    patch_init_var(out / "source" / "model" / "init_var_mod.f90")
    patch_density(out / "source" / "model" / "lake.f90")
    patch_bathym(out / "source" / "model" / "lake.f90")
    patch_sfcflx(out / "source" / "shared" / "SfcFlx_mod.F90")
    patch_heatbalance(out / "source" / "model" / "heatbalsurf_mod.f90")
    patch_senslatmom(out / "source" / "model" / "heatbalsurf_mod.f90")
    patch_heatfluxsurf(out / "source" / "model" / "heatbalsurf_mod.f90")
    patch_radbalance(out / "source" / "model" / "heatbalsurf_mod.f90")
    patch_surf_char(out / "source" / "model" / "t_solver_mod.f90")
    patch_tdiff(out / "source" / "model" / "t_solver_mod.f90")
    patch_t_solver(out / "source" / "model" / "t_solver_mod.f90")
    patch_turb_scales(out / "source" / "model" / "phys_func.f90")
    patch_momentum_lake(out / "source" / "model" / "lake.f90")
    patch_bathym2(out / "source" / "model" / "lake.f90")
    patch_kepsilon_lake(out / "source" / "model" / "lake.f90")
    patch_kepsilon_internal(out / "source" / "model" / "turb_mod.f90")
    patch_soilforlake_lake(out / "source" / "model" / "lake.f90")
    patch_initvars(out / "source" / "model" / "lake.f90")
    patch_soilcolstemp(out / "source" / "model" / "soil_mod.f90")
    patch_case1_checkpoints(out / "source" / "model" / "lake.f90")
    print(f"probe source: {out/'source'}")

    if args.build:
        env = dict(os.environ)
        cmd = ["cmake", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
               f"-DCMAKE_Fortran_FLAGS=-I{out/'source'/'model'}"]
        if sys.platform == "darwin":
            sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
            cmd += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0", f"-DCMAKE_OSX_SYSROOT={sdk}"]
            env["MACOSX_DEPLOYMENT_TARGET"] = "15.0"
        cmd.append(".")
        r = subprocess.run(cmd, cwd=out / "build", env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / "build" / "cmake.log").write_text(r.stdout)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit("cmake failed")
        r = subprocess.run(["cmake", "--build", ".", "--target", "lake.out",
                            "--parallel", str(args.parallel)],
                           cwd=out / "build", env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / "build" / "build.log").write_text(r.stdout)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit("build failed")
        print(f"built {out/'build'/'lake.out'}")


if __name__ == "__main__":
    main()
