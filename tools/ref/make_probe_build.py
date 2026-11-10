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
