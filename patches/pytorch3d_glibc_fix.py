#!/usr/bin/env python
"""Patch a CUDA-built .so that requires GLIBC_2.35 so it loads on glibc 2.34.

Why: pytorch3d (and some other extensions) compiled against the CUDA 12.8 math
headers emit a versioned reference to `hypotf@GLIBC_2.35`. On a glibc-2.34 host
(Rocky 9.4) the system libm only provides `hypotf@GLIBC_2.2.5`, so import fails
with:  ImportError: /lib64/libm.so.6: version `GLIBC_2.35' not found ...

`hypotf` is ABI-identical across these versions, so we rebind the version
requirement (Verneed entry) down to GLIBC_2.2.5 with LIEF, which recomputes the
version hash on write. Run AFTER building the extension.

Usage:
    python pytorch3d_glibc_fix.py <path-to-_C.so> [<more.so> ...]
    # or, auto-find pytorch3d's _C in the active env:
    python pytorch3d_glibc_fix.py --auto

Requires: pip install lief
"""
import subprocess, sys, glob, os


def patch(so: str) -> bool:
    out = subprocess.run(["objdump", "-T", so], capture_output=True, text=True).stdout
    if "GLIBC_2.35" not in out:
        print(f"  {so}: no GLIBC_2.35 — nothing to do")
        return False
    import lief
    b = lief.ELF.parse(so)
    n = 0
    for req in b.symbols_version_requirement:
        for aux in req.get_auxiliary_symbols():
            if aux.name == "GLIBC_2.35":
                aux.name = "GLIBC_2.2.5"
                n += 1
    b.write(so)
    left = subprocess.run(["objdump", "-T", so], capture_output=True, text=True).stdout.count("GLIBC_2.35")
    print(f"  {so}: rebound {n} GLIBC_2.35 requirement(s); remaining={left}")
    return True


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--auto":
        import pytorch3d
        args = glob.glob(os.path.dirname(pytorch3d.__file__) + "/_C*.so")
        print("auto-found:", args)
    for so in args:
        patch(so)


if __name__ == "__main__":
    main()
