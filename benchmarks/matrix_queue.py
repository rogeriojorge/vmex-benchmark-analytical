"""Stage 4 queue files: one run per line, ``case|code|protocol|ns|tag|extra args``.

Phases are separate files so a loaded box can stop between them:
  a: axisymmetric Solov'ev (cheap), every protocol and NS
  b: 3-D / asymmetric cases at NS 17, 33, 65
  c: NS 129 of the 3-D cases
"""
import sys

AXI = ("solovev_symmetric", "solovev_asymmetric")
THREE_D = ("integer_3d", "integer_3d_tau", "integer_3d_tau_mirror", "sheared_A")
NS_B = (17, 33, 65)


def case_lines(case, ns_list):
    out = []
    for ns in ns_list:
        out.append((case, "vmex", "seed", ns, "", ""))
        out.append((case, "vmex", "cold", ns, "", ""))
        out.append((case, "vmec2000", "cold", ns, "", ""))
        if ns > 17:
            out.append((case, "vmex", "ladder", ns, "", ""))
            out.append((case, "vmec2000", "ladder", ns, "", ""))
        out.append((case, "vmex", "seed", ns, "_tcon0", "--tcon0 0.0"))
    return out


def wp_lines(cases_, proto_large="ladder", ns_list=(17, 33, 65, 129), repeats=3, ftol=1e-14,
             sweep=(1e-6, 1e-8, 1e-10, 1e-12), ns_sweep=65):
    """Work-precision lines for both codes: repeated timings per NS and an ftol sweep at ``ns_sweep``.

    ``proto_large`` is the protocol for NS > 17 (NS 17 is always cold).  Sweep runs loosen every ladder
    stage to the swept tolerance.
    """
    out = []
    for case in cases_:
        for code in ("vmex", "vmec2000"):
            for ns in ns_list:
                proto = "cold" if ns == 17 else proto_large
                extra = "" if ftol == 1e-14 else f"--ftol {ftol} --ftol-stages {ftol}"
                for i in range(repeats):
                    out.append((case, code, proto, ns, f"_rep{i}", extra, f"{case}_{proto}_{ns}_{ftol:.0e}"))
            for f in sweep:
                out.append((case, code, proto_large, ns_sweep, f"_ftol{f:.0e}", f"--ftol {f} --ftol-stages {f}",
                            f"{case}_{proto_large}_{ns_sweep}_sweep"))
    return out


PATH_CASES = ("sheared_path_t25", "sheared_path_t50", "sheared_path_t75", "sheared_issan")


def path_lines(ns=33):
    """Sheared parameter continuation A -> Issan at fixed NS: exact seed, cold and a restart chain per point.

    The chain restarts t25 from the exact-seed solve of sheared A (NS ``ns``), then each point from the
    previous point's restart solve.  VMEC2000 gets the cold solve only.
    """
    out, previous = [], f"sheared_A-seed-ns{ns}-vmex"
    for case in PATH_CASES:
        out += [(case, "vmex", "seed", ns, "", ""), (case, "vmex", "cold", ns, "", ""), (case, "vmec2000", "cold", ns, "", "")]
        run = f"{case}-restart_chain-ns{ns}-vmex"
        out.append((case, "vmex", "restart", ns, "_chain", f"--restart-wout @RUNS@/{previous}/wout_{previous}.nc"))
        previous = run
    return out


def phase(name):
    if name == "a":
        return [x for c in AXI for x in case_lines(c, (17, 33, 65, 129))]
    if name == "b":
        return [x for c in THREE_D for x in case_lines(c, NS_B)]
    if name == "c":
        return [x for c in THREE_D for x in case_lines(c, (129,))]
    raise SystemExit("phase a, b or c")


if __name__ == "__main__":
    if sys.argv[1] == "wp":  # axisymmetric Solov'ev: converged to 1e-14
        for row in wp_lines(sys.argv[2:]):
            print("|".join(str(v) for v in row))
    elif sys.argv[1] == "path":
        for row in path_lines(int(sys.argv[2]) if len(sys.argv) > 2 else 33):
            print("|".join(str(v) for v in row))
    elif sys.argv[1] == "wp3d":  # 3-D cases never reach 1e-14: cold solves to ftol 1e-8, sweep at NS 33
        for row in wp_lines(sys.argv[2:], proto_large="cold", ns_list=(17, 33, 65), repeats=2, ftol=1e-8,
                            sweep=(1e-6, 1e-7, 1e-9, 1e-10), ns_sweep=33):
            print("|".join(str(v) for v in row))
    else:
        for row in phase(sys.argv[1]):
            print("|".join(str(v) for v in row))
