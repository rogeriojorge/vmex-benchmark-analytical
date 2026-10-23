"""Collect the non-stellarator-symmetric results into results/asymmetric/summary.json.

Reads the seeded VMEX runs written by ``project_integer_vmex.py --solve`` and
their ``exact_score_v2.json``, adds the Fourier spectrum of the flux-label error
(where it goes when the transform is rational) and the Boozer check.
"""
import ast
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import ROOT, cases, field
from boozer_exact import score as boozer_score

RUNS = ROOT/"results/projection"
OUT = ROOT/"results/asymmetric/summary.json"
CASES = ("integer_3d", "integer_3d_full_basis", "integer_3d_tau", "integer_3d_tau_mirror",
         "sheared_issan", "sheared_issan_tau", "sheared_issan_tau_mirror")


def label_spectrum(case, wout, j, n=32):
    """Largest (m, n) harmonics of the exact-label error on VMEX surface j."""
    from score_wout_exact import positions
    xyz = positions(wout, j, 2*np.pi*np.arange(n)/n, 2*np.pi*np.arange(n)/(n*int(wout.nfp)))
    label = np.asarray(jax.vmap(lambda x: field(case, x)[1])(jnp.asarray(xyz))).reshape(n, n)
    amp = np.abs(np.fft.fft2(label/case.edge - j/(int(wout.ns)-1)))/n**2
    top = np.argsort(amp.ravel())[::-1][:6:2]  # conjugate pairs: keep one of each
    wrap = lambda k: int(k) if k < n//2 else int(k)-n  # noqa: E731
    return {"mean": float(amp[0, 0]), "modes": [[wrap(k//n), wrap(k % n), float(amp.ravel()[k])] for k in top]}


def main():
    import vmex
    out = {"runs": {}}
    for name in CASES:
        for ns in (33, 65, 129):
            run = RUNS/f"{name}_vmex_ns{ns}"
            log = RUNS/f"{name}_seeded{ns}.log"
            if not log.exists():
                continue
            text = log.read_text()
            row = {"converged": "seeded_solve" in text}
            if row["converged"]:
                row["fsq"] = ast.literal_eval(text[text.index("{'seeded_solve'"):].splitlines()[0])["seeded_solve"]["fsq"]
            elif "unconverged: fsq" in text:
                row["fsq"] = list(ast.literal_eval(text.split("unconverged: fsq ")[1].split(" after")[0]))
            wout_path = run/f"wout_{name}_seeded_ns{ns}.nc"
            if row["converged"] and wout_path.exists() and not (run/"exact_score_v2.json").exists():
                from score_wout_exact import score as exact_score
                deck = ROOT/"inputs"/f"input.{name}_iota"
                (run/"exact_score_v2.json").write_text(json.dumps(exact_score(wout_path, name, deck=deck), indent=1))
            if (run/"exact_score_v2.json").exists():
                score = json.loads((run/"exact_score_v2.json").read_text())
                row.update({k: score[k] for k in ("flux_label_error_max", "B_relative_surface", "iota_max_abs_error")},
                           surfaces=[[r["s"], r["flux_label_error_max"]] for r in score["surfaces"]],
                           routes=score["routes"])
                if ns == 65 and row["converged"]:
                    wout = vmex.read_wout(wout_path)
                    row["label_spectrum"] = {str(j): label_spectrum(cases()[name], wout, j) for j in (16, 32, 48)}
                    row["boozer"] = boozer_score(wout_path, name, 32, 32)["surfaces"]
            out["runs"][f"{name}_ns{ns}"] = row
            print(name, ns, {k: v for k, v in row.items() if k in ("converged", "fsq", "flux_label_error_max")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
