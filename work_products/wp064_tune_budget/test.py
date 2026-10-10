"""WP064 — is tune=600 more than this model needs? Motivated by WP063's depth finding.

NUTS tree depth is EXACTLY 6 in 2,400/2,400 draws, with `maxdepth_reached` at 0%, so the
sampler converges to a stable step size and trajectory length almost immediately and never
wanders. That suggests the 600 tuning draws are overkill.

The budget is 600 tune + 600 draws, so TUNING IS HALF the gradient evaluations. Cutting it
to 300 would be ~25% off every sweep -- larger than enabling the C backend (~15%).

NOT ALREADY TESTED. `SAMPLER_FAST` (300/300) was measured and rejected, but that cut draws
AND tune together; cutting draws costs posterior precision, which is why it failed. Cutting
TUNE alone costs nothing if adaptation has already converged.

THE FAILURE MODE IS SELF-DIAGNOSING, which is what makes this a clean experiment: if 300
tune is not enough, nutpie ends up with a worse mass matrix, trees get DEEPER than 6, and the
fit gets SLOWER -- so the problem announces itself in the depth statistic instead of hiding
in the results.

QUALITY IS JUDGED ON ESS, NOT RPS. A 3-window RPS comparison would be pure noise (project
rule: one-window dRPS is only trustworthy at ~13x MDE). ESS per draw is the right measure:
if the posterior is explored equally well, the saving is safe.

Run SERIALLY so the timings are clean -- the machine is otherwise idle.
"""
import pickle, sys, time
from pathlib import Path
import numpy as np
import arviz as az
import pymc as pm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from football_model.data.prepare_model_data import prepare_model_data   # noqa: E402
from football_model.model.model import build_model                      # noqa: E402
from football_model.types.model_data import ModelConfig                 # noqa: E402

CFG = dict(use_xg_channel=True, xg_channel_weight=0.75,
           use_deep_channel=True, deep_channel_weight=0.5,
           use_stakes_effort=True, stakes_effort_sd=0.3,
           clip_theta=5.0, center_team_strength=False, season_start_window=3)
WINDOWS = (100, 200, 300)
TUNES = (600, 300)
KEY_PARAMS = ["sigma_att", "sigma_def", "rho_att", "rho_def", "home_mu",
              "xg_kappa", "deep_kappa", "stakes_effort"]



def _agg(res, how):
    """min/max over every DataArray in an arviz result.

    arviz returns a Dataset in some versions and a DataTree in others (this venv returns
    DataTree, which has no .to_array()), so walk whatever came back rather than assuming.
    """
    import xarray as xr
    vals = []
    def walk(o):
        if isinstance(o, xr.DataArray):
            vals.append(float(o.min() if how == "min" else o.max()))
        elif isinstance(o, xr.Dataset):
            for v in o.data_vars:
                walk(o[v])
        else:
            for child in getattr(o, "children", {}).values():
                walk(child)
            ds = getattr(o, "dataset", None)
            if ds is not None and len(getattr(ds, "data_vars", {})):
                walk(ds)
    walk(res)
    if not vals:
        return float("nan")
    return min(vals) if how == "min" else max(vals)


def main():
    sh = pickle.loads((REPO / "work_products/wp055_horizon1/cv_shared_data_h1.pkl").read_bytes())
    df, windows = sh["df_cv"], sh["windows"]
    rows = []
    for W in WINDOWS:
        win = windows[W - 1]
        train = df[df["round"] <= win["train_end"]]
        md = prepare_model_data(train, max_round=win["train_end"])
        for tune in TUNES:
            with build_model(md, ModelConfig(**CFG)):
                t0 = time.time()
                idata = pm.sample(draws=600, tune=tune, chains=4, target_accept=0.9,
                                  nuts_sampler="nutpie", progressbar=False,
                                  idata_kwargs={"log_likelihood": False})
                secs = time.time() - t0
            ss = idata.sample_stats
            d = ss["depth"].values.ravel().astype(float)
            have = [p for p in KEY_PARAMS if p in idata.posterior]
            ess = _agg(az.ess(idata, var_names=have), "min")
            rh = _agg(az.rhat(idata, var_names=have), "max")
            rows.append(dict(window=W, tune=tune, secs=secs,
                             depth_mean=d.mean(), depth_max=d.max(),
                             maxdepth_pct=100 * float(np.mean(ss["maxdepth_reached"].values)),
                             div=int(ss["diverging"].values.sum()),
                             min_ess=ess, max_rhat=rh))
            r = rows[-1]
            print(f"  w{W} tune={tune}: {secs:6.0f}s  depth {r['depth_mean']:.2f}/"
                  f"{r['depth_max']:.0f}  maxdepth {r['maxdepth_pct']:.0f}%  div {r['div']}"
                  f"  minESS {ess:5.0f}  rhat {rh:.3f}", flush=True)
    print("\n=== VERDICT ===")
    print(f"  {'window':>7s} {'t600':>7s} {'t300':>7s} {'saving':>8s} "
          f"{'depth 600/300':>14s} {'minESS 600/300':>16s} {'rhat 600/300':>14s}")
    tot6 = tot3 = 0.0
    for W in WINDOWS:
        a = next(r for r in rows if r["window"] == W and r["tune"] == 600)
        b = next(r for r in rows if r["window"] == W and r["tune"] == 300)
        tot6 += a["secs"]; tot3 += b["secs"]
        print(f"  {W:7d} {a['secs']:7.0f} {b['secs']:7.0f} {1-b['secs']/a['secs']:7.1%} "
              f"{a['depth_mean']:6.2f}/{b['depth_mean']:<7.2f} "
              f"{a['min_ess']:7.0f}/{b['min_ess']:<8.0f} "
              f"{a['max_rhat']:6.3f}/{b['max_rhat']:<7.3f}")
    print(f"\n  TOTAL: {tot6:.0f}s -> {tot3:.0f}s  = {1-tot3/tot6:.1%} saving")
    print("  ADOPT only if: depth stays ~6, divergences stay 0, min ESS does not fall")
    print("  materially, and rhat is comparable. A deeper tree means adaptation FAILED.")
    pickle.dump(rows, (Path(__file__).parent / "results.pkl").open("wb"))


if __name__ == "__main__":
    main()
