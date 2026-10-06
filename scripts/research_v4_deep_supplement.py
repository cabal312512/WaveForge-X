"""Small frozen CSI sensitivity, N2048 stress, and receiver timing profiles."""
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from research_v4_deep import OUT, Budget, evaluate_case, rules_for, snapshot

from waveforge6g.channels import ChannelRealization
from waveforge6g.experiments.research_v2 import sha, write_json
from waveforge6g.receivers.certiphy import CertiPHY
from waveforge6g.receivers.certiphy_deep import DeepRefiner
from waveforge6g.waveforms import create_waveform


def run():
    destination = OUT/"supplement"
    if (destination/"manifest.json").exists():
        raise RuntimeError("supplement already completed")
    destination.mkdir(exist_ok=True)
    rules = rules_for("confirm")
    code = snapshot()
    write_json(destination/"protocol.json", dict(source_hash=code, script_sha256=sha(Path(__file__)),
               mismatch=dict(n=256, modulation="qam64", condition="quasi", seeds=[95001,95002,95003],
                             epochs=2, nmse=[0., .01, .1], methods="six frozen d001 methods"),
               stress=dict(n=2048, modulation="qam64", condition="quasi", seeds=[95101,95102],
                           epochs=2, methods="all frozen methods/deltas"),
               profile="first quasi, near-singular, fast and ordinary confirmation cases; OFDM; three methods; 8 timed repetitions"))
    budget = Budget()
    rows = []
    for seed in (95001, 95002, 95003, 95101, 95102):
        stress = seed >= 95100
        n = 2048 if stress else 256
        selected = rules if stress else [r for r in rules if r[0].endswith("d001")]
        for epoch in range(2):
            for wave in ("ofdm", "otfs", "afdm"):
                for nmse in ([0.] if stress else [0., .01, .1]):
                    budget.check(len(selected))
                    part, _, raw = evaluate_case(n, "qam64", "quasi", seed, epoch, wave, selected, nmse)
                    for row in part:
                        row["stage"] = "stress" if stress else "mismatch"
                    rows.extend(part)
                    np.savez_compressed(destination/f"{seed}_{epoch}_{wave}_{nmse}.npz", **raw)
        pd.DataFrame(rows).to_csv(destination/"frames.csv", index=False)
        budget.save()
        print("supplement", seed, flush=True)
    profiles = []
    metadata = pd.concat(pd.read_csv(p) for p in (OUT/"confirm").glob("*/frames.csv"))
    for seed in (92011, 92021, 92031, 92061, 92071):
        case = metadata[(metadata.seed == seed)&(metadata.epoch == 0)&(metadata.waveform == "ofdm")].iloc[0]
        raw = np.load(OUT/f"confirm/{seed}/0_ofdm.npz")
        wave = create_waveform("ofdm", int(case.n), int(case.n)//8, 16)
        channel = ChannelRealization.from_dict(json.loads(str(raw["estimated_channel_json"])))
        for name, rule, mode in [r for r in rules if r[0] in ("gray_d001", "spectral_d001", "residual_fast_d001")]:
            budget.check(9)
            elapsed, cpus = [], []
            for _ in range(8):
                start, cpu = time.perf_counter(), time.process_time()
                receiver = CertiPHY(wave, channel, 10**(-case.snr/10), case.modulation)
                result = receiver.solve(raw["received"], rule, max_work=8e6*case.n/128,
                                        refiner=None if mode is None else DeepRefiner(mode))
                elapsed.append(time.perf_counter()-start)
                cpus.append(time.process_time()-cpu)
            tracemalloc.start()
            receiver = CertiPHY(wave, channel, 10**(-case.snr/10), case.modulation)
            receiver.solve(raw["received"], rule, max_work=8e6*case.n/128,
                           refiner=None if mode is None else DeepRefiner(mode))
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            profiles.append(dict(seed=seed, n=case.n, condition=case.condition, method=name,
                                 elapsed_mean=np.mean(elapsed), elapsed_p95=np.quantile(elapsed,.95),
                                 cpu_mean=np.mean(cpus), traced_peak_bytes=peak,
                                 repeats=8, work=result["work"]))
    pd.DataFrame(profiles).to_csv(destination/"profiles.csv", index=False)
    write_json(destination/"manifest.json", dict(source_hash=code,
               files={p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name != "manifest.json"}))
    budget.save()


if __name__ == "__main__":
    run()
