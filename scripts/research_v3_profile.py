"""Small standard-library CPU/allocation profile; no additional dependencies."""

import time
import tracemalloc

import numpy as np
import pandas as pd

from waveforge6g.core.modulation import demodulate
from waveforge6g.experiments.research_v3 import DOC, make_channel
from waveforge6g.receivers.budgeted import BudgetedReceiver, ReceiverAction
from waveforge6g.waveforms import create_waveform

rows = []
for n in (128, 256):
    channel = make_channel(n, "high_rich", 42001)
    for name in ("ofdm", "otfs", "afdm"):
        for domain, keep, iterations in (("time", 0, 8), ("symbol", 8, 16), ("dense", n, 0)):
            action = ReceiverAction(name, domain, keep, iterations)
            tracemalloc.start()
            wave = create_waveform(name, n, n//8, 16)
            receiver = BudgetedReceiver(wave, channel, 10**(-.8), action)
            received = channel.apply(wave.modulate(np.ones(n, complex)))[n//8:]
            estimate, _ = receiver.solve(received)
            demodulate(estimate, "qpsk")
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            setup_cpu, solve_cpu, elapsed = [], [], []
            for _repeat in range(5):
                start, cpu = time.perf_counter(), time.process_time()
                wave = create_waveform(name, n, n//8, 16)
                receiver = BudgetedReceiver(wave, channel, 10**(-.8), action)
                setup_cpu.append(time.process_time()-cpu)
                cpu = time.process_time()
                # Batch only for timer resolution, not for amortizing receiver setup.
                for _ in range(32):
                    wave.modulate(np.ones(n, complex))
                    estimate, _ = receiver.solve(received)
                    demodulate(estimate, "qpsk")
                solve_cpu.append((time.process_time()-cpu)/32)
                elapsed.append(time.perf_counter()-start)
            rows.append(dict(n=n, action=action.key, domain=domain, traced_peak_bytes=peak,
                             retained_receiver_bytes=receiver.array_bytes,
                             peak_array_bytes_model=receiver.cost["peak_array_bytes_model"],
                             setup_process_cpu_s=np.mean(setup_cpu), frame_process_cpu_s=np.mean(solve_cpu),
                             setup_plus_frame_process_cpu_s=np.mean(setup_cpu)+np.mean(solve_cpu),
                             profile_elapsed_s=sum(elapsed)))
pd.DataFrame(rows).to_csv(DOC/"cpu_memory_profile.csv", index=False)
print(pd.DataFrame(rows).groupby(["n", "domain"])[["traced_peak_bytes", "setup_plus_frame_process_cpu_s"]].mean().to_string())
