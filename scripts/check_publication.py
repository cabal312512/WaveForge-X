"""Small public-package equivalence check against frozen received data."""
import json
from pathlib import Path
import numpy as np
from run_study import configuration
from waveforge6g.channels.tdl_profile import TDLProfile
from waveforge6g.receivers.certiphy_selective import ProfileCertiPHY
from waveforge6g.waveforms import create_waveform

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'results/certiphy'
protocol=json.loads((data/'analysis/freeze.json').read_text(encoding='utf-8'))
for seed in [920000,920500]:
    for wave_name in ['ofdm','otfs','afdm']:
        key=f'{seed}_0_{wave_name}'
        rows=json.loads((data/'confirm'/(key+'.json')).read_text(encoding='utf-8'))
        row=next(r for r in rows if r['method']=='selective:0.01')
        raw=np.load(data/'confirm'/(key+'.npz'),allow_pickle=False)
        c=TDLProfile(row['profile'],seed,row['ds'],row['fd'],row['fs'])
        w=create_waveform(wave_name,row['n'],row['cp'],16)
        rx=ProfileCertiPHY(w,c,10**(-row['snr']/10),row['modulation'])
        rule,refiner=configuration(protocol['methods']['selective:0.01'])
        result=rx.solve(raw['received'],rule,max_work=rx.setup_work+300*rx.step_work+30e6*rx.n/1024,refiner=refiner)
        index=list(raw['methods']).index('selective:0.01')
        np.testing.assert_array_equal(result['bits'],raw['outputs'][index])
        # The integrated receiver now constructs and charges an explicit return
        # selection even for full payloads. Preserve the historical work record;
        # explain this fixed setup increment instead of silently rewriting it.
        selection_work = 4*rx.n*rx.width
        assert result['work_parts']['output_selection'] == selection_work
        assert result['work'] == row['work'] + selection_work
        assert result['iterations'] == row['iterations']
        print(key,'identical output and iterations; work differs only by charged output selection',selection_work)
