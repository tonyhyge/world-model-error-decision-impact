"""Run the fixed thesis protocol and record source/data hashes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json,time,platform
from datetime import datetime, timezone
import numpy,scipy,torch,gymnasium
from experiments.common import ROOT,SEEDS
from experiments import (exp1_gridworld_margins,exp2_cartpole_continuous,
    exp3_baseline_comparison,exp4_fork_signed_direction,exp5_mujoco_signal_audit,
    exp6_cartpole_holdout,exp7_first_order_law)

def hashes(paths):
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

if __name__=='__main__':
    start=time.time()
    source=hashes(list((ROOT/'src').rglob('*.py'))+list((ROOT/'experiments').glob('*.py')))
    for exp in [exp1_gridworld_margins,exp2_cartpole_continuous,exp3_baseline_comparison,
                exp4_fork_signed_direction,exp5_mujoco_signal_audit,exp6_cartpole_holdout,exp7_first_order_law]: exp.run()
    manifest=dict(protocol='thesis-v3',utc=datetime.now(timezone.utc).isoformat(),seeds=SEEDS,
        seconds=time.time()-start,python=platform.python_version(),platform=platform.platform(),
        versions={m.__name__:m.__version__ for m in [numpy,scipy,torch,gymnasium]},
        source_sha256=source,result_sha256=hashes((ROOT/'results').glob('*.csv')),
        # The MuJoCo tables were produced on GPU elsewhere and are inputs here,
        # not outputs; hash them so a silent edit cannot pass verification.
        frozen_evidence_sha256=hashes((ROOT/'results'/'mujoco_signal_audit').glob('*')))
    (ROOT/'results'/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
