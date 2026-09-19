#!/usr/bin/env python3
"""Linear/p1 original-F scores from fixed-checkpoint AdamW event features."""
from __future__ import annotations
import argparse, csv, importlib, json, pickle, sys
from dataclasses import asdict
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT.parent/'legacy_jax')]
from dataset_config import _prompt_tag
from run_expected_residual_jacobian_scores import load_query_bank
from analyze_predicted_noise_probe8_choose4 import cache_group, load_target_data
from analyze_predicted_noise_probe12_sign_flips import rowwise_spearman

METHODS=('four','e1','four_residual','e1_residual')
VARIANTS=('raw','query_l2','train_l2','query_train_l2')
TARGETS=('endpoint_contarfactual','traj_contarfactual','simple_loss','noise_trajectory')

def event(root,epoch,shards=2):
    xs=[]; ids=[]
    for s in range(shards):
        p=root/f'event_gradient_epoch_{epoch:04d}_shard_{s:02d}_of_{shards:02d}.npz'
        if not p.is_file(): raise FileNotFoundError(p)
        with np.load(p,allow_pickle=False) as z:
            xs.append(np.asarray(z['train_features'],np.float32)); ids.append(np.asarray(z['dataset_indices'],np.int64))
    idx=np.concatenate(ids)
    if len(np.unique(idx))!=len(idx): raise ValueError(f'duplicate indices: {root}')
    return np.concatenate(xs),idx

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',default='experiment1'); ap.add_argument('--train-seed',type=int,default=42)
    ap.add_argument('--epochs',type=int,default=200); ap.add_argument('--query-namespace',default='loss_direction_residual_rms_original_f')
    ap.add_argument('--attribution-points',type=int,default=5000); ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--checkpoint-weighting',choices=('stored_lr','uniform'),default='stored_lr',help='uniform removes the outer checkpoint learning-rate factor while retaining equal averaging over timestamps')
    a=ap.parse_args()
    mod=importlib.import_module('DM__training_CIFAR5_MULTI_pixel'); from dtrak.algorithm import _countsketch_project_grad_jax
    ckroot=ROOT/'result'/a.experiment/'model'/'prompted_jax'; art=ROOT/'result'/a.experiment/f'fixed_checkpoint_adamw_four_events_n{a.attribution_points}'
    with (ckroot/f'seed_{a.train_seed}_epoch_0004.ckpt').open('rb') as f: payload=pickle.load(f)
    cfg=mod.TrainConfig(**dict(payload['config'])); dev=mod.choose_devices('cpu')[0]
    ds=mod.CIFAR10Dataset(root=cfg.data_root,batch_names=cfg.batch_names,use_test=cfg.use_test,class_names=cfg.class_names,normalize='minus_one_to_one',channels_last=True,exclude_ranges=cfg.exclude_ranges,exclude_indices=cfg.exclude_indices,cond_mode=cfg.cond_mode)
    cfg=mod.TrainConfig(**{**asdict(cfg),'num_classes':len(ds.label_names)}); model=mod.build_model(cfg)
    template=mod.create_train_state(cfg,model,jax.random.PRNGKey(cfg.seed),dev,(len(ds)//cfg.batch_size)*cfg.epochs)
    class A: pass
    qa=A(); qa.experiment=a.experiment; qa.train_seed=a.train_seed; qa.epochs=a.epochs
    query,meta=load_query_bank(qa,a.query_namespace,'trajectory_next_checkpoint_noise_mse',range(10))
    lookup={(int(c),int(t)):i for i,(c,t) in enumerate(zip(meta['ckpt_indices'],meta['timesteps']))}
    num_timestamps=len(dict.fromkeys(int(x) for x in meta['timesteps']))
    if num_timestamps!=10: raise ValueError(f'expected 10 timestamps, got {num_timestamps}')
    scores={(m,v):np.zeros((10,a.attribution_points),np.float64) for m in METHODS for v in VARIANTS}; score_idx=None
    for c in range(49):
        se=4*(c+1); state,_=mod._restore_checkpoint(str(ckroot/f'seed_{a.train_seed}_epoch_{se:04d}.ckpt'),template)
        zero=jax.tree_util.tree_map(jnp.zeros_like,state.params); hu,_=state.tx.update(zero,state.opt_state,state.params)
        hist=np.asarray(_countsketch_project_grad_jax(hu,4096,seed_parts=(a.train_seed,'traj_tracin_projection',c)),np.float32)
        ev=[]
        for e in range(se+1,se+5):
            x,idx=event(art/f'epoch_{se}_{se+4}',e)
            if score_idx is None: score_idx=idx
            elif not np.array_equal(score_idx,idx): raise ValueError(f'index mismatch checkpoint {c+1}')
            ev.append(x)
        banks={'four':sum(ev),'e1':ev[0],'four_residual':sum(x-hist for x in ev),'e1_residual':ev[0]-hist}
        for t in dict.fromkeys(int(x) for x in meta['timesteps']):
            qi=lookup.get((c,t))
            if qi is None: continue
            q=query[:,qi,:]
            weight=(1.0/num_timestamps if a.checkpoint_weighting=='uniform' else float(meta['term_weights'][qi]))
            qn=np.linalg.norm(q,axis=1)+1e-8
            for m,x in banks.items():
                dots=x@q.T; xn=np.linalg.norm(x,axis=1)+1e-8
                values={
                    'raw':dots,
                    'query_l2':dots/qn[None,:],
                    'train_l2':dots/xn[:,None],
                    'query_train_l2':dots/(xn[:,None]*qn[None,:]),
                }
                for v,value in values.items(): scores[(m,v)]+=weight*value.T
        print(f'[score] checkpoint={c+1}/49',flush=True)
    assert score_idx is not None
    records=json.loads((ROOT/'queries_seed_0_9.json').read_text())['queries']; rows=[]
    for q,r in enumerate(records):
        er=ROOT/'result'/a.experiment/'eval'/'prompted_solo'/f"query_{_prompt_tag(str(r['prompt']))}"/f"initial_seed_{int(r['initial_seed'])}"
        incidence,true=load_target_data(cache_group(er),score_idx)
        for m in METHODS:
            for v in VARIANTS:
                pred=scores[(m,v)][q]@incidence.T
                for target in TARGETS:
                    lds=100*float(rowwise_spearman(pred[None,:],true[target])[0])
                    rows.append({'method':m,'variant':v,'query':q,'target':target,'lds_percent':lds,'prediction_sign':'p1','checkpoint_weighting':a.checkpoint_weighting})
    write(a.out_dir/'per_query.csv',rows)
    trajectory_label = (
        'OWN TRAJECTORY'
        if 'checkpoint_own_trajectory' in a.query_namespace
        else 'REFERENCE TRAJECTORY'
    )
    print(f'LINEAR ORIGINAL-F {trajectory_label} — FIXED P1 — CHECKPOINT WEIGHTING={a.checkpoint_weighting}')
    for m in METHODS:
        for variant in VARIANTS:
            vals=[]
            print(f'\n{m.upper()} — {variant.upper()}')
            for q in range(10):
                value=[next(x['lds_percent'] for x in rows if x['method']==m and x['variant']==variant and x['query']==q and x['target']==t) for t in TARGETS]; vals.append(value)
                print(f"Q{q} "+' '.join(f'{x:+8.3f}%' for x in value))
            print('MEAN '+' '.join(f'{x:+8.3f}%' for x in np.mean(vals,axis=0)))
    print(f'[saved] {a.out_dir}')
if __name__=='__main__': main()
