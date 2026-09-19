#!/usr/bin/env python3
"""Materialize four no-timestep-alignment AdamW event banks as standard train parts."""
from __future__ import annotations
import argparse, importlib, pickle, sys
from dataclasses import asdict
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT.parent/'legacy_jax')]
from run_adamw_four_event_original_f_scores import event
from run_expected_residual_jacobian_scores import load_query_bank

METHODS=(
 'four','e1','e2','e3','e4','four_residual',
 'e1_residual','e2_residual','e3_residual','e4_residual'
); SEM='fixed_checkpoint_adamw_event_no_timestamp_alignment'
def save(path,**kw):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**kw); tmp.replace(path)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--experiment',default='experiment1'); p.add_argument('--train-seed',type=int,default=42); p.add_argument('--attribution-points',type=int,default=5000); a=p.parse_args()
 mod=importlib.import_module('DM__training_CIFAR5_MULTI_pixel'); from dtrak.algorithm import _countsketch_project_grad_jax
 ck=ROOT/'result'/a.experiment/'model'/'prompted_jax'; source=ROOT/'result'/a.experiment/f'fixed_checkpoint_adamw_four_events_n{a.attribution_points}'
 with (ck/f'seed_{a.train_seed}_epoch_0004.ckpt').open('rb') as f: pl=pickle.load(f)
 cfg=mod.TrainConfig(**dict(pl['config'])); dev=mod.choose_devices('cpu')[0]
 ds=mod.CIFAR10Dataset(root=cfg.data_root,batch_names=cfg.batch_names,use_test=cfg.use_test,class_names=cfg.class_names,normalize='minus_one_to_one',channels_last=True,exclude_ranges=cfg.exclude_ranges,exclude_indices=cfg.exclude_indices,cond_mode=cfg.cond_mode)
 cfg=mod.TrainConfig(**{**asdict(cfg),'num_classes':len(ds.label_names)}); template=mod.create_train_state(cfg,mod.build_model(cfg),jax.random.PRNGKey(cfg.seed),dev,(len(ds)//cfg.batch_size)*cfg.epochs)
 class A: pass
 x=A(); x.experiment=a.experiment; x.train_seed=a.train_seed; x.epochs=200
 _,meta=load_query_bank(x,'loss_direction_residual_rms_original_f','trajectory_next_checkpoint_noise_mse',[0])
 timesteps=[]; weights=[]
 for c in range(50):
  mask=np.asarray(meta['ckpt_indices'])==c; timesteps.append(np.asarray(meta['timesteps'])[mask]); weights.append(np.asarray(meta['term_weights'])[mask])
 for c in range(49):
  se=4*(c+1); state,_=mod._restore_checkpoint(str(ck/f'seed_{a.train_seed}_epoch_{se:04d}.ckpt'),template)
  zero=jax.tree_util.tree_map(jnp.zeros_like,state.params); hu,_=state.tx.update(zero,state.opt_state,state.params)
  hist=np.asarray(_countsketch_project_grad_jax(hu,4096,seed_parts=(a.train_seed,'traj_tracin_projection',c)),np.float32)
  ev=[]; idx=None
  for ep in range(se+1,se+5):
   value,part_idx=event(source/f'epoch_{se}_{se+4}',ep); ev.append(value); idx=part_idx if idx is None else idx
   if not np.array_equal(idx,part_idx): raise ValueError('score index mismatch')
  banks={
   'four':sum(ev),
   **{f'e{i+1}':v for i,v in enumerate(ev)},
   'four_residual':sum(v-hist for v in ev),
   **{f'e{i+1}_residual':v-hist for i,v in enumerate(ev)},
  }
  for method,bank in banks.items():
   out=ROOT/'result'/a.experiment/'model'/'prompted_solo'/f'seed_{a.train_seed}_train_gradient'/f'traj_tracin_adamw4_{method}'/'train_datapoint_gradient_artifact.npz.parts'/f'ckpt_{c:04d}.npz'
   if not out.is_file(): save(out,train_features=bank[None,:,:],score_indices=idx,ckpt_indices=np.full(10,c,np.int32),timesteps=timesteps[c],term_weights=weights[c],train_feature_semantics=np.asarray(SEM),timestamp_shared_train_feature=np.asarray(1,np.int8))
  print(f'[materialize] checkpoint={c+1}/49',flush=True)
 # Zero final checkpoint: it has no next-interval event and contributes exactly zero.
 for method in METHODS:
  out=ROOT/'result'/a.experiment/'model'/'prompted_solo'/f'seed_{a.train_seed}_train_gradient'/f'traj_tracin_adamw4_{method}'/'train_datapoint_gradient_artifact.npz.parts'/f'ckpt_{49:04d}.npz'
  if not out.is_file(): save(out,train_features=np.zeros((1,a.attribution_points,4096),np.float32),score_indices=idx,ckpt_indices=np.full(10,49,np.int32),timesteps=timesteps[49],term_weights=weights[49],train_feature_semantics=np.asarray(SEM),timestamp_shared_train_feature=np.asarray(1,np.int8))
if __name__=='__main__': main()
