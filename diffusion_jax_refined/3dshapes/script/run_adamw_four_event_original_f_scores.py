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
ALL_EVENT_METHODS=(
    'four','e1','e2','e3','e4',
    'four_residual','e1_residual','e2_residual','e3_residual','e4_residual',
)
VARIANTS=('raw','query_l2','train_l2','query_train_l2')
TARGETS=('endpoint_contarfactual','traj_contarfactual','simple_loss','noise_trajectory')

def parse_ints(text): return [int(x) for x in text.replace(',', ' ').split() if x]

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

def event_learning_rates(root,epoch,cfg,total_steps,steps_per_epoch,shards=2):
    import DM__training_CIFAR5_MULTI_pixel as module
    schedule=module.make_learning_rate_schedule(cfg,total_steps)
    values=[]
    for s in range(shards):
        p=root/f'event_gradient_epoch_{epoch:04d}_shard_{s:02d}_of_{shards:02d}.npz'
        if not p.is_file(): raise FileNotFoundError(p)
        with np.load(p,allow_pickle=False) as z:
            batches=np.asarray(z['batch_indices'],np.int64)
            saved_epoch=int(np.asarray(z['epoch']).item())
        if saved_epoch!=epoch: raise ValueError(f'epoch mismatch in {p}: {saved_epoch} != {epoch}')
        steps=(epoch-1)*steps_per_epoch+batches
        values.append(np.asarray(schedule(jnp.asarray(steps)),np.float64))
    return np.concatenate(values)

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',default='experiment1'); ap.add_argument('--train-seed',type=int,default=42)
    ap.add_argument('--epochs',type=int,default=200); ap.add_argument('--query-namespace',default='loss_direction_residual_rms_original_f')
    ap.add_argument('--query-file',type=Path,default=ROOT/'queries_seed_0_9.json'); ap.add_argument('--query-ids',default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--attribution-points',type=int,default=5000); ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument(
        '--checkpoint-weighting',
        choices=('stored_lr','uniform','previous_checkpoint_lr'),
        default='stored_lr',
        help=(
            'uniform removes the outer checkpoint learning-rate factor while retaining '
            'equal averaging over timestamps; previous_checkpoint_lr gives checkpoint c '
            'the stored outer weight from checkpoint c-1 and gives checkpoint 0 zero weight'
        ),
    )
    ap.add_argument(
        '--contraction',
        choices=('linear', 'squared', 'timestamp_sum_squared'),
        default='linear',
        help=(
            'linear sums signed dot products; squared squares each checkpoint/timestamp '
            'dot product before summing; timestamp_sum_squared first sums checkpoints '
            'within each query timestamp, then squares and sums timestamps'
        ),
    )
    ap.add_argument('--plot-selection',action='store_true',help='write endpoint and trajectory LDS scatter plots for four_residual/query_train_l2')
    ap.add_argument('--include-all-events',action='store_true',help='also score E2, E3, E4 and their history-subtracted residuals')
    ap.add_argument(
        '--methods',default='',
        help='optional comma-separated subset of four,e1,four_residual,e1_residual',
    )
    ap.add_argument(
        '--save-score-namespace',default='',
        help='optionally save one selected 5000-point score vector per query under attribution_score',
    )
    ap.add_argument('--save-score-method',choices=ALL_EVENT_METHODS,default='four_residual')
    ap.add_argument('--save-score-variant',choices=VARIANTS,default='query_train_l2')
    ap.add_argument(
        '--event-original-lr',action='store_true',
        help=(
            'For eventwise FOUR/FOUR_RESIDUAL, divide each fixed-checkpoint AdamW '
            'event feature by the checkpoint LR, apply the selected contraction '
            'separately, then weight it once by the original LR at that datapoint '
            'event batch.'
        ),
    )
    ap.add_argument(
        '--single-checkpoint-lr',action='store_true',
        help=(
            'For eventwise FOUR/FOUR_RESIDUAL, divide every event feature by '
            'the checkpoint LR, apply the selected contraction separately, '
            'then multiply once by that same checkpoint LR. With squared '
            'contraction this is exactly square(q^T u) / checkpoint_lr.'
        ),
    )
    a=ap.parse_args()
    if a.event_original_lr and a.single_checkpoint_lr:
        ap.error('--event-original-lr and --single-checkpoint-lr are mutually exclusive')
    eventwise_lr = a.event_original_lr or a.single_checkpoint_lr
    if eventwise_lr and a.checkpoint_weighting == 'previous_checkpoint_lr':
        ap.error('eventwise LR modes cannot be combined with previous_checkpoint_lr')
    if eventwise_lr and a.include_all_events:
        ap.error('eventwise LR modes cannot be combined with --include-all-events')
    methods=('four','four_residual') if eventwise_lr else (ALL_EVENT_METHODS if a.include_all_events else METHODS)
    if a.methods:
        requested=tuple(value for value in a.methods.replace(',', ' ').split() if value)
        invalid=sorted(set(requested)-set(METHODS))
        if invalid: ap.error(f'unsupported --methods: {invalid}')
        if not requested: ap.error('--methods selected no methods')
        methods=requested
    if eventwise_lr and any(m not in ('four','four_residual') for m in methods):
        ap.error('eventwise LR modes only support four and four_residual')
    weighting_label = (
        'single_checkpoint_lr'
        if a.single_checkpoint_lr
        else ('original_event_lr_once' if a.event_original_lr else a.checkpoint_weighting)
    )
    mod=importlib.import_module('DM__training_CIFAR5_MULTI_pixel'); from dtrak.algorithm import _countsketch_project_grad_jax
    ckroot=ROOT/'result'/a.experiment/'model'/'prompted_jax'; art=ROOT/'result'/a.experiment/f'fixed_checkpoint_adamw_four_events_n{a.attribution_points}'
    with (ckroot/f'seed_{a.train_seed}_epoch_0004.ckpt').open('rb') as f: payload=pickle.load(f)
    cfg=mod.TrainConfig(**dict(payload['config'])); dev=mod.choose_devices('cpu')[0]
    ds=mod.CIFAR10Dataset(root=cfg.data_root,batch_names=cfg.batch_names,use_test=cfg.use_test,class_names=cfg.class_names,normalize='minus_one_to_one',channels_last=True,exclude_ranges=cfg.exclude_ranges,exclude_indices=cfg.exclude_indices,cond_mode=cfg.cond_mode)
    cfg=mod.TrainConfig(**{**asdict(cfg),'num_classes':len(ds.label_names)}); model=mod.build_model(cfg)
    steps_per_epoch=len(ds)//cfg.batch_size; total_steps=steps_per_epoch*cfg.epochs
    template=mod.create_train_state(cfg,model,jax.random.PRNGKey(cfg.seed),dev,total_steps)
    class A: pass
    qa=A(); qa.experiment=a.experiment; qa.train_seed=a.train_seed; qa.epochs=a.epochs
    qa.query_file=a.query_file; query_ids=parse_ints(a.query_ids)
    records_all=json.loads(a.query_file.read_text())['queries']
    if not query_ids or any(q < 0 or q >= len(records_all) for q in query_ids): raise ValueError('invalid --query-ids')
    query,meta=load_query_bank(qa,a.query_namespace,'trajectory_next_checkpoint_noise_mse',query_ids)
    lookup={(int(c),int(t)):i for i,(c,t) in enumerate(zip(meta['ckpt_indices'],meta['timesteps']))}
    num_timestamps=len(dict.fromkeys(int(x) for x in meta['timesteps']))
    if num_timestamps <= 0:
        raise ValueError('query artifact contains no trajectory timestamps')
    scores={(m,v):np.zeros((len(query_ids),a.attribution_points),np.float64) for m in methods for v in VARIANTS}; score_idx=None
    timestamp_order=tuple(dict.fromkeys(int(x) for x in meta['timesteps']))
    timestamp_position={t:i for i,t in enumerate(timestamp_order)}
    timestamp_scores=(
        {
            (m,v):np.zeros(
                (len(timestamp_order),len(query_ids),a.attribution_points),
                np.float64,
            )
            for m in methods for v in VARIANTS
        }
        if a.contraction=='timestamp_sum_squared'
        else None
    )
    for c in range(49):
        se=4*(c+1); state,_=mod._restore_checkpoint(str(ckroot/f'seed_{a.train_seed}_epoch_{se:04d}.ckpt'),template)
        zero=jax.tree_util.tree_map(jnp.zeros_like,state.params); hu,_=state.tx.update(zero,state.opt_state,state.params)
        hist=np.asarray(_countsketch_project_grad_jax(hu,4096,seed_parts=(a.train_seed,'traj_tracin_projection',c)),np.float32)
        ev=[]; event_lrs=[]
        for e in range(se+1,se+5):
            x,idx=event(art/f'epoch_{se}_{se+4}',e)
            if score_idx is None: score_idx=idx
            elif not np.array_equal(score_idx,idx): raise ValueError(f'index mismatch checkpoint {c+1}')
            ev.append(x)
            if a.event_original_lr:
                event_lrs.append(event_learning_rates(
                    art/f'epoch_{se}_{se+4}',e,cfg,total_steps,steps_per_epoch
                ))
        banks={
            'four':sum(ev),
            'e1':ev[0],
            'four_residual':sum(x-hist for x in ev),
            'e1_residual':ev[0]-hist,
        }
        if a.include_all_events:
            for event_index,event_feature in enumerate(ev,1):
                banks[f'e{event_index}']=event_feature
                banks[f'e{event_index}_residual']=event_feature-hist
        checkpoint_lr=mod.learning_rate_at_step(cfg,se*steps_per_epoch,total_steps)
        if eventwise_lr and checkpoint_lr<=0:
            raise ValueError(f'checkpoint {c+1} has nonpositive LR {checkpoint_lr}')
        for t in timestamp_order:
            qi=lookup.get((c,t))
            if qi is None: continue
            q=query[:,qi,:]
            if a.checkpoint_weighting=='uniform':
                weight=1.0/num_timestamps
            elif a.checkpoint_weighting=='previous_checkpoint_lr':
                previous_qi=lookup.get((c-1,t)) if c>0 else None
                weight=(
                    float(meta['term_weights'][previous_qi])
                    if previous_qi is not None
                    else 0.0
                )
            else:
                weight=float(meta['term_weights'][qi])
            qn=np.linalg.norm(q,axis=1)+1e-8
            if eventwise_lr:
                weighted_banks={
                    'four':ev,
                    'four_residual':[x-hist for x in ev],
                }
                lr_weights = event_lrs if a.event_original_lr else [checkpoint_lr] * len(ev)
                for m in methods:
                    event_bank=weighted_banks[m]
                    accumulated={v:np.zeros((len(idx),q.shape[0]),np.float64) for v in VARIANTS}
                    for x,event_lr in zip(event_bank,lr_weights):
                        direction=x/checkpoint_lr
                        dots=direction@q.T; xn=np.linalg.norm(direction,axis=1)+1e-8
                        values={
                            'raw':dots,
                            'query_l2':dots/qn[None,:],
                            'train_l2':dots/xn[:,None],
                            'query_train_l2':dots/(xn[:,None]*qn[None,:]),
                        }
                        for v,value in values.items():
                            transformed=np.square(value) if a.contraction=='squared' else value
                            if np.ndim(event_lr) == 0:
                                accumulated[v]+=float(event_lr)*transformed
                            else:
                                accumulated[v]+=np.asarray(event_lr)[:,None]*transformed
                    for v,value in accumulated.items():
                        scores[(m,v)]+=(value/num_timestamps).T
                continue
            for m in methods:
                x=banks[m]
                dots=x@q.T; xn=np.linalg.norm(x,axis=1)+1e-8
                values={
                    'raw':dots,
                    'query_l2':dots/qn[None,:],
                    'train_l2':dots/xn[:,None],
                    'query_train_l2':dots/(xn[:,None]*qn[None,:]),
                }
                for v,value in values.items():
                    if a.contraction == 'timestamp_sum_squared':
                        assert timestamp_scores is not None
                        timestamp_scores[(m,v)][timestamp_position[t]]+=weight*value.T
                        continue
                    if a.contraction == 'squared':
                        value = np.square(value)
                    scores[(m,v)]+=weight*value.T
        print(f'[score] checkpoint={c+1}/49',flush=True)
    if timestamp_scores is not None:
        for key,value in timestamp_scores.items():
            scores[key]=np.square(value).sum(axis=0)
    assert score_idx is not None
    records=[records_all[q] for q in query_ids]; rows=[]
    if a.save_score_namespace:
        selected_key=(a.save_score_method,a.save_score_variant)
        if selected_key not in scores:
            raise ValueError(f'cannot save unavailable score {selected_key}; computed methods={methods}')
        for local_q,(query_id,r) in enumerate(zip(query_ids,records)):
            score_dir=(
                ROOT/'result'/a.experiment/'attribution_score'/'prompted_solo'
                /f'train_seed_{a.train_seed}'/f"query_{_prompt_tag(str(r['prompt']))}"
                /f"initial_seed_{int(r['initial_seed'])}"/a.save_score_namespace/'score'
            )
            score_dir.mkdir(parents=True,exist_ok=True)
            np.save(score_dir/'scores.npy',np.asarray(scores[selected_key][local_q],np.float64))
            np.save(score_dir/'score_indices.npy',np.asarray(score_idx,np.int64))
            (score_dir/'score_metadata.json').write_text(json.dumps({
                'method':a.save_score_method,
                'variant':a.save_score_variant,
                'query_namespace':a.query_namespace,
                'checkpoint_weighting':weighting_label,
                'contraction':a.contraction,
                'num_timestamps':num_timestamps,
            },indent=2,sort_keys=True)+'\n')
            print(f'[score artifact] Q{query_id}: {score_dir}',flush=True)
    for local_q,(query_id,r) in enumerate(zip(query_ids,records)):
        er=ROOT/'result'/a.experiment/'eval'/'prompted_solo'/f"query_{_prompt_tag(str(r['prompt']))}"/f"initial_seed_{int(r['initial_seed'])}"
        incidence,true=load_target_data(cache_group(er),score_idx)
        for m in methods:
            for v in VARIANTS:
                pred=scores[(m,v)][local_q]@incidence.T
                for target in TARGETS:
                    lds=100*float(rowwise_spearman(pred[None,:],true[target])[0])
                    rows.append({'method':m,'variant':v,'query':query_id,'target':target,'lds_percent':lds,'prediction_sign':'p1','checkpoint_weighting':weighting_label,'contraction':a.contraction})
                    if (
                        a.plot_selection
                        and m == 'four_residual'
                        and v == 'query_train_l2'
                        and target in ('endpoint_contarfactual', 'traj_contarfactual')
                    ):
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt

                        plot_dir = (
                            a.out_dir
                            / 'lds_scatter_four_residual_query_train_l2'
                            / target
                        )
                        plot_dir.mkdir(parents=True, exist_ok=True)
                        fig, axis = plt.subplots(figsize=(5.2, 4.4))
                        axis.scatter(pred, true[target], s=18, alpha=0.7, edgecolors='none')
                        axis.set_xlabel('Predicted subset score')
                        axis.set_ylabel(f'True {target}')
                        axis.set_title(f'Q{query_id} LDS={lds:+.3f}%')
                        axis.grid(alpha=0.2)
                        fig.tight_layout()
                        fig.savefig(plot_dir / f'Q{query_id}.png', dpi=200)
                        plt.close(fig)
    write(a.out_dir/'per_query.csv',rows)
    trajectory_label = (
        'OWN TRAJECTORY'
        if 'checkpoint_own_trajectory' in a.query_namespace
        else 'REFERENCE TRAJECTORY'
    )
    print(f'{a.contraction.upper()} ORIGINAL-F {trajectory_label} — FIXED P1 — CHECKPOINT WEIGHTING={weighting_label}')
    for m in methods:
        for variant in VARIANTS:
            vals=[]
            print(f'\n{m.upper()} — {variant.upper()}')
            for query_id in query_ids:
                value=[next(x['lds_percent'] for x in rows if x['method']==m and x['variant']==variant and x['query']==query_id and x['target']==t) for t in TARGETS]; vals.append(value)
                print(f"Q{query_id} "+' '.join(f'{x:+8.3f}%' for x in value))
            print('MEAN '+' '.join(f'{x:+8.3f}%' for x in np.mean(vals,axis=0)))
    print(f'[saved] {a.out_dir}')
if __name__=='__main__': main()
