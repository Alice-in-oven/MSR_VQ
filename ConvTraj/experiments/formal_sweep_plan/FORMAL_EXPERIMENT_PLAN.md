# Formal Experiment Plan

## Manifest

Generated manifests:

- [formal_experiment_manifest.csv](/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/experiments/formal_sweep_plan/formal_experiment_manifest.csv:1)
- [formal_experiment_manifest.json](/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/experiments/formal_sweep_plan/formal_experiment_manifest.json:1)

Total tasks: `42`

- Quantized main training: `27`
- Geolife continuous training: `3`
- Geolife offline Faiss PQ/OPQ evaluation: `12`

## Naming Convention

### Quantized Main Runs

Pattern:

`{dataset}_{dist}_{model}_pdtvq_m{m}_k{k}_e{epoch}_trueadc`

Examples:

- `geolife_dtw_msr_pdtvq_m16_k64_e500_trueadc`
- `chengdu_haus_neutraj_pdtvq_m16_k64_e500_trueadc`
- `porto_dfd_simformer_pdtvq_m16_k64_e600_trueadc`

Rules:

- `dataset`: `geolife | chengdu | porto`
- `dist`: `haus | dtw | dfd`
- `model`: `msr | neutraj | simformer`
- all quantized main runs use `true ADC` as the official quantized retrieval head

### Geolife Continuous Runs

Pattern:

`geolife_{dist}_msr_continuous_e500`

Examples:

- `geolife_dtw_msr_continuous_e500`
- `geolife_haus_msr_continuous_e500`

### Offline Faiss Runs

Pattern:

`geolife_{dist}_{quantizer}_m16_k64_from_cont_e{epoch}`

Examples:

- `geolife_dtw_pq_m16_k64_from_cont_e400`
- `geolife_dfd_opq_m16_k64_from_cont_e500`

## Official Evaluation / Save Policy

### Quantized Main Runs

- only run `m16,k64`
- Geolife / Chengdu: train to `500 epoch`
- Porto: train to `600 epoch`
- save checkpoints every `100 epoch`
- Geolife / Chengdu evaluate at `400,500`
- Porto evaluate at `600`
- save continuous embeddings at evaluation milestones
- official quantized retrieval metric: `true ADC`

Recommended config fields:

```json
{
  "epoch_num": 500,
  "eval_save_epochs": "400,500",
  "test_epoch": 500,
  "eval_embedding_type": "both",
  "eval_search_mode": "adc",
  "enable_rerank": false
}
```

### Continuous Runs

- Geolife only
- train to `500 epoch`
- save checkpoints every `100 epoch`
- evaluate/export continuous embeddings at `400,500`

### Offline Faiss Runs

- Geolife only
- based on continuous embeddings from `epoch 400,500`
- only run `m16,k64`
- run both `PQ` and `OPQ`

## Configuration Sources

### MSR Main Chain

- Geolife:
  - follow [prequant_v5_stedec_codebook_sweep_summary.json](/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/data/prequant_v5_stedec_codebook_sweep_summary.json:1)
  - different `(m,k)` should inherit the corresponding best Geolife recipe family
- Chengdu:
  - follow [chengdu_vq_teacher_stage1_summary.json](/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/chengdu_vq_teacher_stage1/reports/chengdu_vq_teacher_stage1_summary.json:1)
  - use one unified Chengdu recipe for all codebooks
- Porto:
  - follow the `slow-handoff` direction from [metrics_porto_raw_m16k64_opqcode_frozen700_slowhandoff_cuda1_epochs_600.json](/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/porto_raw_m16k64_opqcode_frozen700_slowhandoff_cuda1/feature_dir/metrics_porto_raw_m16k64_opqcode_frozen700_slowhandoff_cuda1_epochs_600.json:1)

### NeuTraj + PDT_VQ

- base seed config:
  - [neu_lightprequant_vq_m8_k16_e400.json](/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/neu_lightprequant_vq/train_config/neu_lightprequant_vq_m8_k16_e400.json:1)

### SIMformer + PDT_VQ

- base seed configs:
  - [sim_lightprequant_vq_m8_k16_e400.json](/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/sim_lightprequant_vq/train_config/sim_lightprequant_vq_m8_k16_e400.json:1)
  - [sim_vq_m8_k16.json](/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/sim_vq/train_config/sim_vq_m8_k16.json:1)

## GPU Scheduling Strategy

This workload is not only GPU-bound. Porto especially is CPU-heavy, so a naive uniform split by task count is not actually balanced.

The recommended scheduling policy is:

- keep all four GPUs active
- but cap Porto quantized training to at most `1 concurrent Porto job`
- fill the remaining GPUs with Geolife / Chengdu / Faiss jobs

### GPU Roles

#### `cuda:0`

- primary Porto queue
- fallback heavy Chengdu queue when Porto is idle

#### `cuda:1`

- Chengdu quantized queue
- secondary Porto queue only if CPU is clearly healthy

#### `cuda:2`

- Geolife quantized queue

#### `cuda:3`

- Geolife SIMformer queue
- Geolife continuous queue
- Geolife offline Faiss PQ/OPQ queue

## Queue Order

### Phase A: Calibration / seed verification

Run one seed job per family first:

- Geolife MSR
- Geolife NeuTraj
- Geolife SIMformer
- Chengdu MSR
- Porto MSR

Goal:

- verify config generation
- verify `true ADC`
- verify checkpoint naming / embedding export

### Phase B: Main quantized sweep

Priority order:

1. Geolife all models, all dists, all codebooks
2. Chengdu all models, all dists, all codebooks
3. Porto all models, all dists, all codebooks

Reason:

- Geolife and Chengdu are lower risk and faster
- Porto is most CPU-heavy and should not flood the machine early

### Phase C: Geolife continuous + offline Faiss

- keep one dedicated lane on `cuda:3`
- once continuous checkpoints at `400/500` are ready, launch offline PQ/OPQ jobs immediately

## Practical Sharding Rule

For quantized main jobs, use this static shard assignment:

- `cuda:0`: all `porto/*/*`
- `cuda:1`: all `chengdu/*/*`
- `cuda:2`: all `geolife/*/msr` and `geolife/*/neutraj`
- `cuda:3`: all `geolife/*/simformer`, all Geolife continuous runs, and all Geolife Faiss runs

Why this is the best starting point:

- `porto` is isolated from the other queues
- `simformer` jobs are grouped and easier to compare
- `cuda:3` becomes the "analysis / continuous / faiss" lane
- task counts are not identical, but expected wall time is closer than naive even splitting

## If CPU Saturation Happens

Apply this fallback:

- never run more than `3` training jobs at once
- keep `cuda:3` for Faiss or evaluation-only jobs
- pause new Porto launches until one non-Porto job finishes

This is the recommended fallback because earlier runs showed 4 heavy training jobs can become CPU-bound and slow each other down.
