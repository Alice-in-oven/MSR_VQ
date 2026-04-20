## Chengdu Policy Memory

This note is a hard constraint for all future Chengdu optimization work in this repo.

Rules:

1. Any Chengdu optimization must be Chengdu-specific.
2. Do not change default training or evaluation behavior for Geolife, Porto, or any other dataset.
3. New logic must be opt-in through explicit flags, explicit config values, or Chengdu-only experiment scripts.
4. If a code path is generalized for reuse, it must remain disabled by default and must not alter other datasets unless the new arguments are explicitly supplied.
5. Prefer reusing teacher checkpoints, offline artifacts, and experiment-local policies before changing the shared backbone/model architecture.

Current Chengdu target:

- Goal: make `MSR + PDT-VQ` beat `continuous + Faiss OPQ` on Chengdu within 400 epochs.
- OPQ reference to beat: `Top-10 = 0.2899`
- Reference file:
  `/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/chengdu_cont_opq/faiss_opq_m8_nbits4/chengdu_cont_opq_m8_nbits4_faiss_quant_stats.json`

Current strongest Chengdu continuous teacher:

- Checkpoint:
  `/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/chengdu_cont_opq/checkpoints/chengdu_cont_msr_e300_e300.pt`
- Continuous Top-10:
  `0.4432`

Preferred Chengdu optimization order:

1. Preserve the strong continuous teacher with `backbone_checkpoint` and frozen warmup.
2. Use explicit `improved_vq` schedule for Chengdu `m8 k16` instead of the default adaptive medium-codebook path.
3. Add Chengdu retrieval-aware supervision on decoded space:
   `pre_quant_neighbor`, `pre_quant_landmark`, teacher-fixed banks.
4. If needed, use offline OPQ artifacts only as Chengdu-specific teacher signals, never as a replacement for PDT-VQ.
