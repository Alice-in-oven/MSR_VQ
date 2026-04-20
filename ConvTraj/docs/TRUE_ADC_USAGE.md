# True ADC Usage

`true ADC` is now available directly from the main evaluation path through `train.py`.

It is enabled by:

- `eval_embedding_type = "quantized"`
- `eval_search_mode = "adc"`

This runs real asymmetric distance computation:

- query side: transformed continuous sub-vectors
- base side: discrete codes + student codebook
- scoring: lookup-table distance accumulation over `M` groups

This is different from the old fake `adc` path that compared query continuous embeddings against reconstructed quantized embeddings with plain L2.

## Recommended Config Patterns

### 1. Pure True ADC Evaluation

Use this when you want quantized retrieval scored by true ADC and do not need rerank.

```json
{
  "eval_embedding_type": "quantized",
  "eval_search_mode": "adc",
  "enable_rerank": false
}
```

### 2. True ADC + decoded rerank

Use this when you want a quantized coarse shortlist from ADC, then rerank with decoded embeddings.

```json
{
  "eval_embedding_type": "quantized",
  "eval_search_mode": "adc",
  "enable_rerank": true,
  "rerank_L": 100,
  "rerank_source": "adc"
}
```

### 3. Export both decoded and true ADC metrics

Use this when you want both decoded and ADC metrics in one evaluation pass.

```json
{
  "eval_embedding_type": "quantized",
  "eval_search_mode": "both",
  "enable_rerank": false
}
```

## Notes

- `true ADC` only applies to quantized retrieval.
- `continuous` evaluation is controlled separately by `eval_embedding_type`.
- `rerank_source = "adc"` only makes sense when `eval_search_mode` includes `"adc"`.
- `eval_search_mode = "both"` writes both `decoded` and `adc` sections into the metrics json.

## Geolife Example

Example config:

- [prequant_v5_stedec_sweep_m8_k16_e500_trueadc_eval_400.json](/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/data/train_config/prequant_v5_stedec_sweep_m8_k16_e500_trueadc_eval_400.json:1)

Example command:

```bash
cd /data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj

python train.py \
  --mode test \
  --train_flag geolife_mainchain_trueadc_e400 \
  --load_model_train_flag prequant_v5_stedec_sweep_m8_k16_e500 \
  --eval_embedding_type quantized \
  --eval_search_mode adc \
  --test_epoch 400
```
