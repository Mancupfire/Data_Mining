# Scientific BPR-MF vs SEARec

## Best BPR-MF configuration

- `dim=64`, `lr=0.05`, `reg=0.0001`
- `epochs=12`, `batch_size=2048`, `batches_per_epoch=64`

## Multi-seed test comparison

| Model | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|
| BPR-MF (mean ± std, 3 seeds) | 0.0019 ± 0.0001 | 0.0075 ± 0.0002 | 0.0130 ± 0.0002 | 0.0046 ± 0.0001 | 0.0064 ± 0.0000 |
| SEARec (saved reference) | 0.0075 | 0.0265 | 0.0419 | 0.0169 | 0.0219 |

## Per-seed BPR-MF runs

| Seed | Best epoch | R@10 | N@10 |
|---|---:|---:|---:|
| 2020 | 16 | 0.0129 | 0.0063 |
| 2021 | 16 | 0.0133 | 0.0064 |
| 2022 | 16 | 0.0129 | 0.0064 |
