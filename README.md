# Decoding Parameters
## Fastdllm
- `decoding`: 'fastdllm'
- `remasking`: {'low_confidence' | 'random'} (default: 'low_confidence')
- `use_cache`: boolean (default: False)
- `threshold`: {float (0 - 1) | None} (default: None)
- `factor` : {flaot (0 - 1), None} (default: None)
- `dual_cache`: boolean (default: False)

## Hierachy Decoding
### hierarchy_fast
- `decoding`: 'hierarchy_fast_v2'
- `threshold`: {float (0 - 1) | None} (default: None)
- `low_threshold`: {float (0 - threshold) | None} (default: None)

### hierarchy_remasking
- `decoding`: 'hierarchy_remasking'
- `threshold`: {float (0 - 1) | None} (default: None)
- `low_threshold`: {float (0 - threshold) | None} (default: None)