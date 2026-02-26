# GPU NaN Operator Investigation (2026-02-26)

## Scope
- Target: `decoder_step` GPU NaN issue (operator-level isolation), including re-export candidate checks.
- Goal: make Unity inference use GPU if numerically stable.

## Environment
- Unity: `6000.3.6f1`
- `com.unity.sentis` (manifest): `2.5.0`
- Resolved runtime package: `com.unity.ai.inference` `2.4.1` (`Packages/packages-lock.json`)
- Model files:
  - `Assets/NNG2P/Models/encoder.onnx`
    - SHA256: `6E5C2F06E1CE5FF6B8D3A67B23A5E3FE4CCBD3498B055D46ADB4D3CB22CC761D`
  - `Assets/NNG2P/Models/decoder_step.onnx`
    - SHA256: `11821663B2B914ACC68910576BE958E87EAF2820A0E245486B14B35B091617B7`

## Method
- Added temporary Editor diagnostics (used during investigation, not part of runtime)
- Probes executed via MCP (`uloop execute-menu-item`):
  - Output-level checks (native fallback enabled/disabled, backend variants)
  - Layer-by-layer checks (`ScheduleIterable`) to find first NaN layer
- Baseline CPU probes were executed to validate probe correctness.

## Key Findings
1. GPUCompute (native fallback, current models) returns all-NaN tensors.
   - Encoder output: `nan=196608/196608` for `memory (1,512,384)`
   - Decoder output: `phone_logits nan=23552/23552`, `prosody_logits nan=26624/26624`
2. GPUPixel also returns all-NaN tensors with the same pattern.
3. CPU baseline is numerically healthy.
   - Encoder: `nan=0/196608`
   - Decoder: `phone_logits nan=0/23552`, `prosody_logits nan=0/26624`
4. Operator-level first NaN (pure GPU execution) was isolated to `Dense`:
   - Encoder: first NaN at `layer=15`, `op=Dense`, `out=[47]`, shape `(512,1,1152)`
   - Decoder: first NaN at `layer=7`, `op=Dense`, `out=[23]`, shape `(512,1,768)`
5. Re-export candidate models were checked (existing `third_party/nn-g2p-jp/exports/*` variants).
   - Dynamic candidate (`ja_m9_sentis`) and fixed128 candidate both hit `Reshape` assertion failures in Sentis/Inference.

## Conclusion
- With the current package/model combination, GPU backends are not numerically usable for this model family.
- GPU execution can be attempted, but must immediately fallback to CPU based on numeric validation.

## Implemented Safety Change
- `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs` now performs GPU numeric validation at initialization.
- If encoder/decoder probe detects NaN, runtime automatically falls back to CPU before normal inference.
- Verified in Play Mode logs:
  - Warning: `Backend GPUCompute failed numeric validation (encoder output contains NaN). Falling back to CPU backend.`
  - Inference continues with correct CPU output.

## Recommended Next Roadmap
1. Track and retest with newer `com.unity.ai.inference` versions (GPU numerical stability fixes).
2. Prepare a minimal reproducible sample (current ONNX + one input) for upstream issue reporting.
3. If GPU is mandatory now, evaluate alternative runtime backends (outside current Sentis stack) for this model.
