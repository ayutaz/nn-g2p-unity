# nn-g2p-unity

Unity 上で日本語 G2P（Grapheme-to-Phoneme）推論を実行する OSS プロジェクトです。  
入力テキストから `phones`（音素列）と `prosody`（韻律記号列）を推論します。

## Demo

![Demo UI](docs/demo.png)

## Features

- 日本語 G2P 推論（`phones` / `prosody`）
- AR（Autoregressive）推論のみ対応
- TextMeshPro + IME による日本語入力対応
- サンプルシーンで即時実行可能
- GPU 数値異常検知時の自動 CPU フォールバック

## Runtime Scope

- モデル: `ayousanz/nn-g2p-jp`（`ja_m9`）
- 推論モード: AR のみ（CTC は削除済み）
- 想定入力: 日本語
- デフォルト入力: `こんにちは、今日はいい天気ですね`

## Requirements

- Unity: `6000.3.6f1`
- Packages:
  - `com.unity.sentis: 2.5.0`
  - `com.unity.ugui: 2.0.0`
  - `com.unity.inputsystem: 1.18.0`
- 実行時 API: `Unity.InferenceEngine`（Unity 6000 系）

## Quick Start

1. このリポジトリを clone
2. Unity でプロジェクトを開く
3. `Assets/Scenes/NnG2pSampleScene.unity` を開く
4. Play 実行
5. 入力欄に日本語文を入れて `Run AR` を押す

## Model Files

- ONNX:
  - `Assets/NNG2P/Models/encoder.onnx`
  - `Assets/NNG2P/Models/decoder_step.onnx`
- Vocab:
  - `Assets/StreamingAssets/nn-g2p/vocab/ja_grapheme_m4.txt`
  - `Assets/StreamingAssets/nn-g2p/vocab/ja_phones_m8.txt`
  - `Assets/StreamingAssets/nn-g2p/vocab/ja_prosody_or_stress_m8.txt`
- Meta:
  - `Assets/StreamingAssets/nn-g2p/model_meta.json`

## Update Model (Optional)

Hugging Face から `Assets/StreamingAssets/nn-g2p` を更新する場合:

```powershell
uv venv .venv
. .\.venv\Scripts\Activate.ps1
$env:HF_TOKEN="<your_hf_token>"  # private repo の場合
uv run python tools/download_hf_nn_g2p.py --repo ayousanz/nn-g2p-jp
```

## Known Limitations

- 現在の同梱モデルは日本語前提です（英語品質は保証しません）。
- GPU backend で NaN が発生する既知ケースがあり、実装は自動で CPU にフォールバックします。
- ビルド時のサンプル起動には `ProjectSettings/EditorBuildSettings.asset` で
  `Assets/Scenes/NnG2pSampleScene.unity` を含めてください。

## Documentation

- 技術調査: `docs/nn-g2p-model_technical_investigation.md`
- GPU NaN 切り分け: `docs/gpu_nan_operator_investigation_2026-02-26.md`
- AR 最適化調査: `docs/ar_optimization_15agent_investigation.md`

## License

Apache License 2.0 (`Apache-2.0`)  
Copyright 2026 ayutaz
