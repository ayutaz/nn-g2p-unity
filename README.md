# nn-g2p-unity

`nn-g2p-jp` モデルを Unity で実行するためのサンプル実装です。  
日本語テキストから `phones`（音素列）と `prosody`（韻律記号列）を推論します。

## 現在の実装スコープ

- 推論モードは `AR (Autoregressive)` のみ
- 日本語入力に対応（TextMeshPro + IME）
- Unity Sentis 2.5.0（Unity 6000 系では `Unity.InferenceEngine` 名前空間で実行）
- サンプルシーンでそのまま動作確認可能

## 言語対応（重要）

- 現在同梱しているモデルは `ayousanz/nn-g2p-jp` の `ja_m9`（日本語モデル）です。
- そのため実運用の入力は日本語が前提です。
- Python実装のコードベースは `ja/en` 分岐を持ちますが、現在のチェックポイントと語彙は `ja_*` 固定です。
- 英語や英数字を高品質に扱うには、英語対応チェックポイントへ差し替えるか、事前に日本語読みに正規化する必要があります。

## 動作環境

- Unity: `6000.3.6f1`
- Package:
  - `com.unity.sentis: 2.5.0`
  - `com.unity.ugui: 2.0.0`
  - `com.unity.inputsystem: 1.18.0`
- モデル配置:
  - `Assets/NNG2P/Models/encoder.onnx`
  - `Assets/NNG2P/Models/decoder_step.onnx`
  - `Assets/StreamingAssets/nn-g2p/vocab/*.txt`
  - `Assets/StreamingAssets/nn-g2p/model_meta.json`

## クイックスタート

1. Unity でプロジェクトを開く
2. `Assets/Scenes/NnG2pSampleScene.unity` を開く
3. Play を実行
4. 入力欄に日本語文を入れて `Run AR` を押す

デフォルト入力は `こんにちは、今日はいい天気ですね` です。

## 実装の主要ファイル

- ランタイム本体: `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs`
- サンプルUI: `Assets/Scripts/Sample/NnG2pSampleUiController.cs`
- サンプルシーン制御: `Assets/Scripts/NNG2P/NnG2pSampleSceneController.cs`
- 推論モード定義: `Assets/Scripts/NNG2P/NnG2pInferenceMode.cs`
- 技術調査/ロードマップ: `docs/nn-g2p-model_technical_investigation.md`

## モデルを再取得する場合（任意）

`Assets/StreamingAssets/nn-g2p` を Hugging Face から更新したい場合:

```powershell
uv venv .venv
. .\.venv\Scripts\Activate.ps1
$env:HF_TOKEN="<your_hf_token>"   # private repo の場合
uv run python tools/download_hf_nn_g2p.py --repo ayousanz/nn-g2p-jp
```

ダウンロード後、必要に応じて `Assets/NNG2P/Models/*.onnx` を更新し、  
`NnG2pSentisRuntime` の `ModelAsset` 参照を確認してください。

## テスト

Unity Test Runner で以下を実行できます。

- EditMode: `Assets/Tests/EditMode`
- PlayMode: `Assets/Tests/PlayMode`

uLoop MCP 経由の例（port `8746`）:

```bash
uloop compile -p 8746 --wait-for-domain-reload true
uloop run-tests -p 8746 --test-mode EditMode
uloop run-tests -p 8746 --test-mode PlayMode
```

最新確認（2026-02-25）:

- `compile`: Error `0` / Warning `0`
- `EditMode`: `19/19` Passed
- `PlayMode`: `9/9` Passed

## 注意点

- CTC 実装は削除済みです。`Auto` 指定時も AR に解決されます。
- モデルは文入力を想定し、`maxLen=512` / `fixedDecoderContextLength=512` で動作します。
- ビルド実行時にサンプルを起動するため、`ProjectSettings/EditorBuildSettings.asset` で `Assets/Scenes/NnG2pSampleScene.unity` を先頭に設定しています。
