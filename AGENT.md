# AGENT.md

## 目的
`nn-g2p-unity` で日本語 G2P 推論（phones / prosody）を安定運用する。  
このファイルは、本リポジトリを編集するエージェント向けの最新ルール。

## 現在の実装方針（最重要）
- 推論モードは **AR（Autoregressive）のみ**。
- **CTC 実装は削除済み**。再導入は明示指示がある場合のみ。
- 対応言語は **日本語運用前提**（現行モデルが `ja_m9`）。
- `C:\Users\yuta\Desktop\Private\nn-g2p-model` は **参照のみ**。変更禁止。

## Source of Truth
- ランタイム実装:
  - `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs`
- サンプルUI:
  - `Assets/Scripts/Sample/NnG2pSampleUiController.cs`
  - `Assets/Scripts/NNG2P/NnG2pSampleSceneController.cs`
- サンプルシーン:
  - `Assets/Scenes/NnG2pSampleScene.unity`
- 技術ドキュメント:
  - `docs/nn-g2p-model_technical_investigation.md`
  - `docs/gpu_nan_operator_investigation_2026-02-26.md`

## モデル・配置ルール
- ONNX（Unityで使用）:
  - `Assets/NNG2P/Models/encoder.onnx`
  - `Assets/NNG2P/Models/decoder_step.onnx`
- StreamingAssets（再現用アセット）:
  - `Assets/StreamingAssets/nn-g2p/onnx/*.onnx`
  - `Assets/StreamingAssets/nn-g2p/vocab/*.txt`
  - `Assets/StreamingAssets/nn-g2p/model_meta.json`
- vocab ファイル名（現行）:
  - `ja_grapheme_m4.txt`
  - `ja_phones_m8.txt`
  - `ja_prosody_or_stress_m8.txt`

## 推論仕様（実装時の固定事項）
- Special token は vocab から解決する（ハードコード禁止）:
  - `<pad>`, `<unk>`, `<s>`, `</s>`
- `src` は grapheme ID 列（BOS/EOSなし）。
- `ja` は文字単位 tokenize。
- デフォルト長:
  - `fixedEncoderInputLength = 512`
  - `fixedDecoderContextLength = 512`

## GPU運用ポリシー（2026-02-26更新）
- 現行環境（Unity 6000.3.6f1 + `com.unity.ai.inference` 2.4.1）では、
  現行モデルで GPU backend が NaN を返す既知問題あり。
- `NnG2pSentisRuntime` は以下を実装済み:
  - GPU優先初期化（利用可能時）
  - 初期化時の数値健全性チェック
  - NaN 検出時の即 CPU フォールバック
- 期待ログ:
  - `Backend GPUCompute failed numeric validation ... Falling back to CPU backend.`
  - 推論ログは `backend=CPU`

## UI/入力ルール
- UIコンポーネントは TextMeshPro を使用する。
- デフォルト入力:
  - `こんにちは、今日はいい天気ですね`
- IME 入力を有効化する（日本語入力）。
- フォントはシーン上で `Assets/Fonts/NotoSansSC-Regular SDF.asset` を使用。

## MCP / uloop 運用
- 接続ポートは **8746**。
- 基本コマンド:
  - `uloop compile -p 8746 --wait-for-domain-reload true`
  - `uloop run-tests -p 8746 --test-mode EditMode`
  - `uloop run-tests -p 8746 --test-mode PlayMode`
- Play確認時は、サンプルシーンで実行して `backend=` を含むログを確認する。

## モデル再取得（必要時のみ）
- Hugging Face: `ayousanz/nn-g2p-jp`
- ダウンロードツール:
  - `tools/download_hf_nn_g2p.py`
- 実行例:
  - `uv venv .venv`
  - `.\\.venv\\Scripts\\Activate.ps1`
  - `$env:HF_TOKEN="<token>"`
  - `uv run python tools/download_hf_nn_g2p.py --repo ayousanz/nn-g2p-jp`
- 注意:
  - トークンや `.env` はコミットしない。
  - 検証用の candidate ONNX はコミットしない。

## 受け入れ基準（Done）
- Unity compile: Error 0
- EditMode テスト成功（最低）
- サンプル文で phones/prosody が空でない
- 文書入力で途中停止しない
- docs が実装状態と一致している

## 作業順の推奨
1. `README.md` と `docs/` の方針確認
2. 実コード更新（AR専用・日本語運用を維持）
3. `uloop compile` / テスト実行
4. サンプルシーン実行確認
5. docs 更新
