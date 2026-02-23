# AGENT.md

## 目的
`C:\Users\yuta\Desktop\Private\nn-g2p-model` にあるニューラル G2P モデルを、Unity プロジェクト（このリポジトリ）で動作させる。

このファイルは、将来このリポジトリを編集するエージェント向けの実装ルールと作業手順を定義する。

## 前提（Source of Truth）
- モデル実装: `C:\Users\yuta\Desktop\Private\nn-g2p-model\scripts\train\g2p_utils.py`
- ONNX エクスポート: `C:\Users\yuta\Desktop\Private\nn-g2p-model\scripts\export\export_onnx.py`
- 推奨日本語設定（最新系）:
  - config: `C:\Users\yuta\Desktop\Private\nn-g2p-model\configs\train\ja_m9.yaml`
  - vocab:
    - `C:\Users\yuta\Desktop\Private\nn-g2p-model\configs\vocab\ja_grapheme.txt`
    - `C:\Users\yuta\Desktop\Private\nn-g2p-model\configs\vocab\ja_phones_m8.txt`
    - `C:\Users\yuta\Desktop\Private\nn-g2p-model\configs\vocab\ja_prosody_or_stress_m8.txt`

注意: `nn-g2p-model` リポジトリに checkpoint が含まれていない場合がある。`best_model.pt` は別途配置が必要。

## モデル仕様（Unity 実装に必須）
- Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`（vocab ファイルから ID 解決すること。ID 固定値をハードコードしない）
- Grapheme tokenize:
  - `ja`: 文字単位 `list(text)`
  - `en`: 大文字化して文字単位
- 入力:
  - `src` は grapheme ID 列（BOS/EOS なし）
- 出力:
  - phones 列
  - prosody/stress 列
- 推論モード:
  - 高速: `encoder.onnx + ctc_heads.onnx`
  - 高品質: `encoder.onnx + decoder_step.onnx` の自己回帰ループ

## ONNX 生成ルール
`nn-g2p-model` 側で以下を実行:

```bash
uv run python scripts/export/export_onnx.py \
  --config configs/train/ja_m9.yaml \
  --checkpoint checkpoints/ja_m9/best_model.pt \
  --output-dir exports/ja_m9 \
  --split --opset 15
```

期待される出力:
- `encoder.onnx`
- `ctc_heads.onnx`
- `decoder_step.onnx`

補足:
- Sentis 互換のため `opset 15` を維持する。
- 量子化 ONNX（QDQ）は Sentis 制約に注意。まず非量子化で正しさ確認を優先する。

## Unity 側配置ルール
- ONNX:
  - `Assets/StreamingAssets/nn-g2p/onnx/encoder.onnx`
  - `Assets/StreamingAssets/nn-g2p/onnx/ctc_heads.onnx`
  - `Assets/StreamingAssets/nn-g2p/onnx/decoder_step.onnx`
- vocab:
  - `Assets/StreamingAssets/nn-g2p/vocab/ja_grapheme.txt`
  - `Assets/StreamingAssets/nn-g2p/vocab/ja_phones_m8.txt`
  - `Assets/StreamingAssets/nn-g2p/vocab/ja_prosody_or_stress_m8.txt`
- メタ情報:
  - `Assets/StreamingAssets/nn-g2p/model_meta.json` を置き、使用 config 名・checkpoint 名・vocab ファイル名を記録する。

## Unity 実装ルール
1. 依存パッケージ
- Unity Sentis を導入する（2.5 系を優先）。

2. トークナイズと ID 化
- `ja` は 1 文字ずつ分割して vocab 引き当て。
- 未知文字は `<unk>` にフォールバック。
- `src` には BOS/EOS を付けない。

3. CTC デコード（高速モード）
- `phone_logits` / `prosody_logits` に対して `argmax`。
- CTC collapse を適用:
  - 連続重複を 1 つに縮約
  - `<blank>` を除去
- 最終的に special token を除去。

4. 自己回帰デコード（高品質モード）
- `phone_tokens` / `prosody_tokens` は `<s>` から開始。
- `decoder_step` を反復呼び出しし、各ステップで `argmax` した token を末尾追加。
- `</s>` 到達で停止（両ストリームが終了したら終了）。
- 最大長は config の decode 設定を参照（例: `max_len`, `max_len_ratio`）。

5. リソース管理
- Worker/Tensor は毎推論でリークさせない（必ず dispose）。
- 1 回の推論で encoder は 1 回だけ実行し、memory を decoder で再利用する。

## 受け入れ基準（Done）
- Unity コンパイルエラー 0。
- サンプル入力（例: `東京`, `音声`, `機械学習`）で phones/prosody が空文字にならない。
- 同じ入力で Python 側推論（同 checkpoint・同 vocab）とトークン列が一致、または差分理由を説明可能。
- 例外時にアプリが落ちず、`<unk>` フォールバックで処理継続する。

## 非目標（このリポジトリで今はやらない）
- 学習コードの移植
- 学習再現実験
- checkpoint 生成そのもの

## 作業時の優先順
1. まず ONNX + vocab の整合性確認（モデル名・vocab 名の取り違えを防ぐ）
2. 次に CTC 高速モードを先に通す
3. 最後に decoder_step の自己回帰モードを実装
4. 最終的に Unity 上で回帰テストを追加

