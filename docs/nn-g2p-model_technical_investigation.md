# nn-g2p-model 技術調査・実装方法・ロードマップ

作成日: 2026-02-23  
調査対象: `C:\Users\yuta\Desktop\Private\nn-g2p-model`  
調査方式: 15サブエージェント（観点分割）

注記（2026-02-25更新）: Unity実装はAR専用。本文はAR専用構成に更新済み。

## 0. 結論

- 現在の実運用候補は `M9`。`WikiPron PER 2.17% / Prosody F1 0.834`（`docs/eval/milestone_results.md`）。
- Unity連携は既存実装で可能。`scripts/export/export_onnx.py` に `opset=15` の分割ONNX（`encoder.onnx / decoder_step.onnx`）がある。
- Unity運用は `AR専用モード` で固定し、Python同値性と品質検証を優先する。
- 現在同梱モデル（`ja_m9`）は日本語前提。英語入力を高品質で扱うには別チェックポイントまたは前処理正規化が必要。

---

## 1. 15サブエージェント調査ログ

| # | 担当 | 主対象 | 主な発見 | 実装示唆 |
|---|---|---|---|---|
| 1 | リポジトリ構造 | `README.md`, `pyproject.toml` | 学習・評価・変換・エクスポートが分離済み | Unity側は推論層のみ切り出し可能 |
| 2 | 要件定義 | `docs/specs/requirements.md` | IPA + prosody/stress の2ストリームが確定仕様 | Unity側も2ヘッド前提にする |
| 3 | 英語データ生成 | `scripts/train/build_train_dataset_en.py` | CMU+MFA統合、ARPAbet→IPA変換、eval除外対応 | 将来EN追加時に同パイプライン流用可 |
| 4 | 日本語単語データ生成 | `scripts/train/build_train_dataset_ja.py` | Sudachi OOV分類 + pyopenjtalk/marine対応 | JA更新学習の再現性が高い |
| 5 | 日本語文データ生成 | `scripts/train/build_sentence_dataset_ja.py` | Wikipedia/Aozora由来の文生成 + 並列前処理 | 文脈付き学習に必要な資産が揃っている |
| 6 | vocab生成 | `scripts/train/build_vocab.py` | grapheme/phones/prosodyを別管理 | Unityで3 vocab を個別ロードする設計が必要 |
| 7 | モデル本体 | `scripts/train/g2p_utils.py` | Pre-LN, shared decoder, conformer, joint CTC対応 | Unity運用はAR専用で導入可能 |
| 8 | 学習ループ | `scripts/train/train_g2p.py` | DDP, KD, R-Drop, class-weight focal、resume時vocab拡張対応 | 継続改善の運用設計がしやすい |
| 9 | 評価ループ | `scripts/train/eval_g2p.py` | PER/WER/Prosody F1、OOV分解、詳細解析あり | Unity出力の同一指標比較が容易 |
| 10 | IPA正規化 | `scripts/data/normalize_ipa.py` | `e i -> e:` 等の評価整合ルールが実装済み | Unity側の評価時にも同ルールを移植 |
| 11 | WikiPron変換 | `scripts/data/convert_wikipron.py` | narrow IPA→内部phone変換が高機能 | ベンチ再現はこの変換を必須化 |
| 12 | 外部モデル比較 | `scripts/eval/benchmark_charsiug2p.py` | Charsiu/fdemelo比較を同一条件で実施可能 | 競合比較の定点観測が可能 |
| 13 | ONNX分割出力 | `scripts/export/export_onnx.py` | Sentis向け分割出力+onnxruntime差分検証 | Unity実装へ直接接続可能 |
| 14 | 量子化評価 | `scripts/export/compare_quantization.py` | FP32/FP16/INT8サイズ比較あり（ONNX実推論未実装） | サイズ判断の初期資料として有効 |
| 15 | テスト資産 | `tests/test_*.py` | モデル機能・正規化・データ処理の単体/スモークあり | 回帰検証基盤は既に十分 |

---

## 2. 技術調査サマリ

### 2.1 モデル/学習

- `G2PModel` は `shared_decoder + dual head` と `joint_ctc` を併用可能（`scripts/train/g2p_utils.py`）。
- `FocalLoss` はクラス別重み（tensor alpha）対応済み（`scripts/train/g2p_utils.py`）。
- 学習は `KD`（疑似ラベル）・`R-Drop`・`curriculum`・`DDP` を切替可能（`scripts/train/train_g2p.py`）。
- resume時に埋め込み/出力層の語彙拡張を自動吸収（`scripts/train/train_g2p.py`）。

### 2.2 評価/再現性

- 標準評価は `PER/WER + Prosody F1(or Stress)`。OOV層別と混同行列まで出せる（`scripts/train/eval_g2p.py`）。
- `normalize_ipa.py` の規則が評価値に大きく影響。WikiPron比較は同一正規化を固定すべき。
- `eval_g2p.py` の splitは `dev/test` のみ。WikiPronやSIGMORPHONは別スクリプト運用。

### 2.3 デプロイ適性（Unity/Sentis）

- `export_onnx.py` は `--split --opset 15` がデフォルト推奨で、Sentis向け実装と整合。
- 分割後は `encoder -> decoder_step反復`（AR）で実行できるため、C#で制御しやすい。
- `--quantize` はONNX Runtimeのdynamic INT8。Sentis実行時の精度・速度は別途実測が必要。

### 2.4 リスク

- ドキュメント更新ズレ: モデル更新時に `README.md` / `docs` / `StreamingAssets` メタ情報の同期漏れが起こりやすい。
- `train` で `wandb.enabled=true` かつ `WANDB_API_KEY` 未設定だと停止する設計。
- 疑似ラベルTSVはprosody空欄になるため、KD比率設計を誤るとprosody品質に影響し得る。

### 2.5 言語対応（現行モデル）

- Pythonコードベースは `lang: ja/en` の分岐を持つ（`scripts/train/g2p_utils.py`）。
- ただし現在運用しているモデル設定は `configs/train/ja_m9.yaml` で、語彙も `ja_*`（`ja_grapheme_m4.txt` など）を使用。
- そのため Unity 実装は現時点で「日本語入力前提」の扱いが正しい。
- 英語/英数字を含む入力に対しては `<unk>` 化で情報欠落が起こり得るため、必要なら事前正規化を導入する。

---

## 3. 実装方法（nn-g2p-unity向け）

### 3.1 方針

1. `M9` を固定して `ARモード` を実装・維持。  
2. Python実装との同値性検証を優先。  
3. その後に `beam` や量子化比較を追加。

### 3.2 実装ステップ

1. **モデル成果物固定**
   - 対象: `configs/train/ja_m9.yaml` + `checkpoints/ja_m9/best_model.pt`
   - エクスポート:
     - `uv run python scripts/export/export_onnx.py --config configs/train/ja_m9.yaml --checkpoint checkpoints/ja_m9/best_model.pt --output-dir exports/ja_m9_sentis --split --opset 15`

2. **Unity配置**
   - `encoder.onnx`, `decoder_step.onnx`（`Assets/NNG2P/Models/` に配置し `ModelAsset` として読み込み）
   - `configs/vocab/ja_grapheme_m4.txt`, `configs/vocab/ja_phones_m8.txt`, `configs/vocab/ja_prosody_or_stress_m8.txt`
   - `model_meta.json` と vocab は `Assets/StreamingAssets/nn-g2p/` に配置

3. **Unity実装（最小）**
   - `Tokenizer`: grapheme→id
   - `EncoderRunner`: src→memory
   - `ARDecoder`: `decoder_step.onnx` を反復呼び出ししてgreedy生成
   - `Detokenizer`: id→token列

4. **Unity実装（拡張）**
   - `max_len_ratio`/`repetition_penalty` を Python 実装相当に合わせる
   - 必要時に `beam` と量子化比較を追加

5. **同値性検証**
   - Python baseline: `scripts/train/eval_g2p.py` で同一入力の推論結果を保存
   - Unity出力との差分を token 単位で比較（まず100語→1,000語）

6. **評価正規化**
   - ベンチ比較時は `scripts/data/normalize_ipa.py` と同等ルールを適用
   - 特に `e i -> e:`、`Cʲ/Cʷ` 分解、撥音異音正規化を一致させる

---

## 4. 実行ロードマップ（最新版: 2026-02-25）

### 4.1 現在地

- `完了`: HFモデル取得、分割ONNX生成、Unity取り込み（AR専用）  
  - 元モデル: `https://huggingface.co/ayousanz/nn-g2p-jp`
  - 配置先: `Assets/NNG2P/Models/encoder.onnx`, `Assets/NNG2P/Models/decoder_step.onnx`
- `完了`: 再現性メタデータ記録  
  - `Assets/StreamingAssets/nn-g2p/download_manifest.json`
  - `repo_sha`: `807c3a29fd7b0211545a6e80c032e78c3b6eea7f`
- `完了`: Unityランタイム安定化  
  - `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs`
  - `fixedEncoderInputLength=512`, `fixedDecoderContextLength=512` を実装
- `完了`: 長文入力対応（Python実装に合わせた文入力）  
  - `ja_m9` チェックポイントから分割ONNXを再エクスポート（`src_len seed=512`）
  - `encoder.onnx` / `decoder_step.onnx` の内部固定長を 128 -> 512 に更新
- `完了`: Unity検証  
  - コンパイル: ErrorCount=0（port `8746`）
  - テスト: EditMode `19/19` / PlayMode `9/9` pass（2026-02-25）
  - スモーク: `こんにちは、今日はいい天気ですね` で AR 推論ログを確認（phones/prosodyとも出力）
- `未完了`: Python同値検証（100語/1000語）と指標比較

### Phase 0: 認証確認 + モデル取得（完了）
- 成果物:
  - ONNX 2モデル（`encoder` / `decoder_step`）を `ModelAsset` として導入
  - `download_manifest.json` に repo SHA とファイルSHA256を記録
- 完了条件:
  - Unity上で2モデル読込が可能
  - 取得元追跡情報が保持される

### Phase 1: Unity推論スモーク（完了）
- 成果物:
  - AR 推論実行ルートを実装
  - 入力3サンプルで例外なしを確認
- 完了条件:
  - `Predict` の AR 分岐が動作する

### Phase 2: Python同値検証（次フェーズ）
- 成果物:
  - 同一入力100語で Unity/Python 比較CSV（AR）
  - 差分上位ケース（入力・Python・Unity）レポート
- 完了条件:
  - AR greedy差分率 < 1%
- 実装タスク:
  - Python基準出力生成（`uv run python ...`）
  - Unity側バッチ推論出力（同一語彙）
  - 正規化込み比較スクリプト実行

### Phase 3: 評価接続（未着手）
- 成果物:
  - Unity出力TSVを `eval_g2p.py` 相当指標で自動評価
  - PER/WER/Prosody F1 レポート
- 完了条件:
  - 指標比較が手動なしで再実行できる

### Phase 4: 最適化（未着手）
- 成果物:
  - backend別（CPU/GPU）レイテンシ・メモリ計測
  - FP32/FP16/量子化候補の比較表
- 完了条件:
  - 目標端末向け採用構成を1つ確定

### Phase 5: 運用化（継続）
- 成果物:
  - モデル更新時の再取得→再検証→反映手順
  - 失敗時ロールバック手順
- 完了条件:
  - 次回更新で同手順を再利用できる

### 4.2 直近スケジュール（提案）

1. 2026-02-24: 100語セットで AR 同値検証を完了  
2. 2026-02-25: 100語セットの差分分析を完了  
3. 2026-02-26: 1000語拡張と差分分析レポートを完了

---

## 5. 優先順位（更新）

1. 100語セットの Python/Unity 同値検証（AR）  
2. 1000語拡張と差分ケース分析（正規化込み）  
3. PER/WER/Prosody F1 の自動評価接続

---

## 6. 5専門サブエージェントレビュー（追補, 2026-02-23）

レビュー対象: `docs/nn-g2p-model_technical_investigation.md`  
レビュー方式: 5観点の専門サブエージェント

### Agent A: ML再現性レビュー

- 指摘: 「M9固定」の記述は妥当だが、配布元がHugging Face運用に移行した前提が不足。  
- 改善: `repo_id + commit sha` を必ず記録し、Unity側に `download_manifest.json` を残す運用を必須化。  
- 結論: 再現性担保には「モデルID」ではなく「repo+sha+vocab名」の三点固定が必要。

### Agent B: Unityデプロイレビュー

- 指摘: ドキュメント上は Sentis 前提だが、Unity 6000.3.6f1 では `com.unity.sentis` が shim として解決され、実APIは `Unity.InferenceEngine` になっている。  
- 改善: C#実装は `Unity.InferenceEngine.Worker / Tensor<T>` を使用し、`Unity.Sentis` 直参照を避ける。  
- 結論: 「Sentis 2.5想定」だけでは不足で、実行環境の実解決バージョンを明記すべき。

### Agent C: 配布・セキュリティレビュー

- 指摘: 指定 repo `ayousanz/nn-g2p-jp` は匿名アクセスで `401`。  
- 改善: `HF_TOKEN` を使う公式ダウンローダを用意し、Private repo 前提の運用手順を docs に残す。  
- 結論: 現状は認証情報なしでは「ダウンロードして使う」が完遂できない。

### Agent D: 品質同値性レビュー

- 指摘: 「Python同品質」の定義が抽象的。  
- 改善: 少なくとも下記を完了条件に固定:
  - 同一入力集合で token 列一致率を算出（AR）
  - 不一致ケースの上位N件を保存
  - `normalize_ipa.py` 相当ルール適用後の PER/Prosody F1 を比較
- 結論: 文字列一致だけでなく評価指標の一致を必須化する必要がある。

### Agent E: 実装ロードマップレビュー

- 指摘: 既存ロードマップは妥当だが、認証ブロッカー（HF private）が工程に未反映。  
- 改善: Phase 0 に「HF認証確認」を追加し、失敗時は即時エスカレーション。  
- 結論: 実行順は以下が安全:
  1. HF認証確認  
  2. モデル取得 + SHA固定  
  3. Unity AR  
  4. Python同値検証  
  5. 指標評価と運用手順の確定

### レビュー総括

- 文書の技術方向性は正しい。  
- ただし実運用には「配布元認証」「実際のUnity API系統」「同値性判定基準」の3点を明文化する必要がある。  
- 上記3点を反映すれば、実装と検証の手戻りを大きく減らせる。
