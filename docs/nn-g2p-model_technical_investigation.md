# nn-g2p-model 技術調査・実装方法・ロードマップ

作成日: 2026-02-23  
調査対象: `C:\Users\yuta\Desktop\Private\nn-g2p-model`  
調査方式: 15サブエージェント（観点分割）

## 0. 結論

- 現在の実運用候補は `M9`。`WikiPron PER 2.17% / Prosody F1 0.834`（`docs/eval/milestone_results.md`）。
- Unity連携は既存実装で可能。`scripts/export/export_onnx.py` に `opset=15` の分割ONNX（`encoder.onnx / ctc_heads.onnx / decoder_step.onnx`）がある。
- 先に実装すべきは `CTC高速モード`。その後に `AR高精度モード` を追加するのが安全。
- ドキュメント整合性に差分あり。`README.md` はM4中心で、`docs/eval/milestone_results.md` はM9まで更新済み。

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
| 7 | モデル本体 | `scripts/train/g2p_utils.py` | Pre-LN, shared decoder, conformer, joint CTC対応 | CTC/ARの二段導入が可能 |
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
- 分割後は `encoder -> (ctc_heads | decoder_step反復)` で実行できるため、C#で制御しやすい。
- `--quantize` はONNX Runtimeのdynamic INT8。Sentis実行時の精度・速度は別途実測が必要。

### 2.4 リスク

- ドキュメント更新ズレ: `README.md` と `docs/eval/milestone_results.md` の最新版が一致していない。
- `train` で `wandb.enabled=true` かつ `WANDB_API_KEY` 未設定だと停止する設計。
- 疑似ラベルTSVはprosody空欄になるため、KD比率設計を誤るとprosody品質に影響し得る。

---

## 3. 実装方法（nn-g2p-unity向け）

### 3.1 方針

1. まず `M9` を固定して `CTC高速モード` を実装。  
2. 次に `AR高精度モード`（greedy）を追加。  
3. 最後に `beam` や量子化比較を追加。

### 3.2 実装ステップ

1. **モデル成果物固定**
   - 対象: `configs/train/ja_m9.yaml` + `checkpoints/ja_m9/best_model.pt`
   - エクスポート:
     - `uv run python scripts/export/export_onnx.py --config configs/train/ja_m9.yaml --checkpoint checkpoints/ja_m9/best_model.pt --output-dir exports/ja_m9_sentis --split --opset 15`

2. **Unity配置**
   - `encoder.onnx`, `ctc_heads.onnx`, `decoder_step.onnx`
   - `configs/vocab/ja_grapheme_m4.txt`, `configs/vocab/ja_phones_m8.txt`, `configs/vocab/ja_prosody_or_stress_m8.txt`
   - Unity側配置先例: `Assets/StreamingAssets/g2p/`

3. **Unity実装（最小）**
   - `Tokenizer`: grapheme→id
   - `EncoderRunner`: src→memory
   - `CTCDecoder`: memory→phone/prosody（collapse+blank除去）
   - `Detokenizer`: id→token列

4. **Unity実装（拡張）**
   - `ARDecoder`: `decoder_step.onnx` を反復呼び出ししてgreedy生成
   - `max_len_ratio`/`repetition_penalty` を Python 実装相当に合わせる

5. **同値性検証**
   - Python baseline: `scripts/train/eval_g2p.py` で同一入力の推論結果を保存
   - Unity出力との差分を token 単位で比較（まず100語→1,000語）

6. **評価正規化**
   - ベンチ比較時は `scripts/data/normalize_ipa.py` と同等ルールを適用
   - 特に `e i -> e:`、`Cʲ/Cʷ` 分解、撥音異音正規化を一致させる

---

## 4. 実行ロードマップ

### Phase 0: 固定化（0.5日）
- 成果物: モデル版固定（M9）、評価データ固定、エクスポート手順固定
- 完了条件: 同一入力でPython結果が再現可能

### Phase 1: Unity CTC実装（2日）
- 成果物: CTC高速モードでphones/prosody出力
- 完了条件: 100語でPython CTCとの差分率 < 1%

### Phase 2: Unity AR実装（3日）
- 成果物: decoder_step反復によるgreedy推論
- 完了条件: 100語でPython greedyとの差分率 < 1%

### Phase 3: 評価接続（1.5日）
- 成果物: Unity出力TSVを `eval_per/eval_prosody` で自動評価
- 完了条件: PER/Prosody F1を自動レポート化

### Phase 4: 最適化（2日）
- 成果物: FP16/INT8候補比較、レイテンシ/メモリ計測
- 完了条件: 目標端末で実行可能な構成を1つ確定

### Phase 5: 継続改善（継続）
- 成果物: 疑似ラベル更新→再学習→再エクスポートの定期運用
- 完了条件: モデル更新を同じCI手順で再現可能

---

## 5. 優先順位（推奨）

1. `M9 + CTC高速モード` を先に出す  
2. `M9 + AR greedy` を追加して品質を上げる  
3. 量子化・beam・自己学習ループは最後に回す

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
  - 同一入力集合で token 列一致率を算出（CTC/AR別）
  - 不一致ケースの上位N件を保存
  - `normalize_ipa.py` 相当ルール適用後の PER/Prosody F1 を比較
- 結論: 文字列一致だけでなく評価指標の一致を必須化する必要がある。

### Agent E: 実装ロードマップレビュー

- 指摘: 既存ロードマップは妥当だが、認証ブロッカー（HF private）が工程に未反映。  
- 改善: Phase 0 に「HF認証確認」を追加し、失敗時は即時エスカレーション。  
- 結論: 実行順は以下が安全:
  1. HF認証確認  
  2. モデル取得 + SHA固定  
  3. Unity CTC  
  4. Unity AR  
  5. Python同値検証

### レビュー総括

- 文書の技術方向性は正しい。  
- ただし実運用には「配布元認証」「実際のUnity API系統」「同値性判定基準」の3点を明文化する必要がある。  
- 上記3点を反映すれば、実装と検証の手戻りを大きく減らせる。
