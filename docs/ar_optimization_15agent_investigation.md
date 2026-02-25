# AR推論高速化 技術調査（15-agent）

更新日: 2026-02-25  
対象: `nn-g2p-unity` ARランタイム  
方針: 最初に調査、その後P0/P1から段階的に実装

## 1. 事実ベースの現状整理（実装前スナップショット）

- ARデコーダは1ステップごとに `decoder_step` を呼び出している。  
  参照: `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs:251`
- 各ステップで `phone_tokens/prosody_tokens` テンソルを都度生成している。  
  参照: `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs:253`
- 各ステップで `phone/prosody logits` を `ReadbackAndClone` している。  
  参照: `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs:262`, `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs:394`
- エンコーダ出力 `memory` も `ReadbackAndClone` している。  
  参照: `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs:219`
- サンプルシーンは CPU backend 設定。  
  参照: `Assets/Scenes/NnG2pSampleScene.unity:921`
- `decoder_step.onnx` は full-sequence logits を返す運用メタデータになっている。  
  参照: `Assets/StreamingAssets/nn-g2p/download_manifest.json:12`, `Assets/StreamingAssets/nn-g2p/model_meta.json:26`

補足（調査時にONNXをローカル解析）:

- `decoder_step.onnx` 入力: `memory[1,512,384]`, `src_pad_mask[1,512]`, `phone_tokens[1,512]`, `prosody_tokens[1,512]`
- `decoder_step.onnx` 出力: `phone_logits[1,512,46]`, `prosody_logits[1,512,52]`
- ノード数: `decoder_step=1882`, `encoder=1449`

軽量ベンチ（ONNX Runtime CPU, 同環境）:

- `encoder`: 約 `75.66 ms`
- `decoder_step` 1回: 約 `203.11 ms`

## 2. 15エージェント調査結果

注記:
- 本章の観測は実装前スナップショット。第1弾実装で一部は解消済み。

### Agent 1: Runtime Loop
- 観測: 1トークンごとに重い `decoder_step` を再実行。
- 判断: AR本質の逐次性は不可避だが、1stepコストが過大。
- 優先度: P0

### Agent 2: Tensor Allocation
- 観測: ステップごとの `new Tensor<int>` がGC圧を増やす。
- 判断: 再利用化でCPU負荷と割り込みを低減可能。
- 優先度: P1

### Agent 3: Output Readback
- 観測: `ReadbackAndClone` を毎step・2ストリームで実施。
- 判断: backendがGPUの場合、readback待ちが律速になりやすい。
- 優先度: P0

### Agent 4: Encoder Memory Handling
- 観測: `memory` をreadbackしてCPU側へ固定化。
- 判断: end-to-endで同一backendを維持できず、GPU有効時に不利。
- 優先度: P1

### Agent 5: ONNX I/O設計
- 観測: `decoder_step` が full-sequence logits を返却。
- 判断: 推論で必要なのは通常 `last-step logits` のみ。
- 優先度: P0

### Agent 6: Shape Policy
- 観測: `src/tgt` が固定長512。
- 判断: 短文でも512計算になるため、無駄演算が大きい。
- 優先度: P0

### Agent 7: Python実装整合
- 観測: 元 `export_onnx.py` では decoder step は `[:, -1, :]` を返す設計。
  参照: `C:/Users/yuta/Desktop/Private/nn-g2p-model/scripts/export/export_onnx.py:102`
- 判断: Unity配布モデル側の仕様差がレイテンシ増要因。
- 優先度: P0

### Agent 8: Backend戦略
- 観測: 現在CPU固定で運用。
- 判断: 端末依存だが、GPUCompute比較は必須。
- 優先度: P1

### Agent 9: Operator互換
- 観測: `Trilu`, `MatMul`, `Softmax`, `Gather` 等を多用。
- 判断: backend別で実行効率差が出るため、プロファイル前提で判断すべき。
- 優先度: P1

### Agent 10: Decode Length Policy
- 観測: `maxLen=512`, `maxLenRatio=3.0`。
  参照: `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs:26`, `Assets/Scripts/NNG2P/NnG2pSentisRuntime.cs:27`
- 判断: 実運用入力長の分布に合わせた上限チューニング余地が大きい。
- 優先度: P1

### Agent 11: Repetition Penalty Cost
- 観測（実装前）: 毎stepで `HashSet` を作成してペナルティ判定。
- 判断: 全体比では小さいが、軽微なCPU最適化余地あり。第1弾で `bool[]` 化を適用済み。
- 優先度: P2

### Agent 12: Vocab/Decode Postprocess
- 観測: 後処理は相対的に軽い。
- 判断: 主要律速ではないため優先度低。
- 優先度: P3

### Agent 13: Benchmark Design
- 観測: 現在は体感ベースが中心。
- 判断: `tokens/sec`, `p50/p95`, 入力長別カーブを定義すべき。
- 優先度: P0

### Agent 14: Regression Guard
- 観測: 速度最適化は品質回帰リスクを伴う。
- 判断: Python基準との一致率/指標比較を同時に回す必要あり。
- 優先度: P0

### Agent 15: Rollout/運用
- 観測: モデル更新時に再現手順はあるが性能ゲートは弱い。
- 判断: 「性能しきい値を満たさない更新をブロック」する運用が必要。
- 優先度: P1

## 3. 優先度付き最適化バックログ（実装候補）

## P0（最優先）
- `decoder_step.onnx` を「可変 `tgt_len` + last-step出力」に戻す/再設計する。
- Unity側readback回数を最小化する（毎step full cloneを避ける方針）。
- 入力長別の性能計測基盤を先に固定する（最適化効果を可視化するため）。
- 品質ガード（Python基準との一致率/PER/Prosody F1）を同時運用する。

## P1（次点）
- backend比較（CPU vs GPUCompute）を端末別に実施して採用backendを確定。
- Tensor再利用方針を導入してアロケーションを削減。
- `maxLen/maxLenRatio` を運用データ分布ベースで再設定。
- リリースゲートに性能SLOを追加。

## P2（中長期）
- Repetition penalty実装の軽量化（集合再生成回避）。
- 必要ならincremental decode（KV cache型）を検討。

## 4. 期待効果（保守的レンジ）

- P0適用で: 体感レイテンシの大幅改善（最有力）。
- P1適用で: 端末ごとの安定改善とGC由来スパイク抑制。
- P2適用で: 小幅な上積み。

注記:
- ここでの効果は推定。最終判断は同一端末・同一入力セットでの実測を基準にする。

## 5. 実施順（提案）

1. 計測指標を固定（入力長別ベンチ、p50/p95、tokens/sec）
2. ONNX I/O仕様の見直し方針を確定（last-step + dynamic）
3. Unity側のreadback/alloc最小化方針を設計
4. backend採用方針を端末別に確定
5. 品質ガードを含めて最終採用判定

## 6. 今回の対応範囲

- 実施（第1弾）:
  - 調査、論点整理、優先度付け、ロードマップ化
  - Unityランタイム軽量化
    - デコードループでの `Tensor` 再生成を削減（トークンテンソル再利用）
    - CPU backend 時は `PeekOutput + CompleteAllPendingOperations` を使い、毎step `ReadbackAndClone` を回避
    - repetition penalty 用の `HashSet` 生成を廃止し、`bool[]` マップに変更
    - Worker生成時に `GPUCompute -> CPU` フォールバックを追加
  - サンプルシーンのbackend初期値を `GPUCompute` に変更
- 未実施:
  - ONNX再設計（dynamic tgt_len / last-step 出力の安定化）
  - 改善前後の同一端末ベンチマーク比較（Unity実測の再計測）

補足（ブロッカー）:
- upstream `export_onnx.py` で dynamic `tgt_len` をそのまま出すと、decoder内部 `Reshape` が固定長解釈されるケースを確認。
- そのため、ONNX再設計は「エクスポート手順の安定化」とセットで次フェーズ対応とする。
