using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace NnG2p.Runtime
{
    public sealed class NnG2pSentisRuntime : MonoBehaviour, IDisposable
    {
        private const string StreamingAssetsRoot = "nn-g2p";
        private const string VocabFolder = "vocab";

        [Header("ONNX ModelAssets")]
        [SerializeField] private ModelAsset encoderModelAsset;
        [SerializeField] private ModelAsset ctcHeadsModelAsset;
        [SerializeField] private ModelAsset decoderStepModelAsset;

        [Header("Runtime")]
        [SerializeField] private BackendType backendType = BackendType.CPU;
        [SerializeField] private NnG2pInferenceMode defaultMode = NnG2pInferenceMode.Autoregressive;
        [SerializeField] private string language = "ja";
        [SerializeField] private int maxLen = 512;
        [SerializeField] private float maxLenRatio = 3.0f;
        [SerializeField] private float repetitionPenalty = 1.2f;
        [SerializeField] private int fixedEncoderInputLength = 128;
        [SerializeField] private int fixedDecoderContextLength = 3;

        [Header("Vocab Files (StreamingAssets/nn-g2p/vocab)")]
        [SerializeField] private string graphemeVocabFile = "ja_grapheme_m4.txt";
        [SerializeField] private string phoneVocabFile = "ja_phones_m8.txt";
        [SerializeField] private string prosodyVocabFile = "ja_prosody_or_stress_m8.txt";

        private Worker _encoderWorker;
        private Worker _ctcHeadsWorker;
        private Worker _decoderStepWorker;

        private NnG2pVocab _graphemeVocab;
        private NnG2pVocab _phoneVocab;
        private NnG2pVocab _prosodyVocab;

        private bool _isInitialized;
        private string _lastInitError;

        public bool IsInitialized => _isInitialized;
        public string LastInitError => _lastInitError;

        private void OnDestroy()
        {
            Dispose();
        }

        public void Dispose()
        {
            DisposeWorker(ref _encoderWorker);
            DisposeWorker(ref _ctcHeadsWorker);
            DisposeWorker(ref _decoderStepWorker);
            _isInitialized = false;
        }

        public bool TryInitialize(out string error)
        {
            Dispose();

            try
            {
                if (encoderModelAsset == null)
                {
                    throw new InvalidOperationException("Encoder ModelAsset is not assigned.");
                }

                _graphemeVocab = NnG2pVocab.LoadFromFile(ResolveVocabPath(graphemeVocabFile));
                _phoneVocab = NnG2pVocab.LoadFromFile(ResolveVocabPath(phoneVocabFile));
                _prosodyVocab = NnG2pVocab.LoadFromFile(ResolveVocabPath(prosodyVocabFile));

                var encoderModel = ModelLoader.Load(encoderModelAsset);
                _encoderWorker = new Worker(encoderModel, backendType);

                if (ctcHeadsModelAsset != null)
                {
                    var ctcModel = ModelLoader.Load(ctcHeadsModelAsset);
                    _ctcHeadsWorker = new Worker(ctcModel, backendType);
                }

                if (decoderStepModelAsset != null)
                {
                    var decoderModel = ModelLoader.Load(decoderStepModelAsset);
                    _decoderStepWorker = new Worker(decoderModel, backendType);
                }

                _isInitialized = true;
                _lastInitError = string.Empty;
                error = string.Empty;
                return true;
            }
            catch (Exception ex)
            {
                _isInitialized = false;
                _lastInitError = ex.Message;
                error = ex.Message;
                return false;
            }
        }

        public NnG2pInferenceResult Predict(string text, NnG2pInferenceMode? modeOverride = null)
        {
            if (!_isInitialized)
            {
                if (!TryInitialize(out var initError))
                {
                    throw new InvalidOperationException($"Initialization failed: {initError}");
                }
            }

            var graphemes = TokenizeInput(text);
            var srcIds = _graphemeVocab.EncodeTokens(graphemes, addBosEos: false);

            if (srcIds.Length == 0)
            {
                return new NnG2pInferenceResult
                {
                    Input = text,
                    Graphemes = graphemes.ToArray(),
                    SourceIds = srcIds,
                    Mode = ResolveMode(modeOverride),
                };
            }

            var encoderInput = BuildEncoderInput(srcIds, out var srcPadMaskValues, out var effectiveSrcLen);
            using var memory = RunEncoder(encoderInput);
            var mode = ResolveMode(modeOverride);

            return mode switch
            {
                NnG2pInferenceMode.Ctc => DecodeCtc(text, graphemes, srcIds, effectiveSrcLen, memory),
                NnG2pInferenceMode.Autoregressive => DecodeAutoregressive(text, graphemes, srcIds, srcPadMaskValues, effectiveSrcLen, memory),
                _ => throw new InvalidOperationException($"Unsupported mode: {mode}"),
            };
        }

        [ContextMenu("Test Inference (tokyo)")]
        private void TestInferenceTokyo()
        {
            try
            {
                var result = Predict("tokyo", defaultMode);
                Debug.Log($"Mode={result.Mode}, phones={string.Join(" ", result.Phones)}, prosody={string.Join(" ", result.Prosody)}");
            }
            catch (Exception ex)
            {
                Debug.LogError(ex);
            }
        }

#if UNITY_EDITOR
        [ContextMenu("Auto Assign ONNX ModelAssets")]
        private void AutoAssignOnnxModelAssets()
        {
            encoderModelAsset = LoadModelAssetPreferImported("encoder.onnx");
            ctcHeadsModelAsset = LoadModelAssetPreferImported("ctc_heads.onnx");
            decoderStepModelAsset = LoadModelAssetPreferImported("decoder_step.onnx");
            EditorUtility.SetDirty(this);
            Debug.Log("Auto-assign completed for ONNX ModelAssets under Assets/StreamingAssets/nn-g2p/onnx.");
        }
#endif

        private NnG2pInferenceMode ResolveMode(NnG2pInferenceMode? modeOverride)
        {
            var requested = modeOverride ?? defaultMode;
            if (requested != NnG2pInferenceMode.Auto)
            {
                return requested;
            }

            if (_decoderStepWorker != null)
            {
                return NnG2pInferenceMode.Autoregressive;
            }

            return NnG2pInferenceMode.Ctc;
        }

        private int[] BuildEncoderInput(
            IReadOnlyList<int> srcIds,
            out byte[] srcPadMaskValues,
            out int effectiveSourceLength)
        {
            if (fixedEncoderInputLength <= 0)
            {
                effectiveSourceLength = srcIds.Count;
                srcPadMaskValues = new byte[srcIds.Count];
                return srcIds.ToArray();
            }

            effectiveSourceLength = Mathf.Min(srcIds.Count, fixedEncoderInputLength);
            if (srcIds.Count > fixedEncoderInputLength)
            {
                Debug.LogWarning($"Input length {srcIds.Count} exceeds fixedEncoderInputLength={fixedEncoderInputLength}. Truncating input.");
            }

            var padded = Enumerable.Repeat(_graphemeVocab.PadId, fixedEncoderInputLength).ToArray();
            srcPadMaskValues = Enumerable.Repeat((byte)1, fixedEncoderInputLength).ToArray();
            for (var i = 0; i < effectiveSourceLength; i++)
            {
                padded[i] = srcIds[i];
                srcPadMaskValues[i] = 0;
            }

            return padded;
        }

        private Tensor<float> RunEncoder(IReadOnlyList<int> srcIds)
        {
            var srcShape = new TensorShape(1, srcIds.Count);
            using var srcTensor = new Tensor<int>(srcShape, srcIds.ToArray());

            _encoderWorker.SetInput("src", srcTensor);
            _encoderWorker.Schedule();

            var memoryOutput = _encoderWorker.PeekOutput("memory") as Tensor<float>;
            if (memoryOutput == null)
            {
                memoryOutput = _encoderWorker.PeekOutput() as Tensor<float>;
            }

            if (memoryOutput == null)
            {
                throw new InvalidOperationException("Encoder output 'memory' could not be read as Tensor<float>.");
            }

            return memoryOutput.ReadbackAndClone();
        }

        private NnG2pInferenceResult DecodeCtc(
            string input,
            IReadOnlyList<string> graphemes,
            IReadOnlyList<int> srcIds,
            int effectiveSrcLen,
            Tensor<float> memory)
        {
            if (_ctcHeadsWorker == null)
            {
                throw new InvalidOperationException("CTC mode requested but ctc_heads ModelAsset/Worker is not available.");
            }

            _ctcHeadsWorker.SetInput("memory", memory);
            _ctcHeadsWorker.Schedule();

            using var phoneLogits = ReadOutputClone<float>(_ctcHeadsWorker, "phone_logits");
            using var prosodyLogits = ReadOutputClone<float>(_ctcHeadsWorker, "prosody_logits");

            var phoneIds = DecodeCtcGreedy(phoneLogits, _phoneVocab.BlankId ?? _phoneVocab.PadId, effectiveSrcLen);
            var prosodyIds = DecodeCtcGreedy(prosodyLogits, _prosodyVocab.BlankId ?? _prosodyVocab.PadId, effectiveSrcLen);

            var phoneTokens = _phoneVocab.DecodeTokenIds(phoneIds, stripSpecial: true);
            var prosodyTokens = _prosodyVocab.DecodeTokenIds(prosodyIds, stripSpecial: true);

            return new NnG2pInferenceResult
            {
                Input = input,
                Mode = NnG2pInferenceMode.Ctc,
                Graphemes = graphemes.ToArray(),
                SourceIds = srcIds.ToArray(),
                PhoneIds = phoneIds.ToArray(),
                ProsodyIds = prosodyIds.ToArray(),
                Phones = phoneTokens.ToArray(),
                Prosody = prosodyTokens.ToArray(),
            };
        }

        private NnG2pInferenceResult DecodeAutoregressive(
            string input,
            IReadOnlyList<string> graphemes,
            IReadOnlyList<int> srcIds,
            IReadOnlyList<byte> srcPadMaskValues,
            int effectiveSrcLen,
            Tensor<float> memory)
        {
            if (_decoderStepWorker == null)
            {
                throw new InvalidOperationException("Autoregressive mode requested but decoder_step ModelAsset/Worker is not available.");
            }

            using var srcPadMask = BuildSrcPadMask(srcPadMaskValues);

            var phoneIds = new List<int> { _phoneVocab.BosId };
            var prosodyIds = new List<int> { _prosodyVocab.BosId };
            var phoneFinished = false;
            var prosodyFinished = false;

            var decoderContextLength = Mathf.Max(1, fixedDecoderContextLength);
            var effectiveMaxLen = ComputeEffectiveMaxLen(effectiveSrcLen);
            for (var step = 0; step < effectiveMaxLen; step++)
            {
                var phoneInputIds = BuildDecoderContextTokens(phoneIds, decoderContextLength, _phoneVocab.PadId);
                var prosodyInputIds = BuildDecoderContextTokens(prosodyIds, decoderContextLength, _prosodyVocab.PadId);
                using var phoneTokensTensor = new Tensor<int>(new TensorShape(1, decoderContextLength), phoneInputIds);
                using var prosodyTokensTensor = new Tensor<int>(new TensorShape(1, decoderContextLength), prosodyInputIds);

                _decoderStepWorker.SetInput("memory", memory);
                _decoderStepWorker.SetInput("src_pad_mask", srcPadMask);
                _decoderStepWorker.SetInput("phone_tokens", phoneTokensTensor);
                _decoderStepWorker.SetInput("prosody_tokens", prosodyTokensTensor);
                _decoderStepWorker.Schedule();

                using var phoneLogits = ReadOutputClone<float>(_decoderStepWorker, "phone_logits");
                using var prosodyLogits = ReadOutputClone<float>(_decoderStepWorker, "prosody_logits");

                var nextPhone = SelectNextToken(
                    phoneLogits,
                    phoneIds,
                    phoneFinished,
                    _phoneVocab.EosId,
                    repetitionPenalty);
                var nextProsody = SelectNextToken(
                    prosodyLogits,
                    prosodyIds,
                    prosodyFinished,
                    _prosodyVocab.EosId,
                    repetitionPenalty);

                phoneIds.Add(nextPhone);
                prosodyIds.Add(nextProsody);

                phoneFinished |= nextPhone == _phoneVocab.EosId;
                prosodyFinished |= nextProsody == _prosodyVocab.EosId;

                if (phoneFinished && prosodyFinished)
                {
                    break;
                }
            }

            var decodedPhones = _phoneVocab.DecodeTokenIds(phoneIds, stripSpecial: true);
            var decodedProsody = _prosodyVocab.DecodeTokenIds(prosodyIds, stripSpecial: true);

            return new NnG2pInferenceResult
            {
                Input = input,
                Mode = NnG2pInferenceMode.Autoregressive,
                Graphemes = graphemes.ToArray(),
                SourceIds = srcIds.ToArray(),
                PhoneIds = phoneIds.ToArray(),
                ProsodyIds = prosodyIds.ToArray(),
                Phones = decodedPhones.ToArray(),
                Prosody = decodedProsody.ToArray(),
            };
        }

        private static Tensor<int> BuildSrcPadMask(IReadOnlyList<byte> srcPadMaskValues)
        {
            var mask = srcPadMaskValues.Select(v => (int)v).ToArray();
            return new Tensor<int>(new TensorShape(1, mask.Length), mask);
        }

        private static int[] BuildDecoderContextTokens(IReadOnlyList<int> generatedTokens, int contextLength, int padId)
        {
            var context = Enumerable.Repeat(padId, contextLength).ToArray();
            if (generatedTokens.Count == 0)
            {
                return context;
            }

            var copyLen = Mathf.Min(contextLength, generatedTokens.Count);
            var srcStart = generatedTokens.Count - copyLen;
            var dstStart = contextLength - copyLen;
            for (var i = 0; i < copyLen; i++)
            {
                context[dstStart + i] = generatedTokens[srcStart + i];
            }

            return context;
        }

        private int ComputeEffectiveMaxLen(int srcLen)
        {
            if (maxLenRatio <= 0.0f)
            {
                return Mathf.Max(1, maxLen);
            }

            var ratioBound = Mathf.FloorToInt((srcLen * maxLenRatio) + 5.0f);
            return Mathf.Max(1, Mathf.Min(maxLen, ratioBound));
        }

        private static List<int> DecodeCtcGreedy(Tensor<float> logits, int blankId, int validTimeSteps)
        {
            if (logits.shape.rank != 3)
            {
                throw new InvalidOperationException($"CTC logits tensor rank must be 3, but got {logits.shape.rank}.");
            }

            var sequenceLength = logits.shape[1];
            var vocabSize = logits.shape[2];
            var steps = Mathf.Clamp(validTimeSteps, 0, sequenceLength);
            var decoded = new List<int>(steps);

            var prevToken = -1;
            for (var t = 0; t < steps; t++)
            {
                var token = ArgMaxAtTime(logits, t, vocabSize);
                if (token != blankId && token != prevToken)
                {
                    decoded.Add(token);
                }

                prevToken = token;
            }

            return decoded;
        }

        private static int ArgMaxAtTime(Tensor<float> logits, int timeStep, int vocabSize)
        {
            var bestIndex = 0;
            var bestScore = logits[0, timeStep, 0];
            for (var i = 1; i < vocabSize; i++)
            {
                var score = logits[0, timeStep, i];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        private static int SelectNextToken(
            Tensor<float> logits,
            IReadOnlyCollection<int> generatedTokens,
            bool finished,
            int eosId,
            float repetitionPenaltyValue)
        {
            if (finished)
            {
                return eosId;
            }

            if (logits.shape.rank != 2 || logits.shape[0] != 1)
            {
                throw new InvalidOperationException($"Decoder logits shape must be [1, vocab], but got {logits.shape}.");
            }

            var vocabSize = logits.shape[1];
            HashSet<int> penalized = null;
            if (repetitionPenaltyValue > 1.0f)
            {
                penalized = new HashSet<int>(generatedTokens);
            }

            var bestIndex = 0;
            var bestScore = float.NegativeInfinity;
            for (var i = 0; i < vocabSize; i++)
            {
                var score = logits[0, i];
                if (penalized != null && penalized.Contains(i))
                {
                    score /= repetitionPenaltyValue;
                }

                if (score > bestScore)
                {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        private static Tensor<T> ReadOutputClone<T>(Worker worker, string outputName) where T : unmanaged
        {
            var output = worker.PeekOutput(outputName) as Tensor<T>;
            if (output == null)
            {
                throw new InvalidOperationException($"Output '{outputName}' is unavailable or has unexpected type.");
            }

            return output.ReadbackAndClone();
        }

        private List<string> TokenizeInput(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return new List<string>();
            }

            if (string.Equals(language, "en", StringComparison.OrdinalIgnoreCase))
            {
                return text.ToUpperInvariant().Select(c => c.ToString()).ToList();
            }

            return text.Select(c => c.ToString()).ToList();
        }

        private static void DisposeWorker(ref Worker worker)
        {
            if (worker == null)
            {
                return;
            }

            worker.Dispose();
            worker = null;
        }

        private static string ResolveVocabPath(string vocabFileName)
        {
            var streamingRoot = Application.streamingAssetsPath;
            return Path.Combine(streamingRoot, StreamingAssetsRoot, VocabFolder, vocabFileName);
        }

#if UNITY_EDITOR
        private static ModelAsset LoadModelAssetPreferImported(string fileName)
        {
            var imported = AssetDatabase.LoadAssetAtPath<ModelAsset>($"Assets/NNG2P/Models/{fileName}");
            if (imported != null)
            {
                return imported;
            }

            return AssetDatabase.LoadAssetAtPath<ModelAsset>($"Assets/StreamingAssets/nn-g2p/onnx/{fileName}");
        }
#endif
    }
}
