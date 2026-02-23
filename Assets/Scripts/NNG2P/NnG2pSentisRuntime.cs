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

            using var memory = RunEncoder(srcIds);
            var mode = ResolveMode(modeOverride);

            return mode switch
            {
                NnG2pInferenceMode.Ctc => DecodeCtc(text, graphemes, srcIds, memory),
                NnG2pInferenceMode.Autoregressive => DecodeAutoregressive(text, graphemes, srcIds, memory),
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
            encoderModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/StreamingAssets/nn-g2p/onnx/encoder.onnx");
            ctcHeadsModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/StreamingAssets/nn-g2p/onnx/ctc_heads.onnx");
            decoderStepModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/StreamingAssets/nn-g2p/onnx/decoder_step.onnx");
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

            var phoneIds = DecodeCtcGreedy(phoneLogits, _phoneVocab.BlankId ?? _phoneVocab.PadId);
            var prosodyIds = DecodeCtcGreedy(prosodyLogits, _prosodyVocab.BlankId ?? _prosodyVocab.PadId);

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
            Tensor<float> memory)
        {
            if (_decoderStepWorker == null)
            {
                throw new InvalidOperationException("Autoregressive mode requested but decoder_step ModelAsset/Worker is not available.");
            }

            using var srcPadMask = BuildSrcPadMask(srcIds.Count);

            var phoneIds = new List<int> { _phoneVocab.BosId };
            var prosodyIds = new List<int> { _prosodyVocab.BosId };
            var phoneFinished = false;
            var prosodyFinished = false;

            var effectiveMaxLen = ComputeEffectiveMaxLen(srcIds.Count);
            for (var step = 0; step < effectiveMaxLen; step++)
            {
                using var phoneTokensTensor = new Tensor<int>(new TensorShape(1, phoneIds.Count), phoneIds.ToArray());
                using var prosodyTokensTensor = new Tensor<int>(new TensorShape(1, prosodyIds.Count), prosodyIds.ToArray());

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

        private static Tensor<byte> BuildSrcPadMask(int srcLen)
        {
            var mask = new byte[srcLen];
            return new Tensor<byte>(new TensorShape(1, srcLen), mask);
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

        private static List<int> DecodeCtcGreedy(Tensor<float> logits, int blankId)
        {
            if (logits.shape.rank != 3)
            {
                throw new InvalidOperationException($"CTC logits tensor rank must be 3, but got {logits.shape.rank}.");
            }

            var sequenceLength = logits.shape[1];
            var vocabSize = logits.shape[2];
            var decoded = new List<int>(sequenceLength);

            var prevToken = -1;
            for (var t = 0; t < sequenceLength; t++)
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
    }
}
