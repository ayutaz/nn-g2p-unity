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
        [SerializeField] private ModelAsset decoderStepModelAsset;

        [Header("Runtime")]
        [SerializeField] private BackendType backendType = BackendType.CPU;
        [SerializeField] private NnG2pInferenceMode defaultMode = NnG2pInferenceMode.Autoregressive;
        [SerializeField] private string language = "ja";
        [SerializeField] private int maxLen = 512;
        [SerializeField] private float maxLenRatio = 3.0f;
        [SerializeField] private float repetitionPenalty = 1.2f;
        [SerializeField] private int fixedEncoderInputLength = 512;
        [SerializeField] private int fixedDecoderContextLength = 512;

        [Header("Vocab Files (StreamingAssets/nn-g2p/vocab)")]
        [SerializeField] private string graphemeVocabFile = "ja_grapheme_m4.txt";
        [SerializeField] private string phoneVocabFile = "ja_phones_m8.txt";
        [SerializeField] private string prosodyVocabFile = "ja_prosody_or_stress_m8.txt";

        private Worker _encoderWorker;
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
                _encoderWorker = CreateWorkerWithFallback(encoderModel, "encoder");

                if (decoderStepModelAsset != null)
                {
                    var decoderModel = ModelLoader.Load(decoderStepModelAsset);
                    _decoderStepWorker = CreateWorkerWithFallback(decoderModel, "decoder_step");
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
            var needsInit =
                !_isInitialized ||
                _graphemeVocab == null ||
                _phoneVocab == null ||
                _prosodyVocab == null ||
                _encoderWorker == null ||
                _decoderStepWorker == null;

            if (needsInit)
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
            return DecodeAutoregressive(text, graphemes, srcIds, srcPadMaskValues, effectiveSrcLen, memory, mode);
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
            decoderStepModelAsset = LoadModelAssetPreferImported("decoder_step.onnx");
            EditorUtility.SetDirty(this);
            Debug.Log("Auto-assign completed for ONNX ModelAssets under Assets/StreamingAssets/nn-g2p/onnx.");
        }
#endif

        private NnG2pInferenceMode ResolveMode(NnG2pInferenceMode? modeOverride)
        {
            var requested = modeOverride ?? defaultMode;
            if (requested == NnG2pInferenceMode.Autoregressive)
            {
                return NnG2pInferenceMode.Autoregressive;
            }

            if (requested == NnG2pInferenceMode.Auto)
            {
                return NnG2pInferenceMode.Autoregressive;
            }

            throw new NotSupportedException("Requested inference mode is not supported. Use Autoregressive mode.");
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

            var padded = new int[fixedEncoderInputLength];
            srcPadMaskValues = new byte[fixedEncoderInputLength];
            for (var i = 0; i < fixedEncoderInputLength; i++)
            {
                padded[i] = _graphemeVocab.PadId;
                srcPadMaskValues[i] = 1;
            }

            for (var i = 0; i < effectiveSourceLength; i++)
            {
                padded[i] = srcIds[i];
                srcPadMaskValues[i] = 0;
            }

            return padded;
        }

        private Tensor<float> RunEncoder(IReadOnlyList<int> srcIds)
        {
            var srcData = srcIds as int[] ?? srcIds.ToArray();
            var srcShape = new TensorShape(1, srcData.Length);
            using var srcTensor = new Tensor<int>(srcShape, srcData);

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

        private NnG2pInferenceResult DecodeAutoregressive(
            string input,
            IReadOnlyList<string> graphemes,
            IReadOnlyList<int> srcIds,
            IReadOnlyList<byte> srcPadMaskValues,
            int effectiveSrcLen,
            Tensor<float> memory,
            NnG2pInferenceMode resolvedMode)
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
            var phoneInputIds = new int[decoderContextLength];
            var prosodyInputIds = new int[decoderContextLength];
            for (var i = 0; i < decoderContextLength; i++)
            {
                phoneInputIds[i] = _phoneVocab.PadId;
                prosodyInputIds[i] = _prosodyVocab.PadId;
            }

            phoneInputIds[0] = _phoneVocab.BosId;
            prosodyInputIds[0] = _prosodyVocab.BosId;
            var phoneSeen = repetitionPenalty > 1.0f ? new bool[_phoneVocab.Tokens.Count] : null;
            var prosodySeen = repetitionPenalty > 1.0f ? new bool[_prosodyVocab.Tokens.Count] : null;
            if (phoneSeen != null)
            {
                phoneSeen[_phoneVocab.BosId] = true;
            }

            if (prosodySeen != null)
            {
                prosodySeen[_prosodyVocab.BosId] = true;
            }

            var decodePosition = 0;
            var effectiveMaxLen = ComputeEffectiveMaxLen(effectiveSrcLen);
            for (var step = 0; step < effectiveMaxLen; step++)
            {
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
                    decodePosition,
                    phoneSeen,
                    phoneFinished,
                    _phoneVocab.EosId,
                    repetitionPenalty);
                var nextProsody = SelectNextToken(
                    prosodyLogits,
                    decodePosition,
                    prosodySeen,
                    prosodyFinished,
                    _prosodyVocab.EosId,
                    repetitionPenalty);

                phoneIds.Add(nextPhone);
                prosodyIds.Add(nextProsody);
                if (phoneSeen != null && nextPhone >= 0 && nextPhone < phoneSeen.Length)
                {
                    phoneSeen[nextPhone] = true;
                }

                if (prosodySeen != null && nextProsody >= 0 && nextProsody < prosodySeen.Length)
                {
                    prosodySeen[nextProsody] = true;
                }

                phoneFinished |= nextPhone == _phoneVocab.EosId;
                prosodyFinished |= nextProsody == _prosodyVocab.EosId;

                decodePosition++;
                if (decodePosition < decoderContextLength)
                {
                    phoneInputIds[decodePosition] = nextPhone;
                    prosodyInputIds[decodePosition] = nextProsody;
                }

                if (phoneFinished && prosodyFinished)
                {
                    break;
                }

                if (decodePosition >= decoderContextLength)
                {
                    Debug.LogWarning($"Decoder context length ({decoderContextLength}) exhausted before EOS.");
                    break;
                }
            }

            var decodedPhones = _phoneVocab.DecodeTokenIds(phoneIds, stripSpecial: true);
            var decodedProsody = _prosodyVocab.DecodeTokenIds(prosodyIds, stripSpecial: true);

            return new NnG2pInferenceResult
            {
                Input = input,
                Mode = resolvedMode,
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
            var mask = new int[srcPadMaskValues.Count];
            for (var i = 0; i < srcPadMaskValues.Count; i++)
            {
                mask[i] = srcPadMaskValues[i];
            }

            return new Tensor<int>(new TensorShape(1, mask.Length), mask);
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

        private static int SelectNextToken(
            Tensor<float> logits,
            int decodePosition,
            IReadOnlyList<bool> seenTokens,
            bool finished,
            int eosId,
            float repetitionPenaltyValue)
        {
            if (finished)
            {
                return eosId;
            }

            int vocabSize;
            int timeIndex = 0;
            if (logits.shape.rank == 2 && logits.shape[0] == 1)
            {
                vocabSize = logits.shape[1];
            }
            else if (logits.shape.rank == 3 && logits.shape[0] == 1)
            {
                var seqLen = logits.shape[1];
                timeIndex = Mathf.Clamp(decodePosition, 0, seqLen - 1);
                vocabSize = logits.shape[2];
            }
            else
            {
                throw new InvalidOperationException($"Decoder logits shape must be [1, vocab] or [1, seq, vocab], but got {logits.shape}.");
            }

            var bestIndex = 0;
            var bestScore = float.NegativeInfinity;
            var applyPenalty = repetitionPenaltyValue > 1.0f && seenTokens != null;
            for (var i = 0; i < vocabSize; i++)
            {
                var score = logits.shape.rank == 2 ? logits[0, i] : logits[0, timeIndex, i];
                if (applyPenalty && i < seenTokens.Count && seenTokens[i])
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

        private Worker CreateWorkerWithFallback(Model model, string modelName)
        {
            try
            {
                return new Worker(model, backendType);
            }
            catch (Exception primaryError) when (backendType != BackendType.CPU)
            {
                Debug.LogWarning(
                    $"Failed to create {modelName} worker with backend={backendType}. Falling back to CPU. Error: {primaryError.Message}");
                backendType = BackendType.CPU;
                return new Worker(model, backendType);
            }
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
