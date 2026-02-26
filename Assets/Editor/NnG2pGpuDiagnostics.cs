using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using NnG2p.Runtime;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

public static class NnG2pGpuDiagnostics
{
    private const string EncoderAssetPath = "Assets/NNG2P/Models/encoder.onnx";
    private const string DecoderAssetPath = "Assets/NNG2P/Models/decoder_step.onnx";
    private const string InputText = "こんにちは、今日はいい天気ですね";
    private const int FixedSourceLength = 512;
    private const int FixedDecoderContextLength = 512;

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Encoder GPU NaN")]
    private static void ProbeEncoderGpuNan()
    {
        try
        {
            var encoderAsset = LoadModelAssetOrThrow(EncoderAssetPath);
            var encoderModel = ModelLoader.Load(encoderAsset);
            BuildSourceInput(InputText, FixedSourceLength, out var srcIds, out _, out _);

            using var encoderWorker = new Worker(encoderModel, BackendType.GPUCompute);
            ClearCpuFallback(encoderWorker);

            using var srcTensor = new Tensor<int>(new TensorShape(1, srcIds.Length), srcIds);
            encoderWorker.SetInput("src", srcTensor);

            var firstNan = FindFirstNanLayer(encoderWorker, encoderModel, "Encoder");

            if (firstNan.HasValue)
            {
                var hit = firstNan.Value;
                Debug.LogWarning(
                    $"[GPU-NAN][Encoder] firstNaN layer={hit.LayerIndex} op={hit.OpName} outputIndex={hit.OutputIndex} shape={hit.Shape} nan={hit.NaNCount}/{hit.ValueCount}");
            }
            else
            {
                Debug.Log("[GPU-NAN][Encoder] no NaN detected in per-layer probe.");
                var memoryTensor = encoderWorker.PeekOutput("memory") as Tensor<float>;
                var memoryStats = CountNaN(memoryTensor);
                Debug.Log(
                    $"[GPU-NAN][Encoder] final memory nan={memoryStats.NaNCount}/{memoryStats.ValueCount} shape={(memoryTensor != null ? memoryTensor.shape.ToString() : "null")}");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[GPU-NAN][Encoder] probe failed: {ex}");
        }
    }

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Encoder GPU Output")]
    private static void ProbeEncoderGpuOutput()
    {
        ProbeEncoderOutputCore("[GPU-OUT][Encoder]", EncoderAssetPath, clearCpuFallback: true);
    }

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Encoder GPU Output (Native Fallback)")]
    private static void ProbeEncoderGpuOutputNativeFallback()
    {
        ProbeEncoderOutputCore("[GPU-OUT][EncoderNative]", EncoderAssetPath, clearCpuFallback: false);
    }

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Encoder GPUPixel Output (Native Fallback)")]
    private static void ProbeEncoderGpuPixelOutputNativeFallback()
    {
        ProbeEncoderOutputCore(
            "[GPU-OUT][EncoderPixel]",
            EncoderAssetPath,
            clearCpuFallback: false,
            fixedSourceLength: FixedSourceLength,
            backendType: BackendType.GPUPixel);
    }

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Decoder GPU NaN (CPU Memory)")]
    private static void ProbeDecoderGpuNanWithCpuMemory()
    {
        try
        {
            var encoderAsset = LoadModelAssetOrThrow(EncoderAssetPath);
            var decoderAsset = LoadModelAssetOrThrow(DecoderAssetPath);
            var encoderModel = ModelLoader.Load(encoderAsset);
            var decoderModel = ModelLoader.Load(decoderAsset);

            BuildSourceInput(InputText, FixedSourceLength, out var srcIds, out var srcPadMask, out _);
            BuildDecoderSeedTokens(FixedDecoderContextLength, out var phoneTokens, out var prosodyTokens);

            using var srcTensor = new Tensor<int>(new TensorShape(1, srcIds.Length), srcIds);
            using var srcPadMaskTensor = new Tensor<int>(new TensorShape(1, srcPadMask.Length), srcPadMask);
            using var phoneTokensTensor = new Tensor<int>(new TensorShape(1, phoneTokens.Length), phoneTokens);
            using var prosodyTokensTensor = new Tensor<int>(new TensorShape(1, prosodyTokens.Length), prosodyTokens);

            using var encoderCpuWorker = new Worker(encoderModel, BackendType.CPU);
            encoderCpuWorker.SetInput("src", srcTensor);
            encoderCpuWorker.Schedule();
            var memoryCpu = encoderCpuWorker.PeekOutput("memory") as Tensor<float>;
            using var memoryClone = memoryCpu?.ReadbackAndClone() as Tensor<float>;

            if (memoryClone == null)
            {
                throw new InvalidOperationException("CPU encoder output 'memory' is null.");
            }

            using var decoderGpuWorker = new Worker(decoderModel, BackendType.GPUCompute);
            ClearCpuFallback(decoderGpuWorker);
            decoderGpuWorker.SetInput("memory", memoryClone);
            decoderGpuWorker.SetInput("src_pad_mask", srcPadMaskTensor);
            decoderGpuWorker.SetInput("phone_tokens", phoneTokensTensor);
            decoderGpuWorker.SetInput("prosody_tokens", prosodyTokensTensor);

            var firstNan = FindFirstNanLayer(decoderGpuWorker, decoderModel, "Decoder");

            if (firstNan.HasValue)
            {
                var hit = firstNan.Value;
                Debug.LogWarning(
                    $"[GPU-NAN][Decoder] firstNaN layer={hit.LayerIndex} op={hit.OpName} outputIndex={hit.OutputIndex} shape={hit.Shape} nan={hit.NaNCount}/{hit.ValueCount}");
            }
            else
            {
                Debug.Log("[GPU-NAN][Decoder] no NaN detected in per-layer probe.");
                var phoneLogits = decoderGpuWorker.PeekOutput("phone_logits") as Tensor<float>;
                var prosodyLogits = decoderGpuWorker.PeekOutput("prosody_logits") as Tensor<float>;
                var phoneStats = CountNaN(phoneLogits);
                var prosodyStats = CountNaN(prosodyLogits);
                Debug.Log(
                    $"[GPU-NAN][Decoder] phone_logits nan={phoneStats.NaNCount}/{phoneStats.ValueCount} shape={(phoneLogits != null ? phoneLogits.shape.ToString() : "null")}");
                Debug.Log(
                    $"[GPU-NAN][Decoder] prosody_logits nan={prosodyStats.NaNCount}/{prosodyStats.ValueCount} shape={(prosodyLogits != null ? prosodyLogits.shape.ToString() : "null")}");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[GPU-NAN][Decoder] probe failed: {ex}");
        }
    }

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Decoder GPU Output (CPU Memory)")]
    private static void ProbeDecoderGpuOutputWithCpuMemory()
    {
        ProbeDecoderOutputCore("[GPU-OUT][Decoder]", EncoderAssetPath, DecoderAssetPath, clearCpuFallback: true);
    }

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Decoder GPU Output (CPU Memory, Native Fallback)")]
    private static void ProbeDecoderGpuOutputWithCpuMemoryNativeFallback()
    {
        ProbeDecoderOutputCore("[GPU-OUT][DecoderNative]", EncoderAssetPath, DecoderAssetPath, clearCpuFallback: false);
    }

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Decoder GPUPixel Output (CPU Memory, Native Fallback)")]
    private static void ProbeDecoderGpuPixelOutputWithCpuMemoryNativeFallback()
    {
        ProbeDecoderOutputCore(
            "[GPU-OUT][DecoderPixel]",
            EncoderAssetPath,
            DecoderAssetPath,
            clearCpuFallback: false,
            fixedSourceLength: FixedSourceLength,
            fixedDecoderContextLength: FixedDecoderContextLength,
            backendType: BackendType.GPUPixel);
    }

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Encoder CPU Output")]
    private static void ProbeEncoderCpuOutput()
    {
        ProbeEncoderOutputCore(
            "[CPU-OUT][Encoder]",
            EncoderAssetPath,
            clearCpuFallback: false,
            fixedSourceLength: FixedSourceLength,
            backendType: BackendType.CPU);
    }

    [MenuItem("Tools/NN-G2P/Diagnostics/Probe Decoder CPU Output (CPU Memory)")]
    private static void ProbeDecoderCpuOutputWithCpuMemory()
    {
        ProbeDecoderOutputCore(
            "[CPU-OUT][Decoder]",
            EncoderAssetPath,
            DecoderAssetPath,
            clearCpuFallback: false,
            fixedSourceLength: FixedSourceLength,
            fixedDecoderContextLength: FixedDecoderContextLength,
            backendType: BackendType.CPU);
    }

    private static void ProbeEncoderOutputCore(
        string tag,
        string encoderPath,
        bool clearCpuFallback,
        int fixedSourceLength = FixedSourceLength,
        BackendType backendType = BackendType.GPUCompute)
    {
        try
        {
            var encoderAsset = LoadModelAssetOrThrow(encoderPath);
            var encoderModel = ModelLoader.Load(encoderAsset);
            BuildSourceInput(InputText, fixedSourceLength, out var srcIds, out _, out _);

            using var encoderWorker = new Worker(encoderModel, backendType);
            if (clearCpuFallback)
            {
                ClearCpuFallback(encoderWorker);
            }

            using var srcTensor = new Tensor<int>(new TensorShape(1, srcIds.Length), srcIds);
            encoderWorker.SetInput("src", srcTensor);
            encoderWorker.Schedule();

            var memoryTensor = encoderWorker.PeekOutput("memory") as Tensor<float>;
            var memoryStats = CountNaN(memoryTensor);
            var runId = DateTime.UtcNow.Ticks;
            Debug.Log(
                $"{tag} run={runId} memory nan={memoryStats.NaNCount}/{memoryStats.ValueCount} shape={(memoryTensor != null ? memoryTensor.shape.ToString() : "null")} clearCpuFallback={clearCpuFallback} backend={backendType}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"{tag} probe failed: {ex}");
        }
    }

    private static void ProbeDecoderOutputCore(
        string tag,
        string encoderPath,
        string decoderPath,
        bool clearCpuFallback,
        int fixedSourceLength = FixedSourceLength,
        int fixedDecoderContextLength = FixedDecoderContextLength,
        BackendType backendType = BackendType.GPUCompute)
    {
        try
        {
            var encoderAsset = LoadModelAssetOrThrow(encoderPath);
            var decoderAsset = LoadModelAssetOrThrow(decoderPath);
            var encoderModel = ModelLoader.Load(encoderAsset);
            var decoderModel = ModelLoader.Load(decoderAsset);

            BuildSourceInput(InputText, fixedSourceLength, out var srcIds, out var srcPadMask, out _);
            BuildDecoderSeedTokens(fixedDecoderContextLength, out var phoneTokens, out var prosodyTokens);

            using var srcTensor = new Tensor<int>(new TensorShape(1, srcIds.Length), srcIds);
            using var srcPadMaskTensor = new Tensor<int>(new TensorShape(1, srcPadMask.Length), srcPadMask);
            using var phoneTokensTensor = new Tensor<int>(new TensorShape(1, phoneTokens.Length), phoneTokens);
            using var prosodyTokensTensor = new Tensor<int>(new TensorShape(1, prosodyTokens.Length), prosodyTokens);

            using var encoderCpuWorker = new Worker(encoderModel, BackendType.CPU);
            encoderCpuWorker.SetInput("src", srcTensor);
            encoderCpuWorker.Schedule();
            var memoryCpu = encoderCpuWorker.PeekOutput("memory") as Tensor<float>;
            using var memoryClone = memoryCpu?.ReadbackAndClone() as Tensor<float>;
            if (memoryClone == null)
            {
                throw new InvalidOperationException("CPU encoder output 'memory' is null.");
            }

            using var decoderGpuWorker = new Worker(decoderModel, backendType);
            if (clearCpuFallback)
            {
                ClearCpuFallback(decoderGpuWorker);
            }

            decoderGpuWorker.SetInput("memory", memoryClone);
            decoderGpuWorker.SetInput("src_pad_mask", srcPadMaskTensor);
            decoderGpuWorker.SetInput("phone_tokens", phoneTokensTensor);
            decoderGpuWorker.SetInput("prosody_tokens", prosodyTokensTensor);
            decoderGpuWorker.Schedule();

            var phoneLogits = decoderGpuWorker.PeekOutput("phone_logits") as Tensor<float>;
            var prosodyLogits = decoderGpuWorker.PeekOutput("prosody_logits") as Tensor<float>;
            var phoneStats = CountNaN(phoneLogits);
            var prosodyStats = CountNaN(prosodyLogits);
            var runId = DateTime.UtcNow.Ticks;

            Debug.Log(
                $"{tag} run={runId} phone_logits nan={phoneStats.NaNCount}/{phoneStats.ValueCount} shape={(phoneLogits != null ? phoneLogits.shape.ToString() : "null")} clearCpuFallback={clearCpuFallback} backend={backendType}");
            Debug.Log(
                $"{tag} run={runId} prosody_logits nan={prosodyStats.NaNCount}/{prosodyStats.ValueCount} shape={(prosodyLogits != null ? prosodyLogits.shape.ToString() : "null")} clearCpuFallback={clearCpuFallback} backend={backendType}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"{tag} probe failed: {ex}");
        }
    }

    private static ModelAsset LoadModelAssetOrThrow(string path)
    {
        var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(path);
        if (asset == null)
        {
            throw new FileNotFoundException($"ModelAsset not found at {path}");
        }

        return asset;
    }

    private static void BuildSourceInput(string text, int fixedLength, out int[] srcIds, out int[] srcPadMask, out int effectiveLength)
    {
        var graphemeVocabPath = Path.Combine(Application.streamingAssetsPath, "nn-g2p", "vocab", "ja_grapheme_m4.txt");
        var graphemeVocab = NnG2pVocab.LoadFromFile(graphemeVocabPath);
        var graphemes = text.Select(c => c.ToString());
        var source = graphemeVocab.EncodeTokens(graphemes, addBosEos: false);

        effectiveLength = Mathf.Min(source.Length, fixedLength);
        srcIds = new int[fixedLength];
        srcPadMask = new int[fixedLength];

        for (var i = 0; i < fixedLength; i++)
        {
            srcIds[i] = graphemeVocab.PadId;
            srcPadMask[i] = 1;
        }

        for (var i = 0; i < effectiveLength; i++)
        {
            srcIds[i] = source[i];
            srcPadMask[i] = 0;
        }
    }

    private static void BuildDecoderSeedTokens(int contextLength, out int[] phoneTokens, out int[] prosodyTokens)
    {
        var phoneVocabPath = Path.Combine(Application.streamingAssetsPath, "nn-g2p", "vocab", "ja_phones_m8.txt");
        var prosodyVocabPath = Path.Combine(Application.streamingAssetsPath, "nn-g2p", "vocab", "ja_prosody_or_stress_m8.txt");
        var phoneVocab = NnG2pVocab.LoadFromFile(phoneVocabPath);
        var prosodyVocab = NnG2pVocab.LoadFromFile(prosodyVocabPath);

        phoneTokens = new int[contextLength];
        prosodyTokens = new int[contextLength];
        for (var i = 0; i < contextLength; i++)
        {
            phoneTokens[i] = phoneVocab.PadId;
            prosodyTokens[i] = prosodyVocab.PadId;
        }

        phoneTokens[0] = phoneVocab.BosId;
        prosodyTokens[0] = prosodyVocab.BosId;
    }

    private static LayerNanHit? FindFirstNanLayer(Worker worker, Model model, string tag)
    {
        var storageField = typeof(Worker).GetField("m_Storage", BindingFlags.Instance | BindingFlags.NonPublic);
        var storage = storageField?.GetValue(worker);
        if (storage == null)
        {
            throw new InvalidOperationException("Could not access Worker.m_Storage via reflection.");
        }

        var peekTensorMethod = storage.GetType().GetMethod("PeekTensor", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (peekTensorMethod == null)
        {
            throw new InvalidOperationException("Could not access ModelStorage.PeekTensor via reflection.");
        }

        var iterator = worker.ScheduleIterable();
        var layerIndex = 0;
        var preceding = new System.Collections.Generic.Queue<string>();
        while (iterator.MoveNext() && layerIndex < model.layers.Count)
        {
            var layer = model.layers[layerIndex];
            foreach (var outputIndex in layer.outputs)
            {
                var tensor = peekTensorMethod.Invoke(storage, new object[] { outputIndex }) as Tensor;
                var stats = CountNaN(tensor as Tensor<float>);
                var summary =
                    $"[GPU-NAN][{tag}] layer={layerIndex} op={layer.opName} type={layer.GetType().Name} in=[{string.Join(",", layer.inputs)}] out=[{string.Join(",", layer.outputs)}] shape={(tensor != null ? tensor.shape.ToString() : "null")} nan={stats.NaNCount}/{stats.ValueCount}";
                if (layerIndex < 4 || stats.NaNCount > 0)
                {
                    Debug.Log(summary);
                }

                if (stats.NaNCount > 0)
                {
                    foreach (var prev in preceding)
                    {
                        Debug.Log($"[GPU-NAN][{tag}] preNaN {prev}");
                    }

                    return new LayerNanHit
                    {
                        LayerIndex = layerIndex,
                        OpName = layer.opName,
                        OutputIndex = outputIndex,
                        Shape = tensor != null ? tensor.shape.ToString() : "null",
                        NaNCount = stats.NaNCount,
                        ValueCount = stats.ValueCount,
                    };
                }

                preceding.Enqueue(summary);
                while (preceding.Count > 4)
                {
                    preceding.Dequeue();
                }
            }

            layerIndex++;
        }

        return null;
    }

    private static (int NaNCount, int ValueCount) CountNaN(Tensor<float> tensor)
    {
        if (tensor == null)
        {
            return (0, 0);
        }

        using var clone = tensor.ReadbackAndClone() as Tensor<float>;
        if (clone == null)
        {
            return (0, 0);
        }

        clone.CompleteAllPendingOperations();
        var nanCount = 0;
        var valueCount = clone.count;
        for (var i = 0; i < valueCount; i++)
        {
            if (float.IsNaN(clone[i]))
            {
                nanCount++;
            }
        }

        return (nanCount, valueCount);
    }

    private static void ClearCpuFallback(Worker worker)
    {
        ClearFieldCollection(worker, "m_LayerCPUFallback");
        ClearFieldCollection(worker, "m_LayerCPUFallbackShouldFlushGPU");
    }

    private static void ClearFieldCollection(Worker worker, string fieldName)
    {
        var field = typeof(Worker).GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        var value = field?.GetValue(worker);
        var clearMethod = value?.GetType().GetMethod("Clear", BindingFlags.Instance | BindingFlags.Public);
        clearMethod?.Invoke(value, null);
    }

    private struct LayerNanHit
    {
        public int LayerIndex;
        public string OpName;
        public int OutputIndex;
        public string Shape;
        public int NaNCount;
        public int ValueCount;
    }
}
