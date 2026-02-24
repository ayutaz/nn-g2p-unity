using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using NnG2p.Runtime;
using UnityEngine;

namespace NnG2p.Tests.EditMode
{
    public class NnG2pSentisRuntimeEditModeTests
    {
        private GameObject _go;
        private NnG2pSentisRuntime _runtime;
        private readonly List<string> _tempPaths = new();

        [SetUp]
        public void SetUp()
        {
            _go = new GameObject("NnG2pSentisRuntimeEditModeTests");
            _runtime = _go.AddComponent<NnG2pSentisRuntime>();
        }

        [TearDown]
        public void TearDown()
        {
            if (_runtime != null)
            {
                _runtime.Dispose();
            }

            if (_go != null)
            {
                UnityEngine.Object.DestroyImmediate(_go);
            }

            foreach (var path in _tempPaths)
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }

            _tempPaths.Clear();
        }

        [Test]
        public void TokenizeInput_WhenJapanese_SplitsCharactersAsIs()
        {
            SetField("language", "ja");
            var tokens = (List<string>)InvokeInstance("TokenizeInput", "東京");
            CollectionAssert.AreEqual(new[] { "東", "京" }, tokens);
        }

        [Test]
        public void TokenizeInput_WhenEnglish_UppercasesAndSplitsCharacters()
        {
            SetField("language", "en");
            var tokens = (List<string>)InvokeInstance("TokenizeInput", "tokyo");
            CollectionAssert.AreEqual(new[] { "T", "O", "K", "Y", "O" }, tokens);
        }

        [Test]
        public void BuildEncoderInput_WhenFixedLengthDisabled_ReturnsOriginalAndZeroMask()
        {
            SetField("fixedEncoderInputLength", 0);
            SetField("_graphemeVocab", CreateVocab());

            var args = new object[] { new[] { 10, 11, 12 }, null, 0 };
            var encoded = (int[])InvokeInstance("BuildEncoderInput", args);
            var mask = (byte[])args[1];
            var effectiveLen = (int)args[2];

            CollectionAssert.AreEqual(new[] { 10, 11, 12 }, encoded);
            CollectionAssert.AreEqual(new byte[] { 0, 0, 0 }, mask);
            Assert.That(effectiveLen, Is.EqualTo(3));
        }

        [Test]
        public void BuildEncoderInput_WhenFixedLengthEnabled_PadsTailAndSetsMask()
        {
            SetField("fixedEncoderInputLength", 5);
            SetField("_graphemeVocab", CreateVocab());

            var args = new object[] { new[] { 7, 8, 9 }, null, 0 };
            var encoded = (int[])InvokeInstance("BuildEncoderInput", args);
            var mask = (byte[])args[1];
            var effectiveLen = (int)args[2];

            CollectionAssert.AreEqual(new[] { 7, 8, 9, 0, 0 }, encoded);
            CollectionAssert.AreEqual(new byte[] { 0, 0, 0, 1, 1 }, mask);
            Assert.That(effectiveLen, Is.EqualTo(3));
        }

        [Test]
        public void BuildEncoderInput_WhenSourceIsLonger_UsesTruncatedPrefix()
        {
            SetField("fixedEncoderInputLength", 4);
            SetField("_graphemeVocab", CreateVocab());

            var args = new object[] { new[] { 1, 2, 3, 4, 5, 6 }, null, 0 };
            var encoded = (int[])InvokeInstance("BuildEncoderInput", args);
            var mask = (byte[])args[1];
            var effectiveLen = (int)args[2];

            CollectionAssert.AreEqual(new[] { 1, 2, 3, 4 }, encoded);
            CollectionAssert.AreEqual(new byte[] { 0, 0, 0, 0 }, mask);
            Assert.That(effectiveLen, Is.EqualTo(4));
        }

        [Test]
        public void FixedEncoderInputLength_DefaultValue_Is512()
        {
            var field = typeof(NnG2pSentisRuntime).GetField("fixedEncoderInputLength", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.That(field, Is.Not.Null);
            var value = (int)field.GetValue(_runtime);
            Assert.That(value, Is.EqualTo(512));
        }

        [Test]
        public void BuildEncoderInput_WhenSourceExceeds512_TruncatesTo512()
        {
            SetField("fixedEncoderInputLength", 512);
            SetField("_graphemeVocab", CreateVocab());

            var src = new int[600];
            for (var i = 0; i < src.Length; i++)
            {
                src[i] = i + 1;
            }

            var args = new object[] { src, null, 0 };
            var encoded = (int[])InvokeInstance("BuildEncoderInput", args);
            var mask = (byte[])args[1];
            var effectiveLen = (int)args[2];

            Assert.That(encoded.Length, Is.EqualTo(512));
            Assert.That(mask.Length, Is.EqualTo(512));
            Assert.That(effectiveLen, Is.EqualTo(512));
            Assert.That(mask[511], Is.EqualTo((byte)0));
            Assert.That(encoded[0], Is.EqualTo(1));
            Assert.That(encoded[511], Is.EqualTo(512));
        }

        [Test]
        public void BuildDecoderContextTokens_WhenShortSequence_LeftPadsWithPadId()
        {
            var method = typeof(NnG2pSentisRuntime).GetMethod("BuildDecoderContextTokens", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.That(method, Is.Not.Null);

            var context = (int[])method.Invoke(null, new object[] { new[] { 2 }, 3, 0 });
            CollectionAssert.AreEqual(new[] { 0, 0, 2 }, context);
        }

        [Test]
        public void BuildDecoderContextTokens_WhenLongSequence_UsesTailWindow()
        {
            var method = typeof(NnG2pSentisRuntime).GetMethod("BuildDecoderContextTokens", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.That(method, Is.Not.Null);

            var context = (int[])method.Invoke(null, new object[] { new[] { 4, 5, 6, 7 }, 3, 0 });
            CollectionAssert.AreEqual(new[] { 5, 6, 7 }, context);
        }

        [Test]
        public void ComputeEffectiveMaxLen_WhenRatioEnabled_RespectsRatioBound()
        {
            SetField("maxLen", 20);
            SetField("maxLenRatio", 2.0f);

            var value = (int)InvokeInstance("ComputeEffectiveMaxLen", 4);
            Assert.That(value, Is.EqualTo(13));
        }

        [Test]
        public void ComputeEffectiveMaxLen_WhenRatioDisabled_UsesMaxLen()
        {
            SetField("maxLen", 9);
            SetField("maxLenRatio", 0.0f);

            var value = (int)InvokeInstance("ComputeEffectiveMaxLen", 4);
            Assert.That(value, Is.EqualTo(9));
        }

        [Test]
        public void ResolveMode_WhenAuto_ReturnsAutoregressive()
        {
            SetField("defaultMode", NnG2pInferenceMode.Auto);
            var value = (NnG2pInferenceMode)InvokeInstance("ResolveMode", (NnG2pInferenceMode?)null);
            Assert.That(value, Is.EqualTo(NnG2pInferenceMode.Autoregressive));
        }

        [Test]
        public void ResolveMode_WhenOverrideProvided_UsesOverride()
        {
            SetField("defaultMode", NnG2pInferenceMode.Auto);
            var value = (NnG2pInferenceMode)InvokeInstance("ResolveMode", NnG2pInferenceMode.Autoregressive);
            Assert.That(value, Is.EqualTo(NnG2pInferenceMode.Autoregressive));
        }

        private NnG2pVocab CreateVocab()
        {
            var path = Path.Combine(Path.GetTempPath(), $"nng2p-runtime-vocab-{Guid.NewGuid():N}.txt");
            File.WriteAllLines(path, new[] { "<pad>", "<unk>", "<s>", "</s>", "a", "b", "c" });
            _tempPaths.Add(path);
            return NnG2pVocab.LoadFromFile(path);
        }

        private object InvokeInstance(string methodName, params object[] args)
        {
            var method = typeof(NnG2pSentisRuntime).GetMethod(methodName, BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.That(method, Is.Not.Null, $"Method '{methodName}' not found.");
            return method.Invoke(_runtime, args);
        }

        private void SetField(string fieldName, object value)
        {
            var field = typeof(NnG2pSentisRuntime).GetField(fieldName, BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.That(field, Is.Not.Null, $"Field '{fieldName}' not found.");
            field.SetValue(_runtime, value);
        }
    }
}
