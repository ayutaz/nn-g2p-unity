using System;
using System.Collections;
using NUnit.Framework;
using NnG2p.Runtime;
using UnityEngine;
using UnityEngine.TestTools;

namespace NnG2p.Tests.PlayMode
{
    public class NnG2pSentisRuntimePlayModeTests
    {
        private GameObject _go;
        private NnG2pSentisRuntime _runtime;

        [UnitySetUp]
        public IEnumerator UnitySetUp()
        {
            _go = new GameObject("NnG2pSentisRuntimePlayModeTests");
            _runtime = _go.AddComponent<NnG2pSentisRuntime>();
            yield return null;
        }

        [UnityTearDown]
        public IEnumerator UnityTearDown()
        {
            if (_runtime != null)
            {
                _runtime.Dispose();
            }

            if (_go != null)
            {
                UnityEngine.Object.Destroy(_go);
            }

            yield return null;
        }

        [UnityTest]
        public IEnumerator StartsAsUninitialized()
        {
            Assert.That(_runtime.IsInitialized, Is.False);
            yield return null;
        }

        [UnityTest]
        public IEnumerator LastInitError_IsEmptyBeforeInitialization()
        {
            Assert.That(_runtime.LastInitError, Is.Null.Or.Empty);
            yield return null;
        }

        [UnityTest]
        public IEnumerator TryInitialize_WithoutEncoderModel_ReturnsFalse()
        {
            var ok = _runtime.TryInitialize(out var error);

            Assert.That(ok, Is.False);
            Assert.That(error, Does.Contain("Encoder ModelAsset is not assigned"));
            yield return null;
        }

        [UnityTest]
        public IEnumerator TryInitialize_WithoutEncoderModel_SetsLastInitError()
        {
            _runtime.TryInitialize(out _);

            Assert.That(_runtime.LastInitError, Does.Contain("Encoder ModelAsset is not assigned"));
            Assert.That(_runtime.IsInitialized, Is.False);
            yield return null;
        }

        [UnityTest]
        public IEnumerator Predict_AutoregressiveWithoutEncoder_ThrowsInvalidOperationException()
        {
            Assert.Throws<InvalidOperationException>(() => _runtime.Predict("東京", NnG2pInferenceMode.Autoregressive));
            yield return null;
        }

        [UnityTest]
        public IEnumerator Predict_AutoWithoutEncoder_ThrowsInvalidOperationException()
        {
            Assert.Throws<InvalidOperationException>(() => _runtime.Predict("東京", NnG2pInferenceMode.Auto));
            yield return null;
        }

        [UnityTest]
        public IEnumerator Dispose_CanBeCalledMultipleTimes()
        {
            _runtime.Dispose();
            _runtime.Dispose();

            Assert.That(_runtime.IsInitialized, Is.False);
            yield return null;
        }

        [UnityTest]
        public IEnumerator Dispose_AfterFailedInitialize_KeepsUninitialized()
        {
            _runtime.TryInitialize(out _);
            _runtime.Dispose();

            Assert.That(_runtime.IsInitialized, Is.False);
            yield return null;
        }

        [UnityTest]
        public IEnumerator TryInitialize_CalledTwiceWithoutEncoder_FailsConsistently()
        {
            var ok1 = _runtime.TryInitialize(out var error1);
            var ok2 = _runtime.TryInitialize(out var error2);

            Assert.That(ok1, Is.False);
            Assert.That(ok2, Is.False);
            Assert.That(error1, Is.EqualTo(error2));
            Assert.That(_runtime.IsInitialized, Is.False);
            yield return null;
        }
    }
}
