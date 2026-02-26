using System;
using UnityEngine;

namespace NnG2p.Runtime
{
    public sealed class NnG2pSampleSceneController : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private NnG2pSentisRuntime runtime;

        [Header("Input")]
        [SerializeField] private string inputText = "こんにちは、今日はいい天気ですね";
        [SerializeField] private bool runOnStart;
        [SerializeField] private bool showOnGui = true;

        [Header("Last Output")]
        [SerializeField, TextArea(2, 4)] private string phoneOutput = string.Empty;
        [SerializeField, TextArea(2, 4)] private string prosodyOutput = string.Empty;
        [SerializeField, TextArea(2, 4)] private string lastError = string.Empty;

        public string LastPhones => phoneOutput;
        public string LastProsody => prosodyOutput;
        public string LastError => lastError;

        private void Reset()
        {
            runtime = GetComponent<NnG2pSentisRuntime>();
        }

        private void Awake()
        {
            if (runtime == null)
            {
                runtime = GetComponent<NnG2pSentisRuntime>();
            }
        }

        private void Start()
        {
            if (runOnStart)
            {
                RunInference();
            }
        }

        [ContextMenu("Run Inference")]
        public void RunInference()
        {
            if (runtime == null)
            {
                lastError = "NnG2pSentisRuntime is not assigned.";
                Debug.LogError(lastError);
                return;
            }

            try
            {
                var result = runtime.Predict(inputText ?? string.Empty, NnG2pInferenceMode.Autoregressive);
                phoneOutput = string.Join(" ", result.Phones ?? Array.Empty<string>());
                prosodyOutput = string.Join(" ", result.Prosody ?? Array.Empty<string>());
                lastError = string.Empty;
                Debug.Log(
                    $"NN-G2P [{result.Mode}] backend={runtime.ActiveBackendType} input='{inputText}' phones='{phoneOutput}' prosody='{prosodyOutput}'");
            }
            catch (Exception ex)
            {
                phoneOutput = string.Empty;
                prosodyOutput = string.Empty;
                lastError = ex.ToString();
                Debug.LogError(ex);
            }
        }

        [ContextMenu("Clear Output")]
        public void ClearOutput()
        {
            phoneOutput = string.Empty;
            prosodyOutput = string.Empty;
            lastError = string.Empty;
        }

        private void OnGUI()
        {
            if (!showOnGui)
            {
                return;
            }

            GUILayout.BeginArea(new Rect(20, 20, 760, 420), GUI.skin.window);
            GUILayout.Label("NN-G2P Sample");
            GUILayout.Label("Input");
            inputText = GUILayout.TextField(inputText ?? string.Empty);
            GUILayout.Label("Mode: AR only");

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Run", GUILayout.Height(30)))
            {
                RunInference();
            }

            if (GUILayout.Button("Clear", GUILayout.Height(30)))
            {
                ClearOutput();
            }

            GUILayout.EndHorizontal();

            GUILayout.Label("Phones");
            GUILayout.TextArea(phoneOutput ?? string.Empty, GUILayout.Height(70));
            GUILayout.Label("Prosody");
            GUILayout.TextArea(prosodyOutput ?? string.Empty, GUILayout.Height(70));

            if (!string.IsNullOrEmpty(lastError))
            {
                GUILayout.Label("Error");
                GUILayout.TextArea(lastError, GUILayout.Height(110));
            }

            GUILayout.EndArea();
        }
    }
}
