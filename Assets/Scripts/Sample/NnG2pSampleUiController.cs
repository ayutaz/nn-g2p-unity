using System;
using NnG2p.Runtime;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public sealed class NnG2pSampleUiController : MonoBehaviour
{
    [Header("Runtime")]
    [SerializeField] private NnG2pSentisRuntime runtime;

    [Header("UI")]
    [SerializeField] private TMP_InputField inputField;
    [SerializeField] private Button runCtcButton;
    [SerializeField] private Button runArButton;
    [SerializeField] private Button runAutoButton;
    [SerializeField] private Button clearButton;
    [SerializeField] private TMP_Text phonesText;
    [SerializeField] private TMP_Text prosodyText;
    [SerializeField] private TMP_Text errorText;

    [Header("Defaults")]
    [SerializeField] private string defaultInput = "こんにちは、今日はいい天気ですね";

    private void Reset()
    {
        if (runtime == null)
        {
            runtime = FindAnyObjectByType<NnG2pSentisRuntime>();
        }
    }

    private void Awake()
    {
        Input.imeCompositionMode = IMECompositionMode.On;

        if (inputField != null && string.IsNullOrEmpty(inputField.text))
        {
            inputField.text = defaultInput;
        }

        if (runCtcButton != null)
        {
            runCtcButton.onClick.AddListener(RunCtc);
        }

        if (runArButton != null)
        {
            runArButton.onClick.AddListener(RunAr);
        }

        if (runAutoButton != null)
        {
            runAutoButton.onClick.AddListener(RunAuto);
        }

        if (clearButton != null)
        {
            clearButton.onClick.AddListener(ClearOutput);
        }
    }

    private void OnDestroy()
    {
        if (runCtcButton != null)
        {
            runCtcButton.onClick.RemoveListener(RunCtc);
        }

        if (runArButton != null)
        {
            runArButton.onClick.RemoveListener(RunAr);
        }

        if (runAutoButton != null)
        {
            runAutoButton.onClick.RemoveListener(RunAuto);
        }

        if (clearButton != null)
        {
            clearButton.onClick.RemoveListener(ClearOutput);
        }
    }

    [ContextMenu("Run CTC")]
    public void RunCtc()
    {
        RunInference(NnG2pInferenceMode.Ctc);
    }

    [ContextMenu("Run AR")]
    public void RunAr()
    {
        RunInference(NnG2pInferenceMode.Autoregressive);
    }

    [ContextMenu("Run Auto")]
    public void RunAuto()
    {
        RunInference(NnG2pInferenceMode.Auto);
    }

    [ContextMenu("Clear Output")]
    public void ClearOutput()
    {
        if (phonesText != null)
        {
            phonesText.text = string.Empty;
        }

        if (prosodyText != null)
        {
            prosodyText.text = string.Empty;
        }

        if (errorText != null)
        {
            errorText.text = string.Empty;
        }
    }

    private void RunInference(NnG2pInferenceMode mode)
    {
        if (runtime == null)
        {
            SetError("NnG2pSentisRuntime is not assigned.");
            return;
        }

        var text = inputField != null ? inputField.text : string.Empty;
        try
        {
            var result = runtime.Predict(text ?? string.Empty, mode);
            var phones = string.Join(" ", result.Phones ?? Array.Empty<string>());
            var prosody = string.Join(" ", result.Prosody ?? Array.Empty<string>());

            if (phonesText != null)
            {
                phonesText.text = phones;
            }

            if (prosodyText != null)
            {
                prosodyText.text = prosody;
            }

            if (errorText != null)
            {
                errorText.text = string.Empty;
            }

            Debug.Log($"NN-G2P [{result.Mode}] input='{text}' phones='{phones}' prosody='{prosody}'");
        }
        catch (Exception ex)
        {
            SetError(ex.ToString());
        }
    }

    private void SetError(string message)
    {
        if (errorText != null)
        {
            errorText.text = message;
        }

        Debug.LogError(message);
    }
}
