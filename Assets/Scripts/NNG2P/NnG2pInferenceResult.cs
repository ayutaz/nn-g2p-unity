using System;

namespace NnG2p.Runtime
{
    [Serializable]
    public sealed class NnG2pInferenceResult
    {
        public string Input = string.Empty;
        public NnG2pInferenceMode Mode = NnG2pInferenceMode.Autoregressive;
        public string[] Graphemes = Array.Empty<string>();
        public int[] SourceIds = Array.Empty<int>();
        public int[] PhoneIds = Array.Empty<int>();
        public int[] ProsodyIds = Array.Empty<int>();
        public string[] Phones = Array.Empty<string>();
        public string[] Prosody = Array.Empty<string>();
    }
}
