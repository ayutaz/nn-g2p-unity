using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NnG2p.Runtime
{
    public sealed class NnG2pVocab
    {
        private readonly List<string> _tokens;
        private readonly Dictionary<string, int> _tokenToId;

        public IReadOnlyList<string> Tokens => _tokens;

        public int PadId { get; }
        public int UnkId { get; }
        public int BosId { get; }
        public int EosId { get; }
        public int? BlankId { get; }

        private NnG2pVocab(List<string> tokens)
        {
            _tokens = tokens;
            _tokenToId = new Dictionary<string, int>(tokens.Count);
            for (var i = 0; i < tokens.Count; i++)
            {
                _tokenToId[tokens[i]] = i;
            }

            PadId = RequiredTokenId("<pad>");
            UnkId = RequiredTokenId("<unk>");
            BosId = RequiredTokenId("<s>");
            EosId = RequiredTokenId("</s>");
            BlankId = TryGetId("<blank>");
        }

        public static NnG2pVocab LoadFromFile(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Vocab file was not found: {path}");
            }

            var tokens = File.ReadAllLines(path)
                .Select(line => line.Trim())
                .Where(line => !string.IsNullOrEmpty(line))
                .ToList();

            if (tokens.Count == 0)
            {
                throw new InvalidOperationException($"Vocab file is empty: {path}");
            }

            return new NnG2pVocab(tokens);
        }

        public int RequiredTokenId(string token)
        {
            if (_tokenToId.TryGetValue(token, out var id))
            {
                return id;
            }

            throw new InvalidOperationException($"Required token '{token}' was not found in vocab.");
        }

        public int? TryGetId(string token)
        {
            return _tokenToId.TryGetValue(token, out var id) ? id : null;
        }

        public int GetIdOrUnk(string token)
        {
            return _tokenToId.TryGetValue(token, out var id) ? id : UnkId;
        }

        public int[] EncodeTokens(IEnumerable<string> tokens, bool addBosEos = false)
        {
            var ids = new List<int>();
            if (addBosEos)
            {
                ids.Add(BosId);
            }

            foreach (var token in tokens)
            {
                ids.Add(GetIdOrUnk(token));
            }

            if (addBosEos)
            {
                ids.Add(EosId);
            }

            return ids.ToArray();
        }

        public List<string> DecodeTokenIds(IEnumerable<int> ids, bool stripSpecial = true)
        {
            var outTokens = new List<string>();
            foreach (var id in ids)
            {
                var token = (id >= 0 && id < _tokens.Count) ? _tokens[id] : "<unk>";
                if (stripSpecial && IsSpecial(token))
                {
                    continue;
                }

                outTokens.Add(token);
            }

            return outTokens;
        }

        private static bool IsSpecial(string token)
        {
            return token == "<pad>" || token == "<unk>" || token == "<s>" || token == "</s>";
        }
    }
}
