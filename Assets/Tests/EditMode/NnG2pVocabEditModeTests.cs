using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using NnG2p.Runtime;

namespace NnG2p.Tests.EditMode
{
    public class NnG2pVocabEditModeTests
    {
        private readonly List<string> _tempPaths = new();

        [TearDown]
        public void TearDown()
        {
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
        public void LoadFromFile_WhenMissingFile_ThrowsFileNotFoundException()
        {
            var missingPath = Path.Combine(Path.GetTempPath(), $"missing-vocab-{Guid.NewGuid():N}.txt");
            Assert.Throws<FileNotFoundException>(() => NnG2pVocab.LoadFromFile(missingPath));
        }

        [Test]
        public void LoadFromFile_WhenEmpty_ThrowsInvalidOperationException()
        {
            var path = CreateTempVocabFile(Array.Empty<string>());
            Assert.Throws<InvalidOperationException>(() => NnG2pVocab.LoadFromFile(path));
        }

        [Test]
        public void LoadFromFile_AssignsSpecialTokenIds()
        {
            var path = CreateTempVocabFile(new[] { "<pad>", "<unk>", "<s>", "</s>", "<blank>", "a", "b" });
            var vocab = NnG2pVocab.LoadFromFile(path);

            Assert.That(vocab.PadId, Is.EqualTo(0));
            Assert.That(vocab.UnkId, Is.EqualTo(1));
            Assert.That(vocab.BosId, Is.EqualTo(2));
            Assert.That(vocab.EosId, Is.EqualTo(3));
            Assert.That(vocab.BlankId, Is.EqualTo(4));
        }

        [Test]
        public void EncodeTokens_WhenAddBosEos_IsWrappedByBoundaryTokens()
        {
            var path = CreateTempVocabFile(new[] { "<pad>", "<unk>", "<s>", "</s>", "a", "b" });
            var vocab = NnG2pVocab.LoadFromFile(path);

            var encoded = vocab.EncodeTokens(new[] { "a", "b" }, addBosEos: true);
            CollectionAssert.AreEqual(new[] { vocab.BosId, 4, 5, vocab.EosId }, encoded);
        }

        [Test]
        public void EncodeTokens_WhenUnknownToken_MapsToUnkId()
        {
            var path = CreateTempVocabFile(new[] { "<pad>", "<unk>", "<s>", "</s>", "a" });
            var vocab = NnG2pVocab.LoadFromFile(path);

            var encoded = vocab.EncodeTokens(new[] { "a", "zz" });
            CollectionAssert.AreEqual(new[] { 4, vocab.UnkId }, encoded);
        }

        [Test]
        public void DecodeTokenIds_WithStripSpecialTrue_RemovesSpecialTokens()
        {
            var path = CreateTempVocabFile(new[] { "<pad>", "<unk>", "<s>", "</s>", "a", "b" });
            var vocab = NnG2pVocab.LoadFromFile(path);

            var decoded = vocab.DecodeTokenIds(new[] { 2, 4, 5, 3 }, stripSpecial: true);
            CollectionAssert.AreEqual(new[] { "a", "b" }, decoded);
        }

        [Test]
        public void DecodeTokenIds_WithStripSpecialFalse_KeepsSpecialAndOutOfRangeAsUnk()
        {
            var path = CreateTempVocabFile(new[] { "<pad>", "<unk>", "<s>", "</s>", "a" });
            var vocab = NnG2pVocab.LoadFromFile(path);

            var decoded = vocab.DecodeTokenIds(new[] { 2, 4, 99 }, stripSpecial: false);
            CollectionAssert.AreEqual(new[] { "<s>", "a", "<unk>" }, decoded);
        }

        private string CreateTempVocabFile(IEnumerable<string> lines)
        {
            var path = Path.Combine(Path.GetTempPath(), $"nng2p-vocab-{Guid.NewGuid():N}.txt");
            File.WriteAllLines(path, lines);
            _tempPaths.Add(path);
            return path;
        }
    }
}
