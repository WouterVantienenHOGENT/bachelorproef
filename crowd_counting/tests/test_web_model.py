"""Guard against reloading every prediction or reusing the wrong checkpoint."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Optional web dependencies not installed")
class ModelSessionTests(unittest.TestCase):
    def setUp(self):
        import web

        self.web = web
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "best.pt"
        self.checkpoint.write_bytes(b"checkpoint")

        class Session(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        self.state = Session()
        state_patch = patch.object(web.st, "session_state", self.state)
        state_patch.start()
        self.addCleanup(state_patch.stop)
        loader_patch = patch("infer.load_model", side_effect=lambda *args: object())
        self.loader = loader_patch.start()
        self.addCleanup(loader_patch.stop)

    def test_same_selection_reuses_model_but_changes_reload(self):
        first, reused = self.web.get_model(self.checkpoint, "cpu")
        self.assertFalse(reused)
        second, reused = self.web.get_model(self.checkpoint, "cpu")
        self.assertIs(first, second)
        self.assertTrue(reused)
        self.assertEqual(self.loader.call_count, 1)

        # Training may replace a checkpoint at the same path and size.
        stat = self.checkpoint.stat()
        os.utime(self.checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        changed, reused = self.web.get_model(self.checkpoint, "cpu")
        self.assertFalse(reused)
        self.assertIsNot(first, changed)
        self.web.get_model(self.checkpoint, "cuda")
        other = self.checkpoint.with_name("last.pt")
        other.write_bytes(b"checkpoint")
        self.web.get_model(other, "cuda")
        self.assertEqual(self.loader.call_count, 4)

    def test_architecture_change_invalidates_cached_model(self):
        first, _ = self.web.get_model(self.checkpoint, "cpu", "csrnet")
        second, reused = self.web.get_model(self.checkpoint, "cpu", "p2pnet")
        self.assertFalse(reused)
        self.assertIsNot(first, second)
        self.assertEqual(self.loader.call_count, 2)

    def test_failed_replacement_does_not_leave_old_model_selected(self):
        self.web.get_model(self.checkpoint, "cpu")
        self.loader.side_effect = RuntimeError("Invalid checkpoint")
        with self.assertRaises(RuntimeError):
            self.web.get_model(self.checkpoint, "cuda")
        self.assertNotIn("loaded_model", self.state)
        self.assertNotIn("loaded_model_key", self.state)
