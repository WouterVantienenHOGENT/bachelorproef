"""Live input boundaries, snapshot decoding, heatmaps and Streamlit lifecycle."""

import csv
import importlib.util
import io
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from infer import LoadedModel
from livestream import analyze_frame, capture_frame, heatmap_overlay, history_csv, resolve_stream, youtube_url


def jpeg():
    buffer = io.BytesIO()
    Image.new("RGB", (256, 128), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


class LiveInputTests(unittest.TestCase):
    def test_only_youtube_video_links_are_accepted(self):
        for value in ["https://youtu.be/abcdefghijk?t=42", "https://www.youtube.com/watch?v=abcdefghijk&list=abc",
                      "https://youtube.com/live/abcdefghijk", "https://m.youtube.com/embed/abcdefghijk"]:
            self.assertEqual(youtube_url(value), "https://www.youtube.com/watch?v=abcdefghijk")
        for value in ["https://youtube.com.evil.test/watch?v=abcdefghijk", "file:///tmp/video",
                      "https://example.com/video", "https://youtube.com/playlist?list=abc",
                      "https://youtube.com/watch?v=abc", "https://youtu.be/abcdefghijk/extra"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                youtube_url(value)

    @unittest.skipUnless(importlib.util.find_spec("yt_dlp"), "Optional YouTube dependency not installed")
    def test_resolver_requires_currently_live_video(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch("yt_dlp.YoutubeDL") as factory:
            extract = factory.return_value.__enter__.return_value.extract_info
            extract.return_value = {"live_status": "is_live", "url": "https://video.test/live", "title": "Plein"}
            self.assertEqual(resolve_stream("https://youtu.be/abcdefghijk")["title"], "Plein")
            for status in ["not_live", "is_upcoming", "was_live", "post_live"]:
                extract.return_value["live_status"] = status
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, "niet live"):
                    resolve_stream("https://youtu.be/abcdefghijk")

    def test_snapshot_errors_are_bounded_and_do_not_expose_stream_url(self):
        for error in [subprocess.TimeoutExpired(["ffmpeg"], 25),
                      subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"secret stream URL")]:
            with patch("subprocess.run", side_effect=error), self.assertRaisesRegex(RuntimeError, "Geen beeld"):
                capture_frame("https://video.test/secret")
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, b"")):
            with self.assertRaisesRegex(RuntimeError, "geen beeld"):
                capture_frame("https://video.test/secret")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_real_ffmpeg_decodes_a_snapshot(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jpg"
            path.write_bytes(jpeg())
            output = capture_frame(str(path))
        with Image.open(io.BytesIO(output)) as image:
            self.assertEqual(image.size, (1024, 512))


class DensityModel(torch.nn.Module):
    def forward(self, image):
        return torch.ones((1, 1, image.shape[2] // 8, image.shape[3] // 8), device=image.device) / 64


class PointsModel(torch.nn.Module):
    def forward(self, image):
        return {"pred_logits": torch.tensor([[[0., 5.], [0., 5.], [0., -5.]]]),
                "pred_points": torch.tensor([[[64., 64.], [-10., 20.], [120., 60.]]])}


class LiveInferenceTests(unittest.TestCase):
    def test_both_models_produce_count_and_heatmap(self):
        for architecture, model, count in [("csrnet", DensityModel(), 8), ("p2pnet", PointsModel(), 2)]:
            loaded = LoadedModel(model, torch.device("cpu"), Path("/tmp/live-test.pt"), None, architecture)
            result = analyze_frame(jpeg(), loaded)
            self.assertEqual(result["count"], count)
            with Image.open(io.BytesIO(result["heatmap"])) as heatmap:
                self.assertEqual(heatmap.size, (256, 128))
            self.assertNotEqual(result["frame"], result["heatmap"])

    def test_empty_heatmap_and_csv_roundtrip(self):
        heatmap = heatmap_overlay(Image.new("RGB", (64, 32), "white"), np.zeros((4, 8)))
        with Image.open(io.BytesIO(heatmap)) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        rows = [{"timestamp_utc": "2026-10-31T10:00:00+00:00", "estimated_count": 12.5, "processing_ms": 123}]
        exported = list(csv.DictReader(io.StringIO(history_csv(rows))))
        self.assertEqual(float(exported[0]["estimated_count"]), 12.5)


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Optional web dependencies not installed")
class LiveUITests(unittest.TestCase):
    def test_start_interval_rerun_stop_failure_and_restart(self):
        from streamlit.testing.v1 import AppTest
        from model_catalog import ModelEntry

        frame = jpeg()
        measurement = {"count": 7, "forward_ms": 12, "frame": frame, "heatmap": frame, "prediction_type": "points"}
        with patch("livestream.resolve_stream", return_value={"url": "https://video.test", "title": "Testplein"}), \
             patch("livestream.capture_frame", return_value=frame), \
             patch("livestream.analyze_frame", return_value=measurement), \
             patch("web.get_model", return_value=(object(), True)), \
             patch("web.discover_models", return_value=([ModelEntry(Path("/tmp/test.pt"), "Testmodel", "csrnet")], [])):
            app = AppTest.from_string("import web; web.main()", default_timeout=20).run()
            self.assertFalse(app.exception)
            next(w for w in app.text_input if w.label == "YouTube-livestream").set_value("https://youtu.be/abcdefghijk")
            next(w for w in app.button if w.label == "Livestream starten").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["live"]["running"])
            self.assertEqual(len(app.session_state["live"]["rows"]), 1)
            next(w for w in app.number_input if w.label == "Meetinterval in seconden").set_value(10).run()
            self.assertEqual(len(app.session_state["live"]["rows"]), 1)
            next(w for w in app.button if w.label == "Stoppen").click().run()
            self.assertFalse(app.session_state["live"]["running"])
            self.assertEqual(len(app.session_state["live"]["rows"]), 1)
            next(w for w in app.button if w.label == "Livestream starten").click().run()
            self.assertTrue(app.session_state["live"]["running"])
            self.assertEqual(len(app.session_state["live"]["rows"]), 1)
            app.session_state["live"]["next_sample"] = 0
            with patch("livestream.capture_frame", side_effect=RuntimeError("Verbinding verbroken")):
                app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state["live"]["running"])
            self.assertEqual(len(app.session_state["live"]["rows"]), 1)
            self.assertIn("Verbinding verbroken", [error.value for error in app.error])

    def test_missing_models_and_invalid_links_do_not_start_measurements(self):
        from streamlit.testing.v1 import AppTest
        from model_catalog import ModelEntry

        with patch("web.discover_models", return_value=([], [])):
            app = AppTest.from_string("import web; web.main()", default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertTrue(next(w for w in app.button if w.label == "Livestream starten").disabled)
        with patch("web.discover_models", return_value=([ModelEntry(Path("/tmp/test.pt"), "Testmodel", "csrnet")], [])), \
             patch("livestream.resolve_stream") as resolver:
            app = AppTest.from_string("import web; web.main()", default_timeout=20).run()
            next(w for w in app.text_input if w.label == "YouTube-livestream").set_value("https://example.com/live")
            next(w for w in app.button if w.label == "Livestream starten").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            resolver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
