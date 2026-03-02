import unittest

import torch

from src.data.saa_segmentation import segment_saa_recording
from src.models.prosody import SAASample


class TestSAASegmentation(unittest.TestCase):
    def test_segments_follow_speech_regions(self) -> None:
        sr = 16000
        silence = torch.zeros(int(0.5 * sr), dtype=torch.float32)
        speech_a = torch.ones(int(2.4 * sr), dtype=torch.float32) * 0.1
        speech_b = torch.ones(int(2.2 * sr), dtype=torch.float32) * 0.1
        waveform = torch.cat([silence, speech_a, silence, speech_b, silence], dim=0)

        sample = SAASample(speaker_id="1", filename="demo", native_language="polish")
        segments = segment_saa_recording(
            sample=sample,
            waveform=waveform,
            sampling_rate=sr,
            min_duration_sec=1.5,
            max_duration_sec=4.0,
            merge_gap_sec=0.1,
            top_db=30,
        )

        self.assertEqual(len(segments), 2)
        self.assertLess(segments[0].start_sample, int(0.7 * sr))
        self.assertGreater(segments[0].duration_seconds, 2.0)
        self.assertGreater(segments[1].duration_seconds, 2.0)

    def test_long_voiced_region_is_split(self) -> None:
        sr = 16000
        waveform = torch.ones(int(12.0 * sr), dtype=torch.float32) * 0.1
        sample = SAASample(speaker_id="2", filename="long", native_language="spanish")

        segments = segment_saa_recording(
            sample=sample,
            waveform=waveform,
            sampling_rate=sr,
            min_duration_sec=2.0,
            max_duration_sec=5.0,
            merge_gap_sec=0.2,
            top_db=30,
        )

        self.assertEqual(len(segments), 3)
        self.assertTrue(all(seg.duration_seconds >= 2.0 for seg in segments))
        self.assertTrue(all(seg.duration_seconds <= 5.1 for seg in segments))


if __name__ == "__main__":
    unittest.main()
