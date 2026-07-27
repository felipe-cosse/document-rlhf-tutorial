from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_dpo import model_load_kwargs


class ModelLoadKwargsTests(unittest.TestCase):
    def test_cpu_mode_forces_cpu_device_map(self) -> None:
        kwargs = model_load_kwargs(force_cpu=True, quantization=None)

        self.assertEqual(kwargs["device_map"], "cpu")
        self.assertNotIn("quantization_config", kwargs)

    def test_accelerator_mode_preserves_quantization(self) -> None:
        quantization = object()

        kwargs = model_load_kwargs(force_cpu=False, quantization=quantization)

        self.assertEqual(kwargs["device_map"], "auto")
        self.assertIs(kwargs["quantization_config"], quantization)


if __name__ == "__main__":
    unittest.main()
