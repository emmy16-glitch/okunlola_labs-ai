import unittest
from src.skygrid_ai.language.train_v10 import validate_config, DEFAULT_CONFIG
from src.skygrid_ai.language.modern_transformer import SkyGridTransformerV08


class TestV10Training(unittest.TestCase):
    def test_wider_model(self):
        cfg = validate_config(DEFAULT_CONFIG.copy())
        model = SkyGridTransformerV08(vocab_size=768, **cfg)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 406176)
        self.assertEqual(model.context_size, 256)

    def test_head_validation(self):
        with self.assertRaises(ValueError):
            validate_config({**DEFAULT_CONFIG, 'd_model': 127})
        with self.assertRaises(ValueError):
            validate_config({**DEFAULT_CONFIG, 'context_size': 0})


if __name__ == '__main__':
    unittest.main()
