import unittest

import rl_cbf


class VersionTestCase(unittest.TestCase):
    """ Version tests """

    def test_version(self):
        """ check rl_cbf exposes a version attribute """
        self.assertTrue(hasattr(rl_cbf, "__version__"))
        self.assertIsInstance(rl_cbf.__version__, str)


if __name__ == "__main__":
    unittest.main()
