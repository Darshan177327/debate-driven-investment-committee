
import unittest

from agents.committee_controller import apply_risk_veto


class TestCommitteeDecision(unittest.TestCase):

    def test_buy_without_risk_veto(self):
        risk_result = {"veto": False}

        final, reason = apply_risk_veto("BUY", risk_result)

        self.assertEqual(final, "BUY")

    def test_risk_veto_overrides_buy(self):
        risk_result = {"veto": True}

        final, reason = apply_risk_veto("BUY", risk_result)

        self.assertEqual(final, "HOLD")
        self.assertIn("veto", reason.lower())

    def test_avoid_without_risk_veto(self):
        risk_result = {"veto": False}

        final, reason = apply_risk_veto("AVOID", risk_result)

        self.assertEqual(final, "AVOID")

    def test_missing_risk_result_defaults_to_hold(self):
        final, reason = apply_risk_veto("BUY", None)

        self.assertEqual(final, "HOLD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
