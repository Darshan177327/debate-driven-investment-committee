"""Controller-level scenario tests with external agents and logging mocked.

These tests exercise the committee decision flow without API calls, document
retrieval, or changes to the persistent ChromaDB database.
"""

import unittest
from unittest.mock import patch

from agents import committee_controller


class TestCommitteeScenarios(unittest.TestCase):
    """Three representative investment committee scenarios."""

    @patch("agents.committee_controller.log_committee_run")
    @patch("agents.committee_controller.run_risk_agent")
    @patch("agents.committee_controller.run_committee_judge")
    @patch("agents.committee_controller.run_rebuttal_agent")
    def test_positive_case_without_risk_veto(
        self, mock_debate, mock_judge, mock_risk, mock_log
    ):
        """A BUY from the Judge is retained when Risk does not veto."""
        mock_debate.return_value = "Research, Bull, Bear and rebuttal transcript"
        mock_judge.return_value = {
            "recommendation": "BUY",
            "analysis": "The cited evidence supports a positive case."
        }
        mock_risk.return_value = {
            "risk_score": 35,
            "threshold": 70,
            "veto": False
        }

        result = committee_controller.run_committee("Scenario Positive Ltd")

        self.assertEqual(result["judge_recommendation"], "BUY")
        self.assertEqual(result["final_recommendation"], "BUY")
        self.assertIn("retained", result["decision_reason"].lower())
        mock_log.assert_called_once()

    @patch("agents.committee_controller.log_committee_run")
    @patch("agents.committee_controller.run_risk_agent")
    @patch("agents.committee_controller.run_committee_judge")
    @patch("agents.committee_controller.run_rebuttal_agent")
    def test_negative_case_remains_avoid(
        self, mock_debate, mock_judge, mock_risk, mock_log
    ):
        """An AVOID decision remains AVOID when the Risk veto is not triggered."""
        mock_debate.return_value = "Debate transcript identifying material concerns"
        mock_judge.return_value = {
            "recommendation": "AVOID",
            "analysis": "The negative evidence outweighs the positive case."
        }
        mock_risk.return_value = {
            "risk_score": 55,
            "threshold": 70,
            "veto": False
        }

        result = committee_controller.run_committee("Scenario Negative Ltd")

        self.assertEqual(result["judge_recommendation"], "AVOID")
        self.assertEqual(result["final_recommendation"], "AVOID")
        mock_log.assert_called_once()

    @patch("agents.committee_controller.log_committee_run")
    @patch("agents.committee_controller.run_risk_agent")
    @patch("agents.committee_controller.run_committee_judge")
    @patch("agents.committee_controller.run_rebuttal_agent")
    def test_high_risk_overrides_buy_to_hold(
        self, mock_debate, mock_judge, mock_risk, mock_log
    ):
        """A triggered Risk veto changes a Judge BUY recommendation to HOLD."""
        mock_debate.return_value = "Debate transcript with elevated risk indicators"
        mock_judge.return_value = {
            "recommendation": "BUY",
            "analysis": "The Judge finds a positive investment case."
        }
        mock_risk.return_value = {
            "risk_score": 82,
            "threshold": 70,
            "veto": True
        }

        result = committee_controller.run_committee("Scenario High Risk Ltd")

        self.assertEqual(result["judge_recommendation"], "BUY")
        self.assertEqual(result["final_recommendation"], "HOLD")
        self.assertIn("veto", result["decision_reason"].lower())
        mock_log.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
