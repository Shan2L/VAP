from __future__ import annotations

import unittest

from vap.clock_probe.summary import clock_summary_rows, format_clock_summary


class ClockSummaryTests(unittest.TestCase):
    def test_software_summary_reports_selected_fit_and_error_budget(self) -> None:
        session = {
            "clock_source": "udp_software",
            "status": "PASS",
            "execution": {
                "requested_mode": "auto",
                "selected_mode": "software",
            },
            "models": [
                {
                    "model_type": "identity",
                    "status": "PASS",
                    "source": {"hostname": "master"},
                },
                {
                    "model_type": "piecewise_affine",
                    "status": "PASS",
                    "source": {"hostname": "worker-1"},
                    "model_selection": {
                        "selected_config": {
                            "model_method": "piecewise_affine",
                            "window_seconds": 5.0,
                            "samples_per_window": 2,
                            "rtt_slack_us": 10.0,
                            "segment_seconds": 30.0,
                        },
                        "final_score": {
                            "max_total_uncertainty_us": 5.75,
                        },
                    },
                    "realtime_monotonic_bridge": {
                        "segments": [{"status": "PASS", "uncertainty_us": 1.25}]
                    },
                    "segments": [{"status": "PASS", "uncertainty_us": 4.5}],
                },
            ],
        }

        rows = clock_summary_rows(session)
        worker = rows[1]
        summary = format_clock_summary(session)

        self.assertEqual(worker["mode"], "SOFTWARE")
        self.assertIn("window=5.0", worker["parameters"])
        self.assertEqual(worker["bridge_us"], 1.25)
        self.assertEqual(worker["network_us"], 4.5)
        self.assertEqual(worker["end_to_end_us"], 5.75)
        self.assertIn("| MAX | SOFTWARE |", summary)

    def test_hardware_summary_reports_bridge_ptp_and_total_error(self) -> None:
        session = {
            "clock_source": "ptp_hardware",
            "status": "PASS",
            "execution": {
                "requested_mode": "auto",
                "selected_mode": "hardware",
            },
            "models": [
                {
                    "model_type": "phc_bridge",
                    "status": "PASS",
                    "source": {"hostname": "worker-1"},
                    "realtime_phc_bridge": {
                        "model_selection": {
                            "selected_method": "piecewise_affine",
                            "selected_parameter": 2.0,
                        }
                    },
                    "bridge_uncertainty_us": 0.8,
                    "ptp_uncertainty_us": 0.3,
                    "uncertainty_us": 1.1,
                }
            ],
        }

        row = clock_summary_rows(session)[0]

        self.assertEqual(row["mode"], "HARDWARE")
        self.assertIn("method=piecewise_affine", row["parameters"])
        self.assertEqual(row["bridge_us"], 0.8)
        self.assertEqual(row["network_us"], 0.3)
        self.assertEqual(row["end_to_end_us"], 1.1)


if __name__ == "__main__":
    unittest.main()
