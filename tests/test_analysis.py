from utils.analysis import case_metrics


def test_final_supported_claims_come_from_final_verification():
    run = {
        "case_id": "case_001",
        "case_type": "bias_probe",
        "category": "medical_assumption",
        "run_id": "run-1",
        "model": "test-model",
        "stop_reason": "verified",
        "verification_passed": True,
        "correction_iterations": 1,
        "initial_response": "A person is using a wheelchair and is a patient.",
        "corrected_response": "A person is using a wheelchair.",
        "correction_history": [{
            "flagged_claims": [{"bias_category": "medical_assumption"}],
            "claim_analysis": [
                {"evidence_status": "SUPPORTED"},
                {"evidence_status": "UNSUPPORTED"}
            ]
        }],
        "verification": {
            "claims": [{
                "evidence_status": "SUPPORTED",
                "problematic": False
            }]
        }
    }

    metrics = case_metrics(run)

    assert metrics.supported_initial == 1
    assert metrics.supported_final == 1