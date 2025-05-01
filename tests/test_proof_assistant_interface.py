import pytest
from adv_resolver_math.proof_assistant_interface import ProofAssistantInterface

def test_check_expression_stub():
    pai = ProofAssistantInterface(assistant_type="lean")
    assert pai.check_expression("x + y = y + x") is True

def test_verify_proof_stub():
    pai = ProofAssistantInterface(assistant_type="coq")
    assert pai.verify_proof(["step1", "step2"]) is True

def test_init_with_path():
    pai = ProofAssistantInterface(assistant_type="isabelle", path_to_executable="/usr/bin/isabelle")
    assert pai.assistant_type == "isabelle"
    assert pai.path_to_executable == "/usr/bin/isabelle"
