import sys
import subprocess
import json

def test_cli_rstar_returns_json():
    cmd = [sys.executable, "-m", "adv_resolver_math.cli", "rstar", "x = 2 + 2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    assert "solutions" in data
    assert "final_answer" in data
    assert "final_confidence" in data
    assert "supporting_agents" in data
