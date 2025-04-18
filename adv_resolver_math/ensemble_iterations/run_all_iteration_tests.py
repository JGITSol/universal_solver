import subprocess
import sys
import os

def run_pytest(test_file):
    print(f"\nRunning tests in {test_file}...")
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    result = subprocess.run([
        sys.executable, '-m', 'pytest', '--cov=adv_resolver_math', '--cov-report=term-missing', test_file
    ], capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise AssertionError(f"Tests failed in {test_file}")
    print(f"All tests passed for {test_file}\n")

def main():
    test_files = [
        'test_enhanced_solver.py',
        'test_memory_sharing_solver.py',
        'test_latent_space_solver.py',
        'test_rstar_math_solver.py',
    ]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for tf in test_files:
        run_pytest(os.path.join(base_dir, tf))

if __name__ == '__main__':
    main()
