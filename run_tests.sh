# run_tests.sh
#!/bin/bash

# Run unit tests
echo "Running unit tests..."
pytest tests/unit/ -v --cov=ml_core_v15_2 --cov-report=term-missing

# Run integration tests
echo -e "\nRunning integration tests..."
pytest tests/integration/ -v

# Run performance benchmarks
echo -e "\nRunning performance benchmarks..."
python -m pytest tests/performance/ -v

# Generate coverage report
echo -e "\nGenerating coverage report..."
pytest --cov=ml_core_v15_2 --cov-report=html:coverage_report