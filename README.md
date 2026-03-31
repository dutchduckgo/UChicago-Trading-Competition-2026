# UChicago-Trading-Competition-2026

## Python Setup (macOS/Homebrew)

Homebrew Python uses an externally managed environment (PEP 668), so install
dependencies in a local virtual environment instead of system Python.

```bash
cd case_1
python3 -m venv ../.venv
source ../.venv/bin/activate
python -m pip install -U pip
python -m pip install git+https://github.com/UChicagoFM/utcxchangelib.git
```

## Dependency Compatibility Notes

`utcxchangelib` currently expects newer Protobuf/grpc runtime versions than some
default installs. If import fails with version mismatch errors, run:

```bash
source .venv/bin/activate
python -m pip install "protobuf>=6.31.1,<7" "grpcio>=1.78.0"
```

Quick verification:

```bash
python -c "import utcxchangelib; print('ok')"
```
