#!/bin/bash
# Creates a dedicated virtual environment and installs dependencies.
# Run once: bash setup.sh
# Then activate with: source .venv/bin/activate

set -e

VENV_DIR=".venv"
# langchain>=1.2.0 requires Python >=3.10; use python3.11 if available
PYTHON=$(command -v python3.11 || command -v python3.10 || command -v python3)

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_MINOR=10
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    echo "ERROR: Python $PY_VERSION found, but langchain>=1.2.0 requires Python >=3.10."
    echo "       Install Python 3.11+ and re-run, or set PYTHON=/path/to/python3.11 before running."
    exit 1
fi

echo "→ Using $PYTHON ($PY_VERSION)"
echo "→ Creating virtual environment in $VENV_DIR …"
"$PYTHON" -m venv "$VENV_DIR"

echo "→ Upgrading pip …"
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

echo "→ Installing dependencies …"
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "✓ Done. To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run the app with:"
echo "  streamlit run app.py"
