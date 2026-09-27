"""Pre-run the demo in the background so every click is instant on camera.

Drives the real Streamlit page headlessly (Streamlit's AppTest) for the demo
company, once per model: presses "Get recommendation" and then "Run the
check". Every model answer is saved in the app's answer cache, so during the
recording both buttons respond immediately. Needs Ollama running; the check
takes a few minutes per model the first time.

Usage (from the thesis folder):
    code/.venv/bin/python demo/warm_demo_cache.py
"""

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from streamlit.testing.v1 import AppTest  # noqa: E402

MODELS = ["Llama 3.1 8B", "Qwen3 8B"]


def warm(model_label):
    """Run both buttons once for one model and print what the page showed."""
    app = AppTest.from_file(str(CODE / "streamlit_app.py"), default_timeout=1200)
    app.run()
    app.sidebar.selectbox(key="model").select(model_label).run()
    app.button(key="get_rec").click().run()
    app.button(key="check_claim").click().run()

    texts = [str(m.value) for m in app.markdown]
    action = next((t for t in texts if t.startswith("## :")), "?")
    checks = [s.value for s in app.success]
    print(f"{model_label}: recommendation {action}")
    for line in checks:
        print(f"   check: {line}")
    if app.exception:
        print("   error:", app.exception)


if __name__ == "__main__":
    for label in MODELS:
        warm(label)
