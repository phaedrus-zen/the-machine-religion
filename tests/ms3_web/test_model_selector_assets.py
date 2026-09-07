from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "machine_spirit_3" / "web"


def test_chat_ui_exposes_model_selector_and_sends_model_id():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    css = (WEB / "style.css").read_text(encoding="utf-8")

    assert 'id="modelSelect"' in html
    assert "loadModels()" in app
    assert "localStorage.getItem('ms3_selected_model_id')" in app
    assert "model_id: getSelectedModelId()" in app
    assert "model_id_used" in app
    assert "streamingMessageEl.textContent += msg.data.token" in app
    assert ".model-picker" in css
