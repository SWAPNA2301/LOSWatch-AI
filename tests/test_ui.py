from pathlib import Path

from streamlit.testing.v1 import AppTest

from loswatch.seed import seed_database


def test_dashboard_views_and_crud(factory, monkeypatch):
    seed_database(factory)
    monkeypatch.setenv("LOSWATCH_DATABASE_URL", str(factory.kw["bind"].url))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.metric[0].value == "20"
    app.sidebar.radio[0].set_value("Patients").run()
    assert not app.exception
    app.text_input[0].set_value("DEMO-0001").run()
    assert not app.exception
    assert app.subheader[0].value == "Aarav Meadow"
    for page in ("Doctors", "Beds", "Manage records"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    for kind in ("Patient", "Admission", "Diagnosis", "MedicationOrder", "Bed"):
        app.selectbox[0].set_value(kind).run()
        assert not app.exception
    app.selectbox[0].set_value("Doctor").run()
    app.text_input[0].set_value("Dr. UI Fiction")
    app.text_input[1].set_value("General")
    app.button[0].click().run()
    assert not app.exception
    assert app.success
    app.radio[0].set_value("Update").run()
    app.selectbox[1].set_value(5).run()
    app.text_input[1].set_value("Demo specialty")
    app.button[0].click().run()
    assert not app.exception
    app.radio[0].set_value("Delete").run()
    app.selectbox[1].set_value(5).run()
    app.checkbox[0].check()
    app.button[0].click().run()
    assert not app.exception
    assert app.success


def test_empty_database(factory, monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    monkeypatch.setenv("LOSWATCH_DATABASE_URL", str(factory.kw["bind"].url))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.metric[0].value == "0"


def test_home_command_center_and_operational_filters(factory, monkeypatch):
    seed_database(factory)
    monkeypatch.setenv("LOSWATCH_DATABASE_URL", str(factory.kw["bind"].url))
    import streamlit as st
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Home"
    assert app.metric[0].value == "20"
    assert app.metric[1].value == "15"
    app.sidebar.radio[0].set_value("Beds").run()
    assert not app.exception
    assert any("Showing" in item.value for item in app.caption)
    app.sidebar.radio[0].set_value("Doctors").run()
    assert not app.exception
