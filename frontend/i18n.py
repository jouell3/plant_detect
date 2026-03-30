import streamlit as st

APP_LANGUAGE_KEY = "app_language"

LANGUAGE_LABELS = {
    "fr": "Francais",
    "en": "English",
}


def get_language(default: str = "fr") -> str:
    lang = st.session_state.get(APP_LANGUAGE_KEY, default)
    if lang not in LANGUAGE_LABELS:
        lang = default
    st.session_state[APP_LANGUAGE_KEY] = lang
    return lang


def is_english() -> bool:
    return get_language() == "en"


def render_language_selector() -> str:
    current = get_language()
    options = list(LANGUAGE_LABELS.keys())
    selected = st.selectbox(
        "Language / Langue",
        options=options,
        index=options.index(current),
        format_func=lambda code: LANGUAGE_LABELS[code],
        key="language_selector",
    )
    st.session_state[APP_LANGUAGE_KEY] = selected
    return selected
