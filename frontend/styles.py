"""Centralized styling and UX components for Plant Detect app."""

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Color tokens — semantic confidence visualization
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "success": "#2e7d32",        # ≥90% confidence, "good" labels
    "warning": "#f57c00",        # 50-90% confidence, caution states
    "error": "#c62828",          # <50% confidence, critical states
    "text_primary": "#212121",   # Main text
    "text_muted": "#616161",     # Secondary text (timestamps, captions)
    "text_disabled": "#bdbdbd",  # Disabled button text
    "border": "#e0e0e0",         # Card/divider borders
    "background_light": "#f5f5f5",  # Light backgrounds
}

# ─────────────────────────────────────────────────────────────────────────────
# Typography — standardized Streamlit + inline HTML styles
# ─────────────────────────────────────────────────────────────────────────────
TYPOGRAPHY = {
    "caption_muted": f"font-size: 0.75rem; color: {COLORS['text_muted']}; margin-top: 0.25rem;",
    "label_small": f"font-size: 0.875rem; color: {COLORS['text_primary']}; font-weight: 500;",
    "hint_small": f"font-size: 0.8rem; color: {COLORS['text_muted']};",
}


def confidence_color(confidence: float) -> str:
    """Return color hex for a confidence value (0-1)."""
    if confidence >= 0.90:
        return COLORS["success"]
    elif confidence >= 0.50:
        return COLORS["warning"]
    else:
        return COLORS["error"]


def confidence_badge(species: str, confidence: float, model_name: str | None = None) -> None:
    """Render a styled confidence card with semantic coloring.
    
    Args:
        species: Plant species name to display
        confidence: Confidence value (0-1)
        model_name: Optional model identifier (e.g., "pytorch", "sklearn")
    """
    color = confidence_color(confidence)
    muted_color = COLORS["text_muted"]
    subtitle_html = f"<div style='font-size: 0.75rem; color: {muted_color}; margin-top: 4px;'>{model_name}</div>" if model_name else ""
    
    html = f"""
    <div style='
        background: {color}11; 
        border-left: 4px solid {color};
        padding: 12px; 
        border-radius: 4px; 
        margin: 8px 0;
    '>
        <div style='font-weight: 700; color: {COLORS["text_primary"]};'>
            {species}
        </div>
        <div style='font-size: 1.1rem; color: {color}; font-weight: 600;'>
            {confidence:.0%}
        </div>
        {subtitle_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def styled_info_card(title: str, content_dict: dict[str, str], subtitle: str | None = None) -> None:
    """Render a visually distinct info/prediction card.
    
    Args:
        title: Card title
        content_dict: Dictionary of key-value pairs to display (key: label, value: content)
        subtitle: Optional subtitle or metadata
    """
    content_html = "".join(
        f"<div style='margin-bottom: 8px;'><strong>{k}:</strong> {v}</div>"
        for k, v in content_dict.items()
    )
    muted_color = COLORS["text_muted"]
    subtitle_html = f"<div style='font-size: 0.85rem; color: {muted_color}; margin-bottom: 8px;'>{subtitle}</div>" if subtitle else ""
    
    # Keep all style attributes on single lines for proper Streamlit HTML rendering
    html = f"<div style='background: {COLORS['background_light']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 16px; margin: 12px 0;'><div style='font-weight: 700; font-size: 1.85rem; margin-bottom: 8px;'>{title}</div>{subtitle_html}<div style='font-size: 1.2rem; line-height: 1.6; color: {COLORS['text_primary']};'>{content_html}</div></div>"
    st.markdown(html, unsafe_allow_html=True)


def page_header(title: str, description: str = "", icon: str = "") -> None:
    """Render a consistent page header with title and optional description.
    
    Args:
        title: Page title
        description: Optional description/introduction text
        icon: Optional emoji or icon prefix
    """
    full_title = f"{icon} {title}" if icon else title
    st.title(full_title)
    if description:
        st.markdown(description)
