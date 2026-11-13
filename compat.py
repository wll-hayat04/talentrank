"""Compatibilité entre versions de Streamlit.

Les versions récentes utilisent width="stretch" ; les anciennes use_container_width=True.
Ce module traduit automatiquement width="stretch" pour la version installée.
"""
import functools
import inspect

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

_NAMES = ["button", "download_button", "dataframe", "data_editor", "plotly_chart"]


def _supports_stretch(fn) -> bool:
    try:
        p = inspect.signature(fn).parameters.get("width")
    except (TypeError, ValueError):
        return False
    return p is not None and isinstance(p.default, str)  # ex. "content" / "stretch"


def _wrap(fn):
    if _supports_stretch(fn):
        return fn
    try:
        has_ucw = "use_container_width" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        has_ucw = False

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        if kwargs.get("width") == "stretch":
            kwargs.pop("width")
            if has_ucw:
                kwargs["use_container_width"] = True
        return fn(*args, **kwargs)

    return inner


def patch():
    for name in _NAMES:
        if hasattr(DeltaGenerator, name):
            setattr(DeltaGenerator, name, _wrap(getattr(DeltaGenerator, name)))
        if hasattr(st, name):
            setattr(st, name, _wrap(getattr(st, name)))
