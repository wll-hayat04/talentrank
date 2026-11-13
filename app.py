"""TalentRank — aide au recrutement par LLM + méthodes MCDM.

Lancer :  streamlit run app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import compat

compat.patch()

from mcdm import analysis as AN
from mcdm import fuzzy as FZ
from mcdm import ranking as RK
from mcdm import weighting as WG
import llm as LLM

ROOT = Path(__file__).parent
DATA = ROOT / "data"

import themes

st.set_page_config(page_title="TalentRank", layout="wide")
TH = themes.THEMES[themes.DEFAULT]
SERIES = TH["series"]
GRAY = TH["line2"]
SEQ_BLUE = [[0, TH["seq"][0]], [1, TH["seq"][1]]]            # séquentielle : plus clair = plus fort
DIVERGING = [[0, TH["div"][0]], [0.5, TH["div"][1]], [1, TH["div"][2]]]
RANK_SCALE = [[0, TH["seq"][1]], [1, TH["seq"][0]]]          # rang 1 = le plus lumineux

CSS = "".join((ROOT / "assets" / f).read_text(encoding="utf-8") for f in ("fonts.css", "style.css"))
st.markdown(f"<style>{CSS}{themes.css_vars(TH)}</style>" + ('<span class="proj-on"></span>' if TH.get("light") else ""), unsafe_allow_html=True)


# =============================================================================== état
def load_demo():
    demo = json.loads((DATA / "demo.json").read_text(encoding="utf-8"))
    ss = st.session_state
    ss.job = demo["poste"]
    ss.criteria = demo["criteres"]
    # Le texte des CV est intégré au JSON (les fichiers data/cvs/*.txt ne sont que des copies lisibles)
    ss.cvs = {c["nom"]: c.get("cv") or _read_cv(c["fichier"]) for c in demo["candidats"]}
    llm_codes = [c["code"] for c in demo["criteres"] if c["source"] == "llm"]
    ss.scores = pd.DataFrame([{k: c["scores"][k] for k in llm_codes} for c in demo["candidats"]],
                             index=[c["nom"] for c in demo["candidats"]], dtype=float)
    ss.std = pd.DataFrame(0.0, index=ss.scores.index, columns=ss.scores.columns)
    ss.justif = {c["nom"]: c["justifications"] for c in demo["candidats"]}
    ss.recruiters = demo["recruteurs"]
    ss.soft = pd.DataFrame({r["nom"]: r["soft_skills"] for r in demo["recruteurs"]})
    ss.bounds = demo["bornes_maut"]
    ss.ahp = {r["nom"]: ahp_from_priorities(r["priorites_ahp"]) for r in demo["recruteurs"]}
    # Décisions passées fictives (pour le seuil issu des données) : 1 = retenu
    ss.past = pd.Series([1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], index=ss.scores.index)
    ss.source_scores = "démo (pré-calculés)"


RESULTS = DATA / "llm_results.json"


def save_llm_results():
    """Enregistre sur disque les scores LLM : ils survivent à un rafraîchissement de la page."""
    ss = st.session_state
    out = {"source": ss.source_scores, "date": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
           "scores": ss.scores.to_dict("index"), "std": ss.std.to_dict("index"), "justif": ss.justif}
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def load_llm_results():
    ss = st.session_state
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    for name, row in r["scores"].items():
        if name in ss.scores.index:
            for code, v in row.items():
                ss.scores.loc[name, code] = np.nan if v is None else v
            for code, v in r["std"].get(name, {}).items():
                ss.std.loc[name, code] = 0.0 if v is None else v
            ss.justif[name] = r["justif"].get(name, {})
    ss.source_scores = r["source"]
    ss.pop("scores_editor", None)


def _read_cv(fichier: str) -> str:
    path = DATA / "cvs" / fichier
    return path.read_text(encoding="utf-8") if path.exists() else ""


def ahp_from_priorities(p):
    """Matrice AHP de départ : a_ij = rapport p_i/p_j arrondi à l'échelle de Saaty."""
    def saaty(r):
        return float(np.clip(round(r), 1, 9)) if r >= 1 else 1 / float(np.clip(round(1 / r), 1, 9))
    n = len(p)
    return np.array([[saaty(p[i] / p[j]) for j in range(n)] for i in range(n)])


if "criteria" not in st.session_state:
    load_demo()
ss = st.session_state
CODES = [c["code"] for c in ss.criteria]
TYPES = [c["type"] for c in ss.criteria]
LABELS = [f'{c["code"]} {c["nom"]}' for c in ss.criteria]
LLM_CRIT = [c for c in ss.criteria if c["source"] == "llm"]
NAMES = list(ss.scores.index)


def soft_tfn() -> np.ndarray:
    """C6 : agrégation (moyenne) des nombres flous donnés par les recruteurs → (m, 3)."""
    return np.array([FZ.aggregate([FZ.RATINGS[ss.soft.loc[n, r]] for r in ss.soft.columns]) for n in NAMES])


def decision_matrix() -> pd.DataFrame:
    X = ss.scores.copy()
    X["C6"] = [FZ.defuzzify(t) for t in soft_tfn()]
    return X[CODES]


# ============================================================================ graphes
def style(fig, h=380):
    has_legend = sum(1 for t in fig.data if t.showlegend is not False) > 1
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=64 if has_legend else 40, b=10), plot_bgcolor=TH["sheet"],
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TH["ink"], size=17 if TH.get("light") else 14, family="Manrope, Segoe UI, Arial, sans-serif"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), hoverlabel=dict(bgcolor=TH["raised"], font=dict(color=TH["ink"])))
    _tf = dict(size=16 if TH.get("light") else 13, color=TH["ink"])
    fig.update_xaxes(gridcolor=TH["line"], zeroline=False, linecolor=TH["line2"], automargin=True, tickfont=_tf)
    fig.update_yaxes(gridcolor=TH["line"], zeroline=False, linecolor=TH["line2"], automargin=True, tickfont=_tf)
    return fig


def weights_chart(wdict: dict):
    fig = go.Figure()
    for k, (name, w) in enumerate(wdict.items()):
        fig.add_bar(name=name, x=CODES, y=w, marker_color=SERIES[k % len(SERIES)],
                    marker_line_width=0, hovertemplate=f"{name}<br>%{{x}} : %{{y:.3f}}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, yaxis_title="Poids")
    return style(fig)


def heatmap(M, labels, title, colorscale, zmin, zmax, fmt=".2f", ylabels=None):
    dark_text = colorscale is not DIVERGING
    fig = go.Figure(go.Heatmap(z=M, x=labels, y=ylabels or labels, colorscale=colorscale, zmin=zmin, zmax=zmax,
                               text=np.vectorize(lambda v: format(v, fmt))(M), texttemplate="%{text}", textfont=dict(color=themes.BASE["heat_text"] if (dark_text or TH.get("light")) else TH["ink"], size=17 if TH.get("light") else 14),
                               xgap=2, ygap=2, showscale=False, hovertemplate="%{y} / %{x} : %{z:" + fmt + "}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed")
    return style(fig, 120 + 34 * len(ylabels or labels))


def formula(latex, note=None):
    """Formule mathématique encadrée (rendu LaTeX/KaTeX)."""
    st.latex(latex)
    if note:
        st.caption(note)


def K(x):
    """Met en couleur d'accent la grandeur calculée dans une formule."""
    return r"\textcolor{" + TH["accent"] + "}{" + x + "}"


_side_logo = ('<svg viewBox="0 0 46 46" width="40" height="40" aria-hidden="true">'
              f'<rect width="46" height="46" rx="11" fill="{TH["accent"]}"/>'
              + ''.join(f'<rect x="9" y="{y}" width="{w}" height="6" rx="3" fill="#fff" fill-opacity="{o}"/>'
                        for y, w, o in ((10, 28, 1), (20, 20, .8), (30, 12, .6))) + '</svg>')

LLM_SECRETS = ("GROQ_API_KEY", "LLM_API_KEY")


def _read_secret(names) -> str:
    """Clé API lue dans les secrets Streamlit (.streamlit/secrets.toml ou réglages du déploiement), sinon vide."""
    try:
        for n in names:
            if n in st.secrets:
                return str(st.secrets[n])
    except Exception:  # noqa: BLE001 — pas de fichier de secrets en local
        pass
    return ""


def explain_llm_error(e: Exception, base_url: str) -> str:
    """Message d'erreur compréhensible pour l'utilisateur."""
    raw = f"{type(e).__name__} : {e}"
    if "localhost" in (base_url or "") or "127.0.0.1" in (base_url or ""):
        return ("Ollama est injoignable. Vérifiez qu'il est lancé sur cet ordinateur (commande « ollama serve »). "
                "Sur la version en ligne, Ollama ne peut pas fonctionner : choisissez Groq et collez une clé API, "
                f"ou restez en mode Démo. Détail : {raw}")
    if "401" in raw or "Authentication" in raw or "api_key" in raw.lower():
        return f"Clé API refusée ou manquante. Vérifiez la clé du fournisseur. Détail : {raw}"
    return f"Connexion impossible. Détail : {raw}"


# ============================================================================ sidebar
with st.sidebar:
    st.markdown('<div class="side-brand2">' + _side_logo + '<div><b>Talent<span>Rank</span></b>'
                '<small>Recrutement assisté par IA et MCDM</small></div></div>', unsafe_allow_html=True)
    mode = st.radio("Source des scores", ["Démo (scores pré-calculés)", "LLM réel"], index=0)
    llm_cfg = {}
    if mode == "LLM réel":
        _secret_key = _read_secret(LLM_SECRETS)
        _provs = list(LLM.PROVIDERS)
        # En ligne (clé fournie dans les secrets du serveur), Groq est proposé par défaut : Ollama n'y est pas joignable.
        prov = st.selectbox("Fournisseur", _provs, index=_provs.index("Groq") if _secret_key else 0)
        llm_cfg["base_url"] = st.text_input("URL de l'API", LLM.PROVIDERS[prov]["base_url"])
        llm_cfg["model"] = st.text_input("Modèle", LLM.PROVIDERS[prov]["model"])
        llm_cfg["api_key"] = st.text_input("Clé API", type="password",
                                           help="Inutile pour Ollama. La clé reste dans la session.") or (
            _secret_key if prov != "Ollama (local)" else "")
        if prov == "Ollama (local)":
            st.caption("Ollama ne fonctionne qu'avec l'application lancée sur votre ordinateur. "
                       "Sur la version en ligne, choisissez Groq et une clé API.")
        llm_cfg["n_runs"] = st.slider("Passages par CV (mesure de stabilité)", 1, 5, 3)
    st.divider()
    if RESULTS.exists():
        _info = json.loads(RESULTS.read_text(encoding="utf-8"))
        if st.button(f"Charger les derniers scores LLM ({_info['date']})", width="stretch"):
            load_llm_results()
            st.rerun()
    if st.button("Recharger la démo", width="stretch"):
        for k in list(ss.keys()):
            if k != "theme":
                del ss[k]
        st.rerun()
    with st.expander("Mode d'emploi", expanded=False):
        st.markdown("**1 Poste** : fixer les critères et leur sens.  \n**2 Candidats** : contrôler les notes extraites.  \n"
                    "**3 Pondération** : comparer AHP, BWM, entropie, CRITIC.  \n**4 Classement** : confronter 8 méthodes.  \n"
                    "**5 Évaluation** : vérifier le seuil d'admissibilité.  \n**6 Robustesse** : sensibilité, fiabilité, biais.")
        st.caption("Aide à la décision : un évaluateur humain doit vérifier les scores générés par IA. Données de démonstration fictives.")

_logo = ('<svg class="mark" viewBox="0 0 46 46" aria-hidden="true">'
         f'<rect width="46" height="46" rx="11" fill="{TH["accent"]}"/>'
         + ''.join(f'<rect x="9" y="{y}" width="{w}" height="6" rx="3" fill="{TH["on_accent"]}" fill-opacity="{o}"/>' for y, w, o in ((10, 28, 1), (20, 20, .8), (30, 12, .6)))
         + '</svg>')
_live = ss.source_scores.startswith("LLM")
_poste = ss.job["titre"]
st.markdown(
    '<div class="appbar"><div class="appbar-id">'
    + _logo +
    '<div><div class="wordmark">Talent<span>Rank</span></div>'
    '<div class="tagline">Recrutement assisté par LLM et décision multicritère</div></div></div>'
    '<div class="appbar-facts">'
    f'<span class="poste-pill">Poste : <b>{_poste}</b></span>'
    f'<span class="status {"live" if _live else ""}"><i></i>Scores : {ss.source_scores}</span>'
    '</div></div>', unsafe_allow_html=True)
tabs = st.tabs(["Tableau de bord", "1 Poste", "2 Candidats", "3 Pondération",
                "4 Classement", "5 Évaluation", "6 Robustesse"])

# ======================================================================= 1. POSTE
with tabs[1]:
    st.header("Poste à pourvoir et critères de décision")
    c1, c2 = st.columns([1, 2])
    ss.job["titre"] = c1.text_input("Intitulé du poste", ss.job["titre"])
    ss.job["description"] = c2.text_area("Description (envoyée au LLM)", ss.job["description"], height=110)
    st.subheader("Critères")
    st.caption("Le **type** indique le sens de préférence (max = à maximiser, min = à minimiser). "
               "La **grille** est la règle de notation donnée au LLM : plus elle est précise, plus les scores sont stables.")
    dfc = pd.DataFrame(ss.criteria)[["code", "nom", "type", "source", "grille"]]
    edited = st.data_editor(
        dfc, hide_index=True, width="stretch", disabled=["code", "source"],
        column_config={"type": st.column_config.SelectboxColumn("type", options=["max", "min"]),
                       "grille": st.column_config.TextColumn("grille de notation", width="large")})
    for c, (_, row) in zip(ss.criteria, edited.iterrows()):
        c.update(nom=row["nom"], type=row["type"], grille=row["grille"])
    st.info(f"**Formalisation du problème.** {len(ss.cvs)} candidats (les alternatives), {len(ss.criteria)} critères "
            f"(C1 à C{len(ss.criteria)}) et {len(ss.recruiters)} décideurs : " + ", ".join(r["nom"] for r in ss.recruiters) + ".")

# ==================================================================== 2. CANDIDATS
with tabs[2]:
    st.header("Candidats et extraction des scores")
    up = st.file_uploader("Ajouter des CV (.txt ou .pdf)", type=["txt", "pdf"], accept_multiple_files=True)
    if up:
        for f in up:
            if f.name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(f.read())).pages)
            else:
                text = f.read().decode("utf-8", errors="ignore")
            name = Path(f.name).stem.replace("_", " ").title()
            if name not in ss.cvs:
                ss.cvs[name] = text
                ss.scores.loc[name] = np.nan
                ss.std.loc[name] = 0.0
                ss.soft.loc[name] = "Moyen"
                ss.past.loc[name] = 0
        ss.pop("scores_editor", None)
        ss.pop("soft_editor", None)

    if mode == "LLM réel":
        t1, t2 = st.columns([1, 1])
        if t2.button("Tester la connexion au LLM"):
            try:
                with st.spinner("Appel du modèle (le premier appel peut durer une minute)…"):
                    rep = LLM._chat(llm_cfg["base_url"], llm_cfg["api_key"], llm_cfg["model"], 0.0,
                                    [{"role": "user", "content": "Réponds uniquement par le mot : ok"}])
                st.success(f"Connexion réussie. Réponse du modèle : « {rep.strip()[:80]} »")
            except Exception as e:  # noqa: BLE001
                st.error(explain_llm_error(e, llm_cfg["base_url"]))
        if t1.button("Lancer l'extraction sur tous les CV", type="primary"):
            bar = st.progress(0.0)
            status = st.empty()
            errors, ok_count = [], 0
            for k, name in enumerate(list(ss.cvs)):
                status.info(f"Extraction {k + 1}/{len(ss.cvs)} : {name} …")
                try:
                    res = LLM.score_cv(ss.cvs[name], ss.job, LLM_CRIT, base_url=llm_cfg["base_url"],
                                       api_key=llm_cfg["api_key"], model=llm_cfg["model"], n_runs=llm_cfg["n_runs"])
                    for code, v in res.items():
                        ss.scores.loc[name, code] = v["score"]
                        ss.std.loc[name, code] = v["std"]
                    ss.justif[name] = {code: v["justification"] for code, v in res.items()}
                    ok_count += 1
                    ss.source_scores = f"LLM ({llm_cfg['model']})"
                    save_llm_results()
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{name} : {type(e).__name__} : {e}")
                bar.progress((k + 1) / len(ss.cvs))
            ss.llm_report = {"ok": ok_count, "total": len(ss.cvs), "errors": errors}
            if ok_count:
                ss.source_scores = f"LLM ({llm_cfg['model']})"
            ss.pop("scores_editor", None)
            st.rerun()
        rep = ss.get("llm_report")
        if rep:
            if rep["errors"]:
                st.error(f"Extraction terminée : {rep['ok']}/{rep['total']} CV réussis. Erreurs :\n\n"
                         + "\n\n".join(rep["errors"][:5]))
            else:
                st.success(f"Extraction terminée : {rep['ok']}/{rep['total']} CV réussis.")
    else:
        st.caption("Mode démo : les scores ci-dessous ont été pré-remplis pour illustrer la démarche. "
                   "Choisissez « LLM réel » dans la barre latérale pour les faire calculer par un modèle.")

    st.subheader("Matrice des scores extraits (modifiable)")
    st.caption("C1 à C5 sont notés sur 10, C7 est en MAD par mois. Chaque valeur peut être corrigée à la main : la décision reste sous contrôle humain.")
    st.markdown('<div class="crit-key">' + "".join(f'<span><b>{c["code"]}</b> {c["nom"]}</span>' for c in LLM_CRIT)
                + '</div>', unsafe_allow_html=True)
    _ed = st.data_editor(ss.scores.rename_axis("Candidat").reset_index(), width="stretch", key="scores_editor",
                         hide_index=True, disabled=["Candidat"],
                         column_config={"Candidat": st.column_config.TextColumn("Candidat"),
                                        **{c["code"]: st.column_config.NumberColumn(
                                            c["code"], help=c["nom"], format="%.0f" if c.get("numerique") else "%.1f")
                                           for c in LLM_CRIT}})
    ss.scores = _ed.set_index("Candidat").rename_axis(ss.scores.index.name)
    if ss.std.to_numpy().max() > 0:
        st.markdown("**Stabilité du LLM.** Écart-type des notes entre les passages (0 = parfaitement stable).")
        st.dataframe(ss.std.style.format("{:.2f}").background_gradient(cmap="Oranges"), width="stretch")

    st.subheader("C6 Soft skills : évaluations linguistiques des recruteurs")
    st.caption("Chaque terme est un nombre flou triangulaire (échelle de Chen, 2000). "
               "L'avis collectif est la moyenne des nombres flous, puis le centre de gravité donne une note sur 10.")
    formula(r"\tilde{x}_k = (a_k,\, b_k,\, c_k),\qquad " + K(r"\tilde{x}") + r" = \frac{1}{K}\sum_{k=1}^{K} \tilde{x}_k"
            r"\qquad " + K(r"x") + r" = \frac{a + b + c}{3}")
    opts = list(FZ.RATINGS)
    _sf = st.data_editor(ss.soft.rename_axis("Candidat").reset_index(), width="stretch", key="soft_editor",
                         hide_index=True, disabled=["Candidat"],
                         column_config={r: st.column_config.SelectboxColumn(r, options=opts) for r in ss.soft.columns})
    ss.soft = _sf.set_index("Candidat").rename_axis(ss.soft.index.name)
    agg = soft_tfn()
    st.dataframe(pd.DataFrame({"Nombre flou agrégé (a, b, c)": [f"({a:.2f}, {b:.2f}, {c:.2f})" for a, b, c in agg],
                               "Note défuzzifiée C6": [FZ.defuzzify(t) for t in agg]}, index=NAMES)
                 .style.format({"Note défuzzifiée C6": "{:.2f}"}), width="stretch")

    st.subheader("Détail par candidat")
    who = st.selectbox("Candidat", NAMES)
    cl, cr = st.columns(2)
    cl.text_area("CV", ss.cvs.get(who, ""), height=330, disabled=True)
    j = ss.justif.get(who, {})
    cr.dataframe(pd.DataFrame({"Score": [ss.scores.loc[who, c["code"]] for c in LLM_CRIT],
                               "Justification du LLM": [j.get(c["code"], "") for c in LLM_CRIT]},
                              index=[f'{c["code"]} {c["nom"]}' for c in LLM_CRIT]), width="stretch", height=330)

X = decision_matrix()
Xv = X.to_numpy(dtype=float)
if np.isnan(Xv).any():
    st.warning("Certains scores sont manquants : lancez l'extraction LLM ou complétez la matrice dans l'onglet 2.")
    st.stop()

# =================================================================== 3. PONDÉRATION
with tabs[3]:
    st.header("Pondération des critères")
    st.markdown("**Méthodes subjectives** (avis des recruteurs : AHP, BWM) et **objectives** "
                "(calculées sur la matrice : entropie, CRITIC).")
    weights = {}

    st.subheader("AHP")
    r_ahp = st.selectbox("Recruteur qui remplit la matrice", [r["nom"] for r in ss.recruiters])
    st.caption("Échelle de Saaty (1 à 9). Seules les cases **au-dessus de la diagonale** sont lues ; "
               r"le reste est complété par réciprocité : $a_{ji} = 1/a_{ij}$.")
    A0 = pd.DataFrame(ss.ahp[r_ahp], index=CODES, columns=CODES)
    A_ed = st.data_editor(A0.round(3), key=f"ahp_{r_ahp}", width="stretch")
    up_vals = {(i, j): float(A_ed.iloc[i, j]) for i in range(len(CODES)) for j in range(i + 1, len(CODES))}
    ss.ahp[r_ahp] = WG.ahp_matrix_from_upper(len(CODES), up_vals)
    res_ahp = WG.ahp(ss.ahp[r_ahp])
    m1, m2, m3 = st.columns(3)
    formula(K(r"w_i") + r" = \frac{1}{n}\sum_{j=1}^{n}\frac{a_{ij}}{\sum_{k} a_{kj}}"
            r"\qquad " + K(r"CI") + r" = \frac{\lambda_{\max} - n}{n - 1}"
            r"\qquad " + K(r"CR") + r" = \frac{CI}{RI} < 0{,}1")
    m1.metric("λ max", f'{res_ahp["lambda_max"]:.3f}')
    m2.metric("CI", f'{res_ahp["CI"]:.3f}')
    m3.metric("CR", f'{res_ahp["CR"]:.3f}', "cohérente (CR < 0,1)" if res_ahp["consistent"] else "incohérente : revoir la matrice",
              delta_color="normal" if res_ahp["consistent"] else "inverse")
    weights[f"AHP – {r_ahp}"] = res_ahp["weights"]

    st.subheader("BWM (Best-Worst Method)")
    st.caption("Chaque recruteur choisit son critère le plus important (Best) et le moins important (Worst), "
               r"puis compare chaque critère à ces deux repères : $2n - 3$ comparaisons au lieu de $n(n-1)/2$.")
    formula(r"\begin{gathered}\min\ " + K(r"\xi") + r"\quad \text{sous les contraintes} \\[4pt]"
            r"\left|w_B - a_{Bj}\,w_j\right| \le \xi,\qquad \left|w_j - a_{jW}\,w_W\right| \le \xi,"
            r"\qquad \sum_{j} w_j = 1,\quad w_j \ge 0\end{gathered}")
    bwm_w = []
    cols = st.columns(len(ss.recruiters))
    for col, r in zip(cols, ss.recruiters):
        with col:
            st.markdown(f"**{r['nom']}**")
            r["best"] = st.selectbox("Best", CODES, index=CODES.index(r["best"]), key=f"b_{r['nom']}")
            r["worst"] = st.selectbox("Worst", CODES, index=CODES.index(r["worst"]), key=f"w_{r['nom']}")
            vec = pd.DataFrame({"BO": r["bo"], "OW": r["ow"]}, index=CODES)
            vec = st.data_editor(vec, key=f"v_{r['nom']}", width="stretch",
                                 column_config={k: st.column_config.NumberColumn(k, min_value=1, max_value=9, step=1)
                                                for k in ("BO", "OW")})
            r["bo"], r["ow"] = vec["BO"].tolist(), vec["OW"].tolist()
            try:
                res = WG.bwm(CODES.index(r["best"]), CODES.index(r["worst"]), r["bo"], r["ow"])
                bwm_w.append(res["weights"])
                weights[f"BWM – {r['nom']}"] = res["weights"]
                st.caption(f"$\\xi^* = {res['xi']:.3f}$ (proche de 0 = jugements cohérents)")
            except ValueError as e:
                st.error(str(e))
    if bwm_w:
        weights["BWM – groupe"] = np.mean(bwm_w, axis=0)

    st.subheader("Méthodes objectives")
    ent, crt = WG.entropy(Xv, TYPES), WG.critic(Xv, TYPES)
    weights["Entropie"], weights["CRITIC"] = ent["weights"], crt["weights"]
    formula(r"\text{Entropie : } p_{ij} = \frac{r_{ij}}{\sum_i r_{ij}},\quad E_j = -\frac{1}{\ln m}\sum_{i=1}^{m} p_{ij}\ln p_{ij},"
            r"\quad " + K(r"w_j") + r" = \frac{1 - E_j}{\sum_k (1 - E_k)}")
    formula(r"\text{CRITIC : } C_j = \sigma_j \sum_{k=1}^{n}\left(1 - |\rho_{jk}|\right),"
            r"\quad " + K(r"w_j") + r" = \frac{C_j}{\sum_k C_k}")
    with st.expander("Détails des calculs (entropie et CRITIC)"):
        st.dataframe(pd.DataFrame({"Entropie Eⱼ": ent["E"], "1 − Eⱼ": ent["diversification"], "wⱼ entropie": ent["weights"],
                                   "σⱼ (CRITIC)": crt["sigma"], "Cⱼ": crt["C"], "wⱼ CRITIC": crt["weights"]},
                                  index=LABELS).style.format("{:.3f}"), width="stretch")
        st.plotly_chart(heatmap(crt["rho"], CODES, "Corrélations entre critères (CRITIC)", DIVERGING, -1, 1),
                        width="stretch")

    st.subheader("Poids hybrides")
    alpha = st.slider("α : part des poids subjectifs (BWM groupe) face aux poids objectifs (CRITIC)", 0.0, 1.0, 0.7, 0.05)
    formula(K(r"w_j^{\,hyb}") + r" = \alpha\, w_j^{\,BWM} + (1 - \alpha)\, w_j^{\,CRITIC},"
            r"\qquad \alpha = " + f"{alpha:.2f}".replace(".", "{,}"))
    if "BWM – groupe" in weights:
        hyb = alpha * weights["BWM – groupe"] + (1 - alpha) * weights["CRITIC"]
        weights["Hybride"] = hyb / hyb.sum()
    weights["Poids égaux"] = np.full(len(CODES), 1 / len(CODES))

    st.subheader("Comparaison des poids")
    show = [k for k in ["AHP – " + r_ahp, "BWM – groupe", "Entropie", "CRITIC", "Hybride"] if k in weights]
    st.plotly_chart(weights_chart({k: weights[k] for k in show}), width="stretch")
    st.dataframe(pd.DataFrame(weights, index=LABELS).T.style.format("{:.3f}"), width="stretch")
    default = list(weights).index("Hybride") if "Hybride" in weights else 0
    ss.w_name = st.selectbox("Poids retenus pour le classement", list(weights), index=default)

W = np.asarray(weights[ss.w_name], dtype=float)
ss.weights_all, ss.W = weights, W

# ===================================================================== 4. CLASSEMENT
with tabs[4]:
    st.header("Classement des candidats")
    st.caption(f"Poids utilisés : **{ss.w_name}**")
    with st.expander("Paramètres des méthodes"):
        p1, p2, p3 = st.columns(3)
        lam = p1.slider("WASPAS : λ", 0.0, 1.0, 0.5, 0.05)
        v_vik = p1.slider("VIKOR : v (poids de l'utilité de groupe)", 0.0, 1.0, 0.5, 0.05)
        pref = p2.selectbox("PROMETHEE : fonction de préférence", ["linéaire", "en V", "usuelle"])
        pq = p2.slider("PROMETHEE : q (indifférence)", 0.0, 0.5, 0.05, 0.01)
        pp = p2.slider("PROMETHEE : p (préférence stricte)", 0.01, 1.0, 0.30, 0.01)
        eq = p3.slider("ELECTRE III : q", 0.0, 0.5, 0.05, 0.01)
        ep = p3.slider("ELECTRE III : p", 0.01, 0.9, 0.20, 0.01)
        ev = p3.slider("ELECTRE III : v (veto)", 0.05, 1.0, 0.50, 0.01)
        st.caption("Pour PROMETHEE et ELECTRE, les seuils s'appliquent aux valeurs normalisées sur [0, 1].")

    fuzzy_F = FZ.crisp_to_tfn(Xv)
    fuzzy_F[:, CODES.index("C6"), :] = soft_tfn()
    fuzzy_W = np.array([FZ.aggregate([FZ.IMPORTANCE[r["importance"][j]] for r in ss.recruiters]) for j in range(len(CODES))])

    def run_all(w):
        return {
            "WSM": RK.wsm(Xv, TYPES, w), "WPM": RK.wpm(Xv, TYPES, w), "WASPAS": RK.waspas(Xv, TYPES, w, lam),
            "TOPSIS": RK.topsis(Xv, TYPES, w), "VIKOR": RK.vikor(Xv, TYPES, w, v_vik),
            "PROMETHEE II": RK.promethee_ii(Xv, TYPES, w, pref, pq, pp),
            "ELECTRE III": RK.electre_iii(Xv, TYPES, w, eq, ep, ev),
        }

    results = run_all(W)
    results["TOPSIS flou"] = FZ.fuzzy_topsis(fuzzy_F, fuzzy_W, TYPES)
    ss.method_fns = {
        "TOPSIS": lambda w: RK.topsis(Xv, TYPES, w)["rank"],
        "WSM": lambda w: RK.wsm(Xv, TYPES, w)["rank"],
        "VIKOR": lambda w: RK.vikor(Xv, TYPES, w, v_vik)["rank"],
        "PROMETHEE II": lambda w: RK.promethee_ii(Xv, TYPES, w, pref, pq, pp)["rank"],
        "ELECTRE III": lambda w: RK.electre_iii(Xv, TYPES, w, eq, ep, ev)["rank"],
    }
    ranks = pd.DataFrame({k: v["rank"] for k, v in results.items()}, index=NAMES)
    ranks["Rang moyen"] = ranks.mean(axis=1)
    ranks["Consensus"] = RK.ranks_from_scores(ranks["Rang moyen"], higher_is_better=False)
    ranks = ranks.sort_values("Rang moyen")
    ss.ranks = ranks
    ss.results = results

    top = ranks.index[0]
    vk = results["VIKOR"]
    n_first = int((ranks.loc[top, list(results)] == 1).sum())
    worst = float(ranks["Rang moyen"].max())
    rows = ""
    for pos, (nm, rw) in enumerate(ranks.iterrows(), start=1):
        share = 100 * (worst + 1 - rw["Rang moyen"]) / worst
        _cls = " lb-top lb-good" if pos == 1 else (" lb-good" if pos <= 3 else (" lb-bad" if pos > len(ranks) - 3 else ""))
        rows += (f'<div class="lb-row{_cls}" style="animation-delay:{0.04 * pos:.2f}s"><span class="lb-rank">{int(rw["Consensus"])}</span>'
                 f'<span class="lb-name">{nm}</span><span class="lb-bar"><i style="width:{share:.0f}%;animation-delay:{0.15 + 0.05 * pos:.2f}s"></i></span>'
                 f'<span class="lb-val">{rw["Rang moyen"]:.1f}</span></div>')
    lc, rc = st.columns([3, 2])
    lc.markdown('<div class="card lb"><div class="card-label">Classement consensus <span class="lb-note">rang moyen sur '
                f'{len(results)} méthodes, plus petit = meilleur</span></div>{rows}'
                '<div class="lb-legend"><span><i class="g"></i>Tête du classement</span><span><i class="v"></i>Milieu</span>'
                '<span><i class="b"></i>Fin du classement</span></div></div>', unsafe_allow_html=True)
    rc.markdown('<div class="card card-top"><div class="card-label">Candidat recommandé</div>'
                f'<div class="card-value">{top}</div></div>'
                '<div class="card"><div class="card-label">Méthodes qui le placent 1er</div>'
                f'<div class="card-value">{n_first} <span class="card-unit">sur {len(results)}</span></div></div>'
                '<div class="card"><div class="card-label">Compromis VIKOR</div><div class="card-text">'
                + ", ".join(NAMES[i] for i in vk["compromise"]) + '</div></div>', unsafe_allow_html=True)

    st.subheader("Rangs par méthode (1 = meilleur)")
    meth = list(results)
    st.plotly_chart(heatmap(ranks[meth].to_numpy(), meth, "", RANK_SCALE,
                            1, len(NAMES), fmt="d", ylabels=list(ranks.index)), width="stretch")
    st.caption("Méthodes additives (WSM, WPM, WASPAS, TOPSIS, VIKOR) : un bon score compense un mauvais. "
               "Surclassement (PROMETHEE II, ELECTRE III) : la compensation est limitée, une faiblesse forte pèse "
               "(veto dans ELECTRE). TOPSIS flou : intègre directement l'incertitude des avis linguistiques.")
    with st.expander("Formules des méthodes de classement"):
        formula(r"\begin{gathered}\text{WSM : } " + K(r"S_i") + r" = \sum_j w_j\, r_{ij}\qquad \text{WPM : } " + K(r"P_i")
                + r" = \prod_j r_{ij}^{\,w_j} \\[6pt] \text{WASPAS : } " + K(r"Q_i") + r" = \lambda S_i + (1-\lambda) P_i\end{gathered}")
        formula(r"\text{TOPSIS : } S_i^{\pm} = \sqrt{\sum_j \left(v_{ij} - v_j^{\pm}\right)^2},\qquad "
                + K(r"CC_i") + r" = \frac{S_i^-}{S_i^+ + S_i^-}")
        formula(r"\text{VIKOR : } " + K(r"Q_i") + r" = v\,\frac{S_i - S^*}{S^- - S^*} + (1 - v)\,\frac{R_i - R^*}{R^- - R^*}"
                r"\quad (\text{plus petit} = \text{meilleur})")
        formula(r"\text{PROMETHEE II : } " + K(r"\varphi(a)") + r" = \varphi^+(a) - \varphi^-(a),\qquad "
                r"\varphi^+(a) = \frac{1}{m-1}\sum_{b \ne a}\pi(a,b)")
    st.markdown(f"**Conditions VIKOR.** Avantage acceptable (C1) : **{'vérifiée' if vk['C1_avantage'] else 'non vérifiée'}**. "
                f"Stabilité (C2) : **{'vérifiée' if vk['C2_stabilite'] else 'non vérifiée'}**.")

    st.subheader("Détail d'une méthode")
    mname = st.selectbox("Méthode", meth, index=meth.index("TOPSIS"))
    r = results[mname]
    detail = {"Score": r["score"], "Rang": r["rank"]}
    extra = {"TOPSIS": ("S_plus", "S_minus"), "VIKOR": ("S", "R", "Q"), "PROMETHEE II": ("phi_plus", "phi_minus"),
             "WASPAS": ("Q_wsm", "Q_wpm"), "TOPSIS flou": ("D_plus", "D_minus")}
    for key in extra.get(mname, ()):
        detail[key] = r[key]
    st.dataframe(pd.DataFrame(detail, index=pd.Index(NAMES, name="Candidat")).sort_values("Rang").reset_index().style.format(
        {k: "{:.4f}" for k in detail if k != "Rang"}), width="stretch", hide_index=True)
    st.download_button("Exporter les rangs (CSV)", ranks.to_csv().encode("utf-8"), "classement.csv", "text/csv")

# ============================================================ 5. ÉVALUATION ABSOLUE
with tabs[5]:
    st.header("Évaluation absolue d'un candidat (MAUT)")
    st.caption("Les candidats ne sont pas comparés entre eux : chacun est jugé seul, face à des "
               "bornes de référence fixées par l'entreprise.")
    formula(r"\begin{gathered}" + K(r"S(A)") + r" = \sum_{j=1}^{n} w_j\, u_j(x_j)\qquad \text{accepté si } S(A) \ge \tau \\[6pt]"
            r"u_j(x) = \frac{x - \min_j}{\max_j - \min_j}\ \ (\text{à maximiser}),\qquad "
            r"u_j(x) = \frac{\max_j - x}{\max_j - \min_j}\ \ (\text{coût})\end{gathered}")
    b = pd.DataFrame(ss.bounds, index=["min", "max"]).T.loc[CODES]
    b = st.data_editor(b, width="stretch", key="bounds_editor")
    ss.bounds = {c: [b.loc[c, "min"], b.loc[c, "max"]] for c in CODES}
    lo, hi = b["min"].to_numpy(float), b["max"].to_numpy(float)
    S_all = np.array([AN.maut(Xv[i], TYPES, W, lo, hi)["score"] for i in range(len(NAMES))])

    ss.setdefault("tau", 0.60)
    st.markdown(r"Le seuil $\tau$ décide qui est retenu : un candidat est accepté si son score $S \ge \tau$. "
                "On le fixe à la main (politique de l'entreprise) ou on le déduit de décisions passées.")
    tau = st.slider("Seuil τ", 0.0, 1.0, step=0.01, key="tau")
    accepted = S_all >= tau
    _na = int(accepted.sum())
    st.markdown(f"**Avec τ = {tau:.2f}, {_na} candidat{'s' if _na > 1 else ''} sur {len(NAMES)} {'sont acceptés' if _na > 1 else 'est accepté'}.**")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Évaluation d'un candidat")
        who = st.selectbox("Candidat évalué", NAMES, key="maut_who")
        i = NAMES.index(who)
        res = AN.maut(Xv[i], TYPES, W, lo, hi)
        ok = res["score"] >= tau
        st.markdown(f'<div class="card {"card-ok" if ok else "card-ko"}"><div class="card-label">Score MAUT de {who}</div>'
                    f'<div class="card-value">{res["score"]:.3f}</div>'
                    f'<div class="card-text">{"Accepter : invité en entretien" if ok else "Rejeter"} '
                    f'(S {"≥" if ok else "<"} τ = {tau:.2f})</div></div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=res["utilities"], y=LABELS, orientation="h", marker_color=[TH.get("good", SERIES[2]) if u >= .6 else (TH.get("bad", SERIES[1]) if u < .4 else TH["accent"]) for u in res["utilities"]],
                               hovertemplate="%{y} : u = %{x:.2f}<extra></extra>"))
        fig.update_layout(title="Utilités u_j (0 = borne la pire, 1 = la meilleure)", xaxis_range=[0, 1])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 320), width="stretch")
    with c2:
        st.subheader("Seuil issu de décisions passées")
        st.caption("Colonne « Décision passée » : 1 = candidat retenu à l'époque (valeurs fictives de démonstration, "
                   "à remplacer). Elle sert à calculer le seuil optimal ; elle ne dépend pas de τ. "
                   "La colonne « Décision selon τ » suit, elle, le curseur ci-dessus.")
        table = pd.DataFrame({"Score S": S_all,
                              "Décision passée": ss.past.reindex(NAMES).fillna(0).astype(int),
                              "Décision selon τ": np.where(accepted, "Accepter", "Rejeter")}, index=NAMES)
        past = st.data_editor(table, disabled=["Score S", "Décision selon τ"], width="stretch", key="past_editor",
                              column_config={"Score S": st.column_config.NumberColumn(format="%.3f"),
                                             "Décision passée": st.column_config.SelectboxColumn(options=[0, 1])})
        ss.past = past["Décision passée"]
        y = past["Décision passée"].to_numpy(int)
        if 0 < y.sum() < len(y):
            taus = np.unique(np.r_[0, S_all, 1])
            tpr = np.array([((S_all >= t) & (y == 1)).sum() / (y == 1).sum() for t in taus])
            fpr = np.array([((S_all >= t) & (y == 0)).sum() / (y == 0).sum() for t in taus])
            J = tpr - fpr
            d = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
            kb = int(np.argmax(J))
            tpr_t = ((S_all >= tau) & (y == 1)).sum() / (y == 1).sum()
            fpr_t = ((S_all >= tau) & (y == 0)).sum() / (y == 0).sum()
            fig = go.Figure()
            fig.add_scatter(x=fpr, y=tpr, mode="lines+markers", line=dict(color=SERIES[0], width=2, shape="hv"),
                            marker=dict(size=8), name="Courbe ROC",
                            text=[f"τ = {t:.3f}" for t in taus], hovertemplate="%{text}<br>FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>")
            fig.add_scatter(x=[fpr[kb]], y=[tpr[kb]], mode="markers", marker=dict(size=15, color=SERIES[1],
                            line=dict(color="white", width=2)), name=f"Optimum de Youden (τ = {taus[kb]:.3f})")
            fig.add_scatter(x=[fpr_t], y=[tpr_t], mode="markers", marker=dict(size=11, color=TH["ink"], symbol="diamond",
                            line=dict(color="white", width=2)), name=f"τ choisi ({tau:.2f})")
            fig.update_layout(xaxis_title="FPR (mauvais candidats acceptés)", yaxis_title="TPR (bons candidats acceptés)",
                              xaxis_range=[-0.02, 1.02], yaxis_range=[-0.02, 1.05])
            st.plotly_chart(style(fig, 380), width="stretch")
            m1, m2, m3 = st.columns(3)
            m1.metric("TPR à τ choisi", f"{tpr_t:.2f}")
            m2.metric("FPR à τ choisi", f"{fpr_t:.2f}")
            m3.metric("Indice J à τ choisi", f"{tpr_t - fpr_t:.2f}")
            formula(K(r"J(\tau)") + r" = TPR(\tau) - FPR(\tau),\qquad \tau^* = \arg\max_\tau J(\tau)")
            st.markdown(f"Seuil optimal selon Youden : **τ = {taus[kb]:.3f}**, avec J = {J[kb]:.2f} "
                        f"et une distance au point idéal de {d[kb]:.2f}.")
            opt = float(np.floor(taus[kb] * 100) / 100)
            st.button(f"Appliquer le seuil optimal (τ = {opt:.2f})",
                      on_click=lambda v=opt: ss.__setitem__("tau", v))
        else:
            st.info("Il faut au moins un candidat retenu et un candidat rejeté.")

# ========================================================== 6. ROBUSTESSE & ÉTHIQUE
with tabs[6]:
    st.header("Robustesse, fiabilité et équité")
    meth = [m for m in ss.ranks.columns if m not in ("Rang moyen", "Consensus")]
    ranks = ss.ranks.loc[NAMES]

    st.subheader("1. Les méthodes sont-elles d'accord ? Corrélation de Spearman")
    M, labels = AN.spearman_matrix({m: ranks[m].to_numpy() for m in meth})
    crit = AN.spearman_critical(len(NAMES))
    _lo = float(np.floor(np.nanmin(M) * 20) / 20)
    st.plotly_chart(heatmap(M, labels, "", SEQ_BLUE, _lo, 1), width="stretch")
    st.caption(f"L'échelle de couleur va de {_lo:.2f} à 1 pour faire ressortir les écarts entre méthodes.")
    formula(K(r"\rho") + r" = 1 - \frac{6\sum_i d_i^2}{n\,(n^2 - 1)},\qquad d_i = \text{écart de rang du candidat } i")
    st.caption(f"Avec n = {len(NAMES)} candidats, ρ est significatif au seuil de 5 % si ρ ≥ {crit:.3f}. "
               f"Paires sous ce seuil : " + (", ".join(f"{labels[i]} / {labels[j]} ({M[i, j]:.2f})"
               for i in range(len(labels)) for j in range(i + 1, len(labels)) if M[i, j] < crit) or "aucune") + ".")

    st.subheader("2. Le classement résiste-t-il aux changements de poids ?")
    s1, s2 = st.columns(2)
    m_sens = s1.selectbox("Méthode", list(ss.method_fns), key="m_sens")
    c_sens = s2.selectbox("Critère dont on fait varier le poids", LABELS, key="c_sens")
    j = LABELS.index(c_sens)
    grid, R = AN.one_at_a_time(ss.method_fns[m_sens], W, j)
    top5 = list(ss.ranks.index[:5])
    fig = go.Figure()
    for k, n in enumerate(NAMES):
        is_top = n in top5
        fig.add_scatter(x=grid, y=R[:, k], mode="lines", name=n, line=dict(
            width=2.5 if is_top else 1.2, color=SERIES[top5.index(n)] if is_top else GRAY, shape="hv"),
            showlegend=is_top, hovertemplate=f"{n}<br>poids %{{x:.2f}} → rang %{{y}}<extra></extra>")
    fig.add_vline(x=W[j], line_dash="dot", line_color=TH["ink2"], annotation_text="poids actuel", annotation_position="bottom right")
    fig.update_yaxes(autorange="reversed", title="Rang", dtick=1)
    fig.update_xaxes(title=f"Poids de {CODES[j]} (les autres gardent leurs proportions)")
    st.plotly_chart(style(fig, 420), width="stretch")
    st.caption("En couleur, les cinq premiers du consensus. En gris, les autres candidats.")

    st.markdown("**Simulation de Monte-Carlo.** 500 tirages de poids aléatoires autour des poids retenus.")
    conc = st.slider("Concentration (plus elle est faible, plus les poids s'écartent des poids retenus)", 5, 200, 50, 5)
    MC = AN.monte_carlo(ss.method_fns[m_sens], W, 500, conc)
    mc = pd.DataFrame({"% classé 1er": (MC == 1).mean(axis=0) * 100, "% dans le top 3": (MC <= 3).mean(axis=0) * 100},
                      index=NAMES).sort_values("% dans le top 3", ascending=False)
    fig = go.Figure()
    fig.add_bar(name="% dans le top 3", x=mc.index, y=mc["% dans le top 3"], marker_color=SERIES[0])
    fig.add_bar(name="% classé 1er", x=mc.index, y=mc["% classé 1er"], marker_color=SERIES[1])
    fig.update_layout(barmode="group", yaxis_title="% des tirages", yaxis_range=[0, 100])
    st.plotly_chart(style(fig, 360), width="stretch")

    st.subheader("3. Les scores du LLM sont-ils fiables ? Comparaison avec des notes humaines")
    st.caption("Faites noter les CV par une à trois personnes avec la même grille, puis importez le fichier.")
    tpl = pd.DataFrame(index=NAMES, columns=[c["code"] for c in LLM_CRIT])
    tpl.index.name = "candidat"
    st.download_button("Télécharger le modèle CSV", tpl.to_csv().encode("utf-8"), "notes_humaines.csv", "text/csv")
    hum = st.file_uploader("Notes humaines (CSV)", type="csv", key="human")
    if hum:
        H = pd.read_csv(hum, index_col=0).reindex(NAMES)
        rows = []
        from scipy.stats import spearmanr
        for c in LLM_CRIT:
            if c["code"] in H and H[c["code"]].notna().sum() >= 3:
                a, h = ss.scores[c["code"]], H[c["code"]]
                ok = h.notna()
                rows.append({"Critère": f'{c["code"]} {c["nom"]}', "Écart moyen absolu (MAE)": (a[ok] - h[ok]).abs().mean(),
                             "Spearman LLM / humain": spearmanr(a[ok], h[ok]).statistic, "n": int(ok.sum())})
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format({"Écart moyen absolu (MAE)": "{:.2f}", "Spearman LLM / humain": "{:.2f}"}),
                         width="stretch", hide_index=True)

    st.subheader("4. Le LLM est-il biaisé ? Test du prénom")
    st.caption("On soumet exactement le même CV en changeant seulement le nom. Un système équitable doit donner "
               "les mêmes notes et le même rang.")
    b1, b2 = st.columns([1, 2])
    cand = b1.selectbox("CV de référence", NAMES, key="bias_cand")
    variants = b2.text_area("Noms à tester (un par ligne)", "Sara El Amrani\nThomas Martin\nFatima Zahra Bennis\nJean-Baptiste Dupont")
    if mode != "LLM réel":
        st.info("Ce test appelle le LLM : choisissez « LLM réel » dans la barre latérale pour l'exécuter.")
    elif st.button("Lancer le test de biais"):
        rows = []
        names_to_test = [cand] + [v.strip() for v in variants.splitlines() if v.strip()]
        bar = st.progress(0.0)
        for k, nm in enumerate(names_to_test):
            text = ss.cvs[cand] if nm == cand else LLM.swap_name(ss.cvs[cand], cand, nm)
            try:
                res = LLM.score_cv(text, ss.job, LLM_CRIT, base_url=llm_cfg["base_url"], api_key=llm_cfg["api_key"],
                                   model=llm_cfg["model"], n_runs=llm_cfg["n_runs"])
                Xb = Xv.copy()
                i = NAMES.index(cand)
                for c in LLM_CRIT:
                    Xb[i, CODES.index(c["code"])] = res[c["code"]]["score"]
                rank = RK.topsis(Xb, TYPES, W)["rank"][i]
                rows.append({"Nom testé": nm, **{c["code"]: res[c["code"]]["score"] for c in LLM_CRIT}, "Rang TOPSIS": int(rank)})
            except Exception as e:  # noqa: BLE001
                st.error(f"{nm} : {e}")
            bar.progress((k + 1) / len(names_to_test))
        if rows:
            df = pd.DataFrame(rows).set_index("Nom testé")
            st.dataframe(df.style.format("{:.2f}", subset=[c["code"] for c in LLM_CRIT]), width="stretch")
            spread = (df.drop(columns="Rang TOPSIS").max() - df.drop(columns="Rang TOPSIS").min()).max()
            if df["Rang TOPSIS"].nunique() == 1 and spread < 0.5:
                st.success("Aucun effet du nom détecté sur ce CV.")
            else:
                st.warning(f"Effet du nom détecté : écart maximal de {spread:.2f} point, "
                           f"rangs entre {df['Rang TOPSIS'].min()} et {df['Rang TOPSIS'].max()}.")


# ================================================================== TABLEAU DE BORD
# Rempli en dernier : il reprend les résultats calculés dans les autres onglets.
ICON = {
    "users": '<path d="M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM2 21v-1a6 6 0 0 1 6-6h2a6 6 0 0 1 6 6v1M16 3.1a4 4 0 0 1 0 7.8M22 21v-1a6 6 0 0 0-4-5.6"/>',
    "check": '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M8 12l3 3 5-6"/>',
    "people": '<circle cx="12" cy="7" r="4"/><path d="M5 21v-2a7 7 0 0 1 14 0v2"/>',
    "trophy": '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0zM7 6H4a3 3 0 0 0 3 4M17 6h3a3 3 0 0 1-3 4"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
    "file": '<path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4M9 12h6M9 16h6"/>',
    "scale": '<path d="M12 3v18M5 21h14M5 7h14M5 7l-3 7a3 3 0 0 0 6 0zM19 7l-3 7a3 3 0 0 0 6 0z"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "done": '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
}


def icon(name, size=26):
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICON[name]}</svg>')


HERO_ART = """
<svg class="hero-art tr-illustration" viewBox="0 0 520 310" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Illustration de recrutement intelligent : profils de candidats et sélection par intelligence artificielle">
<defs>
 <linearGradient id="trgrad" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#9C6DFF"/><stop offset="1" stop-color="#5532D8"/></linearGradient>
 <linearGradient id="trglass" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#FFFFFF"/><stop offset="1" stop-color="#EEE9FF"/></linearGradient>
 <filter id="trshadow" x="-30%" y="-30%" width="170%" height="180%"><feDropShadow dx="0" dy="12" stdDeviation="12" flood-color="#3F2C98" flood-opacity=".17"/></filter>
</defs>
<circle cx="269" cy="150" r="142" fill="#DED4FF" opacity=".38"/>
<circle cx="399" cy="54" r="9" fill="#C5B3FF" class="tr-float"/>
<circle cx="81" cy="229" r="6" fill="#BDA9FF" class="tr-float"/>
<g class="tr-float-slow" filter="url(#trshadow)" transform="translate(46 72) rotate(-10 72 88)">
<rect width="143" height="178" rx="18" fill="url(#trglass)" stroke="#D9CEFA"/>
<circle cx="72" cy="54" r="26" fill="#E4DAFF"/><circle cx="72" cy="47" r="11" fill="#8F79D8"/><path d="M48 73c6-20 42-20 48 0" fill="#8F79D8"/>
<rect x="30" y="100" width="83" height="9" rx="4.5" fill="#B9ACEB"/><rect x="30" y="117" width="61" height="7" rx="3.5" fill="#DDD5F6"/>
<rect x="30" y="137" width="68" height="7" rx="3.5" fill="#DDD5F6"/>
</g>
<g class="tr-float" filter="url(#trshadow)" transform="translate(315 52) rotate(9 75 90)">
<rect width="147" height="183" rx="18" fill="url(#trglass)" stroke="#D9CEFA"/>
<circle cx="73" cy="53" r="26" fill="#E4DAFF"/><circle cx="73" cy="46" r="11" fill="#8D72E4"/><path d="M49 72c6-20 42-20 48 0" fill="#8D72E4"/>
<rect x="31" y="99" width="85" height="9" rx="4.5" fill="#B9ACEB"/><rect x="31" y="116" width="64" height="7" rx="3.5" fill="#DDD5F6"/>
<rect x="31" y="136" width="73" height="7" rx="3.5" fill="#DDD5F6"/>
</g>
<g filter="url(#trshadow)" transform="translate(177 19)">
<rect width="174" height="237" rx="23" fill="white" stroke="#CEC1F6" stroke-width="1.5"/>
<rect x="14" y="14" width="146" height="40" rx="12" fill="#F2EEFF"/>
<text x="29" y="39" fill="#6B45E3" font-family="Arial,sans-serif" font-size="13" font-weight="700">Candidat n°1</text>
<circle cx="87" cy="105" r="39" fill="#E8E0FF"/><path d="M61 132q26-40 52 0" fill="#6842D4"/><circle cx="87" cy="96" r="16" fill="#A78168"/><path d="M71 93q0-25 22-22 16 2 11 23-10-12-33-1" fill="#242048"/>
<rect x="33" y="158" width="108" height="10" rx="5" fill="#DDD4FA"/>
<rect x="47" y="177" width="80" height="8" rx="4" fill="#E9E4F7"/>
<rect x="33" y="200" width="108" height="10" rx="5" fill="#2FA36B"/><rect x="33" y="218" width="76" height="8" rx="4" fill="#B9ACEB"/>
</g>
<g class="tr-check" filter="url(#trshadow)"><circle cx="348" cy="247" r="32" fill="url(#trgrad)"/><path d="m334 247 10 10 19-21" fill="none" stroke="white" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/></g>
<g class="tr-float-slow" transform="translate(77 20)"><rect width="92" height="36" rx="13" fill="#FFFFFF" stroke="#E1D9FF"/><circle cx="19" cy="18" r="7" fill="#7653E6"/><text x="34" y="23" fill="#5E42B3" font-family="Arial" font-size="13" font-weight="700">CV + LLM</text></g>
<g class="tr-float" transform="translate(387 259)"><rect width="108" height="32" rx="13" fill="#FFFFFF" stroke="#E1D9FF"/><text x="13" y="21" fill="#5E42B3" font-family="Arial" font-size="12" font-weight="700">8 méthodes</text></g>
</svg>
"""

with tabs[0]:
    topsis = results["TOPSIS"]
    order_c = list(ss.ranks.index)                       # ordre du consensus
    best = order_c[0]
    bi = NAMES.index(best)
    mc_top = AN.monte_carlo(ss.method_fns["TOPSIS"], W, 300, 50)
    robust = float((mc_top[:, bi] == 1).mean() * 100)

    st.markdown(
        '<div class="dash-hero"><div class="dash-hero-text">'
        '<div class="dash-eyebrow"><span class="tr-live-dot"></span>Projet MCDM · EMI Rabat</div>'
        '<div class="dash-title">Le bon talent.<br><span>La bonne décision.</span></div>'
        '<p>Un LLM lit les CV et note chaque critère, puis huit méthodes d\'aide à la décision multicritère '
        'pondèrent, classent et vérifient la robustesse du choix. La décision finale reste humaine.</p>'
        f'<div class="dash-chips"><span>Poste : <b>{ss.job["titre"]}</b></span><span>Poids : <b>{ss.w_name}</b></span>'
        f'<span>Scores : <b>{ss.source_scores}</b></span></div>'
        '</div>' + HERO_ART + '</div>', unsafe_allow_html=True)

    kpis = [("users", len(NAMES), "Candidats"), ("check", len(CODES), "Critères"),
            ("people", len(ss.recruiters), "Décideurs"), ("trophy", len(results), "Méthodes MCDM"),
            ("shield", f"{robust:.0f} %", "Stabilité du n°1")]
    st.markdown('<div class="kpi-row">' + "".join(
        f'<div class="kpi"><span class="kpi-ic">{icon(ic)}</span><div><b>{v}</b><small>{lab}</small></div></div>'
        for ic, v, lab in kpis) + '</div>', unsafe_allow_html=True)

    cL, cR = st.columns([1, 1], gap="medium")
    SHORT = {"C1": "Compétences", "C2": "Expérience", "C3": "Formation", "C4": "Projets", "C5": "Langues",
             "C6": "Soft skills", "C7": "Salaire"}
    top5 = order_c[:5]
    medal = {1: "m1", 2: "m2", 3: "m3"}
    rows = "".join(
        f'<tr><td><span class="medal {medal.get(k, "")}">{k}</span></td><td class="nm">{n}</td>'
        f'<td class="sc">{topsis["score"][NAMES.index(n)]:.3f}</td>'
        f'<td><span class="bar"><i style="width:{topsis["score"][NAMES.index(n)] / topsis["score"].max() * 100:.0f}%;animation-delay:{.1 * k:.1f}s"></i></span></td></tr>'
        for k, n in enumerate(top5, start=1))
    cL.markdown(f'<div class="panel"><div class="panel-h">{icon("trophy", 22)} Classement global</div>'
                '<table class="rank-t"><thead><tr><th>#</th><th>Candidat</th><th>TOPSIS</th><th></th></tr></thead>'
                f'<tbody>{rows}</tbody></table><div class="panel-f">Classement complet dans l\'onglet 4 Classement</div></div>',
                unsafe_allow_html=True)

    xb = X.loc[best]
    initials = "".join(p[0] for p in best.split()[:2]).upper()
    lo = np.array([ss.bounds[c][0] for c in CODES], dtype=float)
    hi = np.array([ss.bounds[c][1] for c in CODES], dtype=float)
    util = AN.maut(xb[CODES].to_numpy(dtype=float), TYPES, W, lo, hi)["utilities"] * 10
    tiles = "".join(
        f'<div class="tile"><small>{SHORT.get(c["code"], c["code"])}</small><b>{xb[c["code"]] / 1000:.0f}k</b><em>MAD</em></div>'
        if c.get("numerique") else
        f'<div class="tile"><small>{SHORT.get(c["code"], c["code"])}</small><b>{xb[c["code"]]:.1f}</b><em>/10</em></div>'
        for c in ss.criteria)
    with cR:
        st.markdown(f'<div class="panel"><div class="panel-h">{icon("user", 22)} Profil du candidat recommandé</div>'
                    f'<div class="prof"><span class="avatar">{initials}</span><div><b>{best}</b> <span class="n1">N°1</span>'
                    f'<small>{ss.job["titre"]}</small></div><div class="gscore"><small>Score TOPSIS</small>'
                    f'<b>{topsis["score"][bi]:.3f}</b></div></div><div class="tiles">{tiles}</div></div>',
                    unsafe_allow_html=True)
        short = [SHORT.get(c, c) for c in CODES]
        rad = go.Figure(go.Scatterpolar(r=list(util) + [util[0]], theta=short + [short[0]], fill="toself",
                                        line=dict(color=TH["accent"], width=3), fillcolor="rgba(75,43,201,.22)",
                                        marker=dict(size=8, color=TH["accent"]), name=best,
                                        hovertemplate="%{theta} : %{r:.1f}/10<extra></extra>"))
        rad.update_layout(polar=dict(radialaxis=dict(range=[0, 10], tickfont=dict(size=12), gridcolor=TH["line"]),
                                     angularaxis=dict(tickfont=dict(size=15, color=TH["ink"]), gridcolor=TH["line"],
                                                      rotation=90, direction="clockwise"),
                                     bgcolor="rgba(0,0,0,0)"),
                          title=None)
        rad = style(rad, 340)
        rad.update_layout(margin=dict(l=80, r=80, t=30, b=30), showlegend=False)
    cL2, cR2 = st.columns([1, 1], gap="medium")
    with cL2:
        st.markdown(f'<div class="panel-h radar-h">{icon("chart", 22)} Profil de {best} sur les 7 critères</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(rad, width="stretch")
        st.caption("Utilités sur 10 : plus le point est loin du centre, meilleur est le critère (le salaire est inversé).")

    steps = [("file", "1. CV", "notés par le LLM"), ("scale", "2. Poids", "AHP, BWM, CRITIC"),
             ("chart", "3. Classement", "8 méthodes"), ("shield", "4. Contrôles", "robustesse, équité"),
             ("done", "5. Décision", "seuil MAUT")]
    cR2.markdown('<div class="panel"><div class="panel-h">Le processus TalentRank</div><div class="flow flow-v">' + "".join(
        f'<div class="step"><span class="kpi-ic">{icon(ic)}</span><div><b>{t}</b><small>{d}</small></div></div>'
        for k, (ic, t, d) in enumerate(steps)) + '</div></div>',
        unsafe_allow_html=True)
