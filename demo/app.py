"""Yerel demo: tez modeli ve revize model yan yana.

Çalıştırma (repo kökünden):  streamlit run demo/app.py
Bütün sayılar repodaki model, json ve csv dosyalarından okunur.
"""
import importlib.util
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[1]
THESIS_DIR = ROOT / "models" / "legacy_hybrid"
REVISED_DIR = ROOT / "models"


def import_script(name: str):
    """scripts/ altındaki dosyalar rakamla başladığı için normal import edilemez."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


discover = import_script("04_discover")      # revize model: api/main.py ile aynı kod
prepare = import_script("02_prepare_data")   # V2DB etiketleri için


# --- Tez modelinin özellik çıkarma kodu -------------------------------------
# git show thesis-2026:scripts/04_discover.py içindeki featurize() fonksiyonu.
# Tek fark: çıplak "except:" yerine "except Exception:".
def featurize_thesis(formula, feature_names):
    try:
        comp = Composition(formula)
        frac = comp.get_el_amt_dict()
        total = sum(frac.values())
        feats = {f"elem_{k}": v / total for k, v in frac.items()}
        feats["natoms"] = comp.num_atoms
        return {col: feats.get(col, 0) for col in feature_names}
    except Exception:
        return None


def unknown_elements(reduced: str, feature_names: list) -> list:
    """Modelin eğitimde görmediği elementler. Tez kodu bunları sessizce atardı."""
    elements = Composition(reduced).get_el_amt_dict()
    return sorted(el for el in elements if f"elem_{el}" not in feature_names)


# --- Yükleme (bir kez yapılır, sonra önbellekten gelir) ---------------------
@st.cache_resource
def load_models():
    return {
        "thesis": discover.load_model(THESIS_DIR),
        "revised": discover.load_model(REVISED_DIR),
    }


@st.cache_data
def load_thesis_labels():
    path = ROOT / "data" / "legacy_hybrid" / "combined_dataset.csv"
    df = pd.read_csv(path, usecols=["formula_normalized", "is_magnetic", "source"])
    return df.set_index("formula_normalized")


@st.cache_data(show_spinner="V2DB etiketleri hazırlanıyor (yalnız ilk açılışta)...")
def load_v2db_labels():
    """Her indirgenmiş formül için V2DB'deki farklı etiketler (02_prepare_data.py mantığı)."""
    df = pd.read_csv(ROOT / "data" / "v2db.csv", usecols=["Material", "Material_is_magnetic"])
    df["formula"] = df["Material"].map(prepare.normalize_formula)
    df["label"] = df["Material_is_magnetic"].map(prepare.parse_magnetic_label)
    return df.groupby("formula")["label"].agg(lambda s: sorted(set(s)))


@st.cache_data
def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(relative: str):
    return pd.read_csv(ROOT / relative)


# --- Yardımcılar --------------------------------------------------------------
def score(model, reduced, featurize):
    """(skor, bilinmeyen elementler). Bilinmeyen element varsa skor üretilmez."""
    clf, threshold, meta = model
    names = meta["feature_names"]
    missing = unknown_elements(reduced, names)
    if missing:
        return None, missing
    X = pd.DataFrame([featurize(reduced, names)], columns=names)
    proba = clf.predict_proba(X)[0, discover.positive_class_index(clf)]
    return float(proba), []


def label_text(value: int) -> str:
    return "manyetik" if value == 1 else "manyetik değil"


def thesis_label(reduced, labels):
    if reduced not in labels.index:
        return "veri setinde yok"
    row = labels.loc[reduced]
    return f"{label_text(row['is_magnetic'])} ({row['source']})"


def v2db_label(reduced, labels):
    if reduced not in labels.index:
        return "V2DB'de yok"
    values = labels.loc[reduced]
    if len(values) > 1:
        return "çelişkili etiket, eğitimden çıkarıldı"
    return label_text(values[0])


def elements_of(formula: str) -> set:
    return {el.symbol for el in Composition(formula).elements}


# --- Sayfa -------------------------------------------------------------------
st.set_page_config(page_title="2D manyetik malzeme taraması", layout="wide")
st.title("2D manyetik malzeme taraması")
st.caption(
    "Yalnız kimyasal bileşimden manyetiklik skoru veren iki GradientBoosting modeli. "
    "Skorlar kalibre edilmiş olasılık değildir."
)

models = load_models()
thesis_meta = models["thesis"][2]
revised_meta = models["revised"][2]

tab_predict, tab_candidates, tab_card = st.tabs(["Formül tahmini", "Adaylar", "Model kartı"])

# --- Sekme 1: formül tahmini -------------------------------------------------
with tab_predict:
    st.write(
        f"Tez modeli eşiği: **{models['thesis'][1]:.4f}**, "
        f"revize model eşiği: **{models['revised'][1]:.4f}**. "
        "Skor eşiğe eşit veya büyükse model 'manyetik' der."
    )
    text = st.text_area("Formüller (virgül, boşluk veya satır ile ayırın)", "CrI3, MoS2, MnFeBr3")
    formulas = [f for f in re.split(r"[,\s]+", text) if f]

    thesis_labels = load_thesis_labels()
    v2db_labels = load_v2db_labels()
    rows = []
    for formula in formulas:
        reduced = discover.normalize_formula(formula)
        if reduced is None:
            st.error(f"{formula}: formül okunamadı.")
            continue
        row = {"formül": formula, "indirgenmiş": reduced}
        for key, title, model_name, featurize in [
            ("thesis", "tez", "tez modeli", featurize_thesis),
            ("revised", "revize", "revize model", discover.featurize),
        ]:
            value, missing = score(models[key], reduced, featurize)
            if missing:
                st.warning(
                    f"{formula}: {model_name} {', '.join(missing)} elementini eğitimde "
                    "görmedi. Bu model için skor üretilmedi."
                )
                row[f"{title} skoru"] = None
                row[f"{title} kararı"] = "skor yok"
            else:
                row[f"{title} skoru"] = round(value, 4)
                row[f"{title} kararı"] = (
                    "manyetik" if value >= models[key][1] else "manyetik değil"
                )
        row["tez veri seti etiketi"] = thesis_label(reduced, thesis_labels)
        row["V2DB etiketi"] = v2db_label(reduced, v2db_labels)
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Tez veri seti: data/legacy_hybrid/combined_dataset.csv (C2DB + V2DB). "
        "V2DB etiketi: data/v2db.csv, revize modelin eğitim verisi. "
        "Aynı formül için farklı etiketler varsa revize eğitimde o formül kullanılmadı."
    )

# --- Sekme 2: adaylar --------------------------------------------------------
with tab_candidates:
    st.info(
        "Bunlar doğrulanmamış adaylardır. Hiçbiri DFT ile hesaplanmadı. "
        "Skor yalnız bileşime bakar; yapı ve kararlılık değerlendirilmedi."
    )
    candidates = load_csv("results/legacy/novel_candidates.csv")
    summary = load_json("results/legacy/discovery_summary.json")
    top = candidates[candidates["confidence"] == "Very High"].copy()

    st.write(
        f"Tez taraması: {summary['total_generated']} üretilen formül, "
        f"{summary['known_filtered']} bilinen çıkarıldı, {summary['novel_screened']} skorlandı, "
        f"{summary['magnetic_predicted']} eşik üstü. 'Very High' sınıfında "
        f"**{len(top)}** aday var (discovery_summary.json: {summary['very_high_confidence']}), "
        f"en düşük skor {top['probability'].min():.4f}."
    )

    top["elementler"] = top["formula"].map(lambda f: " ".join(sorted(elements_of(f))))
    all_elements = sorted(set().union(*top["formula"].map(elements_of)))
    chosen = st.multiselect("Element filtresi (seçilenlerin hepsini içeren adaylar)", all_elements)
    order = st.radio("Skora göre sıralama", ["azalan", "artan"], horizontal=True)

    view = top
    for el in chosen:
        view = view[view["formula"].map(lambda f, el=el: el in elements_of(f))]
    view = view.sort_values("probability", ascending=(order == "artan"))

    st.write(f"{len(view)} aday gösteriliyor.")
    st.dataframe(
        view[["formula", "probability", "elementler"]].rename(
            columns={"formula": "formül", "probability": "tez modeli skoru"}
        ),
        hide_index=True,
        width="stretch",
    )

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.hist(view["probability"], bins=20, color="gray", edgecolor="black")
    ax.set_xlabel("tez modeli skoru")
    ax.set_ylabel("aday sayısı")
    st.pyplot(fig, width="content")

    st.subheader("Skorları yeniden üret")
    st.write("Dosyadaki skorlar, tez modeli ve tez sürümündeki özellik kodu ile yeniden hesaplanır.")
    if st.button("Yeniden hesapla"):
        clf, _, meta = models["thesis"]
        names = meta["feature_names"]
        X = pd.DataFrame([featurize_thesis(f, names) for f in candidates["formula"]], columns=names)
        recomputed = clf.predict_proba(X)[:, discover.positive_class_index(clf)]
        diff = np.abs(recomputed - candidates["probability"].to_numpy()).max()
        st.write(f"{len(candidates)} formül yeniden skorlandı. Dosyayla en büyük fark: {diff:.1e}")

# --- Sekme 3: model kartı ----------------------------------------------------
with tab_card:
    st.write(
        "İki modelin test kümeleri farklı veriden geliyor, bu yüzden rakamlar doğrudan "
        "karşılaştırılamaz."
    )
    thesis_metrics = thesis_meta["metrics"]
    revised_metrics = revised_meta["test_metrics"]
    names = {
        "accuracy": "accuracy",
        "roc_auc": "ROC-AUC",
        "precision": "precision",
        "recall": "recall",
        "f1_score": "F1",
    }
    metrics_table = pd.DataFrame(
        {
            "metrik": list(names.values()),
            "tez modeli": [thesis_metrics[k] for k in names],
            "revize model": [revised_metrics[k] for k in names],
        }
    )
    extra = pd.DataFrame(
        {
            "metrik": ["eşik", "özellik sayısı", "kompozisyon sayısı"],
            "tez modeli": [
                models["thesis"][1],
                thesis_meta["n_features"],
                sum(thesis_meta["class_distribution"].values()),
            ],
            "revize model": [
                models["revised"][1],
                revised_meta["n_features"],
                revised_meta["data"]["rows_featurized"],
            ],
        }
    )
    st.dataframe(pd.concat([metrics_table, extra]), hide_index=True)
    st.caption(
        "Kaynak: models/legacy_hybrid/metadata.json ve models/metadata.json. "
        f"Revize modelin eşiği {revised_meta['threshold_selection']['selected_on']} "
        "üzerinde seçildi."
    )

    st.subheader("Karışıklık matrisi (test kümesi)")
    left, right = st.columns(2)
    for column, title, cm in [
        (left, "Tez modeli", thesis_metrics["confusion_matrix"]),
        (right, "Revize model", revised_metrics["confusion_matrix"]),
    ]:
        column.write(title)
        column.dataframe(
            pd.DataFrame(
                [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                index=["gerçek: manyetik değil", "gerçek: manyetik"],
                columns=["tahmin: manyetik değil", "tahmin: manyetik"],
            )
        )

    st.subheader("En önemli 10 özellik")
    left, right = st.columns(2)
    for column, title, path in [
        (left, "Tez modeli", "models/legacy_hybrid/feature_importance.csv"),
        (right, "Revize model", "models/feature_importance.csv"),
    ]:
        column.write(title)
        column.dataframe(
            load_csv(path).sort_values("importance", ascending=False).head(10),
            hide_index=True,
        )

    st.subheader("Leave-element-out (revize model)")
    st.write(
        "Her satırda o elementi içeren bütün kompozisyonlar eğitimden çıkarıldı ve "
        "yalnız onlar test edildi. Model bu elementi hiç görmeden skor üretiyor."
    )
    validation = load_json("results/grouped_validation.json")
    leo = pd.DataFrame.from_dict(validation["leave_element_out"], orient="index")
    leo.index.name = "element"
    st.dataframe(leo.reset_index(), hide_index=True)
    st.caption("Kaynak: results/grouped_validation.json (scripts/05_grouped_validation.py).")
