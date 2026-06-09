# -*- coding: utf-8 -*-

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import Ridge #ml modelleri
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

DATA_FILE  = "LLM_odev2_veriler.xlsx"   

#sütun adları
COL_S      = "Sorunuz"
COL_D      = "CosmosLLM düşünme süreci"
COL_C      = "CosmosLLM cevabı"
COL_SCORE  = "Değerlendirme Puanınız"

# Metin etiket → sayısal puan eşlemesi
SCORE_MAP = {
    "çok iyi":  5,
    "iyi":      4,
    "orta":     3,
    "kötü":     2,
    "çok kötü": 1,
}

RANDOM_SEED   = 42
TEST_SIZE     = 1000
BATCH_SIZE    = 32       
EMBED_CACHE   = Path("embed_cache") #hesaplama tekrarını önlemek için cache yapısı
RESULTS_DIR   = Path("results") #üretilen grafikler için directory

EMBED_CACHE.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
np.random.seed(RANDOM_SEED)


# 5 KONFİGÜRASYON

CONFIGS = {
    "S":    lambda df: df[COL_S].tolist(),
    "D":    lambda df: df[COL_D].tolist(),
    "S+D":  lambda df: (df[COL_S] + " " + df[COL_D]).tolist(),
    "D+C":  lambda df: (df[COL_D] + " " + df[COL_C]).tolist(),
    "S+C":  lambda df: (df[COL_S] + " " + df[COL_C]).tolist(),
}


# 4 EMBEDDİNG MODELLERİ

EMBEDDING_MODELS = {
    "Turkish-E5-Large":  "ytu-ce-cosmos/turkish-e5-large",
    "Jina-v5-Small":     "jinaai/jina-embeddings-v5-text-small",
    "Harrier-0.6B":      "microsoft/harrier-oss-v1-0.6b",
    "Qwen3-Embed-0.6B":  "Qwen/Qwen3-Embedding-0.6B",
}

MODEL_KWARGS = {
    "Turkish-E5-Large": {
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "Jina-v5-Small": {
        "init_kwargs":   {"trust_remote_code": True},
        "encode_kwargs": {"normalize_embeddings": True, "task": "text-matching"},
    },
    "Harrier-0.6B": {
        "init_kwargs":   {"trust_remote_code": True},
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "Qwen3-Embed-0.6B": {
        "init_kwargs":   {"trust_remote_code": True},
        "encode_kwargs": {
            "normalize_embeddings": True,
            "prompt_name": "query",
        },
    },
}

EMBED_ORDER = list(EMBEDDING_MODELS.keys())

# 3 ML algoritmaları tanımlanır ve hiperparametreleri belirlenir.

def build_ml_models():
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)), #L2 penalty for optimized learning
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, #karar ağacı sayısı
            max_depth=None, #
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVR(kernel="rbf", C=1.0, gamma="scale")),
        ]),
    }


# Verilen veri dosyasındaki veriler okunur.

def load_data(path: str) -> pd.DataFrame:
    print("=" * 60)
    print(" VERİ YÜKLEME")
    print("=" * 60)

    df = pd.read_excel(path)
    print(f"Satır: {len(df)}, Sütunlar: {df.columns.tolist()}")

    # Metin sütunları → boş string
    for col in [COL_S, COL_D, COL_C]:
        df[col] = df[col].fillna("").astype(str) # Olası NaN değerleri boş string olarak oku.

    # String tipindeki değerlendirme puanları 1 - 5 arası sayısal puanlara dönüştürülür.
    # Ham etiket: "çok iyi", "iyi", "orta", "kötü", "çok kötü"
    df["puan_ham"] = df[COL_SCORE].astype(str).str.strip().str.lower()
    df["puan"]     = df["puan_ham"].map(SCORE_MAP)

    eslesmeyen = df[df["puan"].isna()]["puan_ham"].unique()
    if len(eslesmeyen) > 0:
        print(f"Eşleşmeyen etiketler (kaldırıldı): {eslesmeyen}")

    before = len(df)
    df = df.dropna(subset=["puan"]).reset_index(drop=True)
    print(f"Geçersiz / kaldırılan satır: {before - len(df)}")
    print(f"\n Puan dağılımı (1=çok kötü, 5=çok iyi):")
    for label, num in sorted(SCORE_MAP.items(), key=lambda x: x[1]):
        n = (df["puan"] == num).sum()
        bar = "█" * (n // 200)
        print(f"  {num} ({label:10s}): {n:5d}  {bar}")

    return df


def split_data(df: pd.DataFrame): # Veri eğitim ve test kümelerine bölünür
    idx = np.random.permutation(len(df))
    test_idx  = idx[:TEST_SIZE]
    train_idx = idx[TEST_SIZE:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)
    print(f"\n  Eğitim: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df

#  EMBEDDİNG HESAPLAMA: Embedding algoritmaları ile metinlerin vektörlere dönüşümü gerçekleştirilir.

def compute_embeddings(texts: list, model_key: str, model_path: str,
                       split_name: str, config: str) -> np.ndarray:
    cache_file = EMBED_CACHE / f"{split_name}_{config}_{model_key}.pkl"

    if cache_file.exists():
        print(f"    [Önbellek] {cache_file.name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers yüklü değil.\n"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu" # cihaz olarak gpu kullanılması hedeflenir  
    kw = MODEL_KWARGS.get(model_key, {})
    init_kw   = kw.get("init_kwargs",   {})
    encode_kw = kw.get("encode_kwargs", {"normalize_embeddings": True})

    print(f"    [{model_key}] Yükleniyor... ({device})")
    model = SentenceTransformer(model_path, device=device, **init_kw)
    if(model_key == "Turkish-E5-Large"):
      model.max_seq_length = 512
    else:
      model.max_seq_length = 1024
    print(f"    [{model_key}] {len(texts):,} metin encode ediliyor...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        **encode_kw,
    )

    del model
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"    Boyut: {embeddings.shape} -> Önbelleğe kaydedildi")
    return embeddings



# DEĞERLENDİRME (Hata hesaplaması)

def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}



# ANA DÖNGÜ — Deney Aşaması

def run_experiments(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    y_train = train_df["puan"].values.astype(float)
    y_test  = test_df["puan"].values.astype(float)

    total_exp = len(CONFIGS) * len(EMBEDDING_MODELS) * len(build_ml_models())
    exp_no = 0
    rows = []

    for cfg_name, cfg_fn in CONFIGS.items():
        train_texts = cfg_fn(train_df)
        test_texts  = cfg_fn(test_df)

        for emb_key, emb_path in EMBEDDING_MODELS.items():
            print(f"\n{'─'*65}")
            print(f"  Konfig: {cfg_name}  |  Embedding: {emb_key}")
            print(f"{'─'*65}")

            X_train = compute_embeddings(
                train_texts, emb_key, emb_path, "train", cfg_name
            )
            X_test = compute_embeddings(
                test_texts, emb_key, emb_path, "test", cfg_name
            )

            for ml_name, ml_model in build_ml_models().items():
                exp_no += 1
                print(
                    f"  [{exp_no:02d}/{total_exp}] ML: {ml_name}  ",
                    end="", flush=True
                )

                ml_model.fit(X_train, y_train)
                tr_met = evaluate(y_train, ml_model.predict(X_train))
                te_met = evaluate(y_test,  ml_model.predict(X_test))

                print(
                    f"Test → RMSE={te_met['RMSE']:.4f}  "
                    f"MAE={te_met['MAE']:.4f}  R²={te_met['R2']:.4f}"
                )

                rows.append({
                    "Config":     cfg_name,
                    "EmbedModel": emb_key,
                    "MLModel":    ml_name,
                    "Train_RMSE": tr_met["RMSE"], 
                    "Train_MAE":  tr_met["MAE"],
                    "Train_R2":   tr_met["R2"],
                    "Test_RMSE":  te_met["RMSE"],
                    "Test_MAE":   te_met["MAE"],
                    "Test_R2":    te_met["R2"],
                })

    return pd.DataFrame(rows)



# GRAFİK AŞAMASI

P_EMBED = {
    "Turkish-E5-Large":  "#2196F3",
    "Jina-v5-Small":     "#FF9800",
    "Harrier-0.6B":      "#E91E63",
    "Qwen3-Embed-0.6B":  "#4CAF50",
}
P_ML  = {"Ridge": "#7E57C2", "RandomForest": "#26A69A", "SVR": "#EF5350"}
P_CFG = {"S": "#5C6BC0", "D": "#29B6F6", "S+D": "#AB47BC", "D+C": "#26A69A", "S+C": "#FF7043"}


def _heatmap(df, ax, row_col, metric, cmap, title):
    pivot = (
        df.pivot_table(values=metric, index=row_col,
                       columns="EmbedModel", aggfunc="mean")
        [EMBED_ORDER]
    )
    sns.heatmap(
        pivot, annot=True, fmt=".4f", ax=ax, cmap=cmap,
        linewidths=0.8, linecolor="white",
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Embedding Modeli")
    ax.set_ylabel(row_col)
    ax.tick_params(axis="x", rotation=20)


def plot_all(df: pd.DataFrame, out_dir: Path):
    sns.set_theme(style="whitegrid", font_scale=1.02)

    # ── Fig 1: Config × Embed heatmaps 
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Test Metrikleri — Konfigurasyon × Embedding Modeli",
                 fontsize=14, fontweight="bold", y=1.02)
    _heatmap(df, axes[0], "Config", "Test_RMSE", "YlOrRd", "Test RMSE ↓")
    _heatmap(df, axes[1], "Config", "Test_MAE",  "YlOrRd", "Test MAE ↓")
    _heatmap(df, axes[2], "Config", "Test_R2",   "YlGn",   "Test R² ↑")
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_heatmap_config_embed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] Fig 1 kaydedildi")

    # ── Fig 2: ML × Embed heatmaps 
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    fig.suptitle("Test Metrikleri — ML Algoritması × Embedding Modeli",
                 fontsize=14, fontweight="bold", y=1.02)
    _heatmap(df, axes[0], "MLModel", "Test_RMSE", "YlOrRd", "Test RMSE ↓")
    _heatmap(df, axes[1], "MLModel", "Test_MAE",  "YlOrRd", "Test MAE ↓")
    _heatmap(df, axes[2], "MLModel", "Test_R2",   "YlGn",   "Test R² ↑")
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_heatmap_ml_embed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] Fig 2 kaydedildi")

    # ── Fig 3: Faktör etkisi (3×3 bar) 
    fig, axes = plt.subplots(3, 3, figsize=(22, 15))
    fig.suptitle("Faktörlerin Test Başarısına Etkisi (altın çerçeve = en iyi)",
                 fontsize=15, fontweight="bold", y=1.01)
    factor_rows = [
        ("Config",     P_CFG,   list(CONFIGS.keys())),
        ("MLModel",    P_ML,    list(P_ML.keys())),
        ("EmbedModel", P_EMBED, EMBED_ORDER),
    ]
    metrics3 = [
        ("Test_RMSE", "Test RMSE", "min"),
        ("Test_MAE",  "Test MAE",  "min"),
        ("Test_R2",   "Test R²",   "max"),
    ]
    for row_i, (factor, palette, _) in enumerate(factor_rows):
        for col_i, (metric, label, best) in enumerate(metrics3):
            ax = axes[row_i][col_i]
            agg = df.groupby(factor)[metric].mean()
            order = agg.sort_values(ascending=(best == "min")).index.tolist()
            colors = [palette[k] for k in order]
            bars = ax.bar(order, agg[order], color=colors,
                          edgecolor="white", linewidth=1.2, zorder=3)
            best_key = agg.idxmin() if best == "min" else agg.idxmax()
            bars[order.index(best_key)].set_edgecolor("gold")
            bars[order.index(best_key)].set_linewidth(3)
            for bar, val in zip(bars, agg[order]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
            ax.set_title(f"{factor} → {label}", fontsize=11, fontweight="bold")
            ax.set_xlabel(factor); ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.35, zorder=0)
            if factor == "EmbedModel":
                plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_bar_factor_effects.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] Fig 3 kaydedildi")

    # ── Fig 4: 60 deneyin tamamı 
    ml_list = list(P_ML.keys())
    fig, ax = plt.subplots(figsize=(24, 7))
    x = 0
    xticks_pos, xticks_lbl = [], []
    for cfg in CONFIGS:
        cfg_start = x
        for emb in EMBED_ORDER:
            for ml in ml_list:
                row = df[(df.Config == cfg) & (df.EmbedModel == emb) & (df.MLModel == ml)]
                val = row["Test_RMSE"].values[0] if len(row) else 0
                ax.bar(x, val, width=0.68, color=P_ML[ml], alpha=0.85,
                       edgecolor="white", linewidth=0.5)
                x += 0.72
            x += 0.06
        xticks_pos.append((cfg_start + x - 0.72 - 0.06) / 2)
        xticks_lbl.append(cfg)
        x += 0.35
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_lbl, fontsize=13, fontweight="bold")
    ax.set_ylabel("Test RMSE", fontsize=12)
    ax.set_title(
        "60 Deneyin Tamamı — Test RMSE  "
        "(Her grup = 1 konfig | Her 3 çubuk = Ridge/RF/SVR)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    patches = [mpatches.Patch(color=P_ML[m], label=m) for m in ml_list]
    ax.legend(handles=patches, title="ML Algoritması", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_all60_experiments.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] Fig 4 kaydedildi")

    # ── Fig 5: Box plots 
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("Test RMSE Dağılımı — Her Faktör", fontsize=14, fontweight="bold")
    factor_palette_pairs = [
        ("Config",     P_CFG),
        ("MLModel",    P_ML),
        ("EmbedModel", P_EMBED),
    ]
    for ax, (factor, palette) in zip(axes, factor_palette_pairs):
        order = df.groupby(factor)["Test_RMSE"].mean().sort_values().index.tolist()
        color_list = {k: palette[k] for k in order}
        sns.boxplot(data=df, x=factor, y="Test_RMSE", order=order,
                    hue=factor, palette=color_list,
                    ax=ax, linewidth=1.5, legend=False)
        sns.stripplot(data=df, x=factor, y="Test_RMSE", order=order,
                      color="black", alpha=0.5, size=4, ax=ax, jitter=True)
        ax.set_title(factor, fontweight="bold", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Test RMSE" if factor == "Config" else "")
        ax.grid(axis="y", alpha=0.35)
        if factor == "EmbedModel":
            plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] Fig 5 kaydedildi")

    # ── Fig 6: Train vs Test scatter 
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("Eğitim vs Test RMSE — Aşırı Öğrenme Analizi",
                 fontsize=14, fontweight="bold")
    for ax, (grp, palette) in zip(axes, [
        ("MLModel",    P_ML),
        ("Config",     P_CFG),
        ("EmbedModel", P_EMBED),
    ]):
        for g, sub in df.groupby(grp):
            ax.scatter(sub["Train_RMSE"], sub["Test_RMSE"],
                       label=g, color=palette[g], s=80, alpha=0.85,
                       edgecolors="white", linewidths=0.5)
        lims = [
            min(df[["Train_RMSE", "Test_RMSE"]].min()) - 0.02,
            max(df[["Train_RMSE", "Test_RMSE"]].max()) + 0.02,
        ]
        ax.plot(lims, lims, "k--", alpha=0.4, lw=1.5, label="y=x (ideal)")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Eğitim RMSE", fontsize=11)
        ax.set_ylabel("Test RMSE", fontsize=11)
        ax.set_title(f"{grp}", fontweight="bold", fontsize=12)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_train_vs_test.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] Fig 6 kaydedildi")

    # ── Fig 7: Top-10 / Bottom-10 
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle("En İyi ve En Kötü 10 Kombinasyon (Test RMSE)",
                 fontsize=14, fontweight="bold")
    for ax, (subset, title, rev) in zip(axes, [
        (df.nsmallest(10, "Test_RMSE").reset_index(drop=True),
         "En İyi 10 (Düşük RMSE)", True),
        (df.nlargest(10, "Test_RMSE").reset_index(drop=True),
         "En Kötü 10 (Yüksek RMSE)", False),
    ]):
        labels = [
            f"{r.Config} | {r.EmbedModel.split('-')[0]} | {r.MLModel}"
            for _, r in subset.iterrows()
        ]
        vals = subset["Test_RMSE"].values
        if rev:
            labels, vals = labels[::-1], vals[::-1]
        colors_ = plt.cm.RdYlGn(
            np.linspace(0.85, 0.15, 10) if rev else np.linspace(0.15, 0.85, 10)
        )
        bars = ax.barh(range(10), vals, color=colors_,
                       edgecolor="white", linewidth=1)
        ax.set_yticks(range(10))
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlabel("Test RMSE", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.35)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fig7_top_bottom10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] Fig 7 kaydedildi")

    print(f"\n  Tüm grafikler → '{out_dir}/' klasörüne kaydedildi.")


# TABLOLAR

def print_and_save_tables(df: pd.DataFrame, out_dir: Path):
    sep = "=" * 70

    print(f"\n{sep}")
    print("TABLO 1: Config × Embedding Modeli (Test RMSE Ortalaması)")
    print(sep)
    t1 = df.pivot_table("Test_RMSE", "Config", "EmbedModel", "mean")[EMBED_ORDER]
    t1["ORTALAMA"] = t1.mean(axis=1)
    t1.loc["ORTALAMA"] = t1.mean()
    print(t1.round(4).to_string())

    print(f"\n{sep}")
    print("TABLO 2: ML Algoritması — Test Metrikleri Ortalaması")
    print(sep)
    t2 = df.groupby("MLModel")[["Test_RMSE", "Test_MAE", "Test_R2"]].mean()
    print(t2.round(4).to_string())

    print(f"\n{sep}")
    print("TABLO 3: Embedding Modeli — Test Metrikleri Ortalaması")
    print(sep)
    t3 = (df.groupby("EmbedModel")[["Test_RMSE", "Test_MAE", "Test_R2"]]
          .mean().loc[EMBED_ORDER])
    print(t3.round(4).to_string())

    print(f"\n{sep}")
    print("TABLO 4: En İyi 10 Kombinasyon (Test RMSE'ye Göre)")
    print(sep)
    top10 = df.nsmallest(10, "Test_RMSE")[
        ["Config", "EmbedModel", "MLModel",
         "Test_RMSE", "Test_MAE", "Test_R2"]
    ]
    print(top10.to_string(index=False))

    print(f"\n{sep}")
    print("TABLO 5: Tüm 60 Sonuç")
    print(sep)
    print(df.to_string(index=False))

    df.to_csv(out_dir / "tum_sonuclar.csv", index=False)
    print(f"\n  Tüm sonuçlar → {out_dir}/tum_sonuclar.csv")


#main fonksiyonu

if __name__ == "__main__":
    # 1. Veri yükle
    df = load_data(DATA_FILE)

    # split train and test
    train_df, test_df = split_data(df)

    # Deney Aşaması
    print("\n" + "=" * 60)
    print("DENEYLER")
    print("=" * 60)
    results = run_experiments(train_df, test_df)

    # Tablolar
    print_and_save_tables(results, RESULTS_DIR)

    # ve Grafikler
    print("\n" + "=" * 60)
    print("GRAFİKLER ÜRETİLİYOR")
    print("=" * 60)
    plot_all(results, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("TÜM DENEYLER TAMAMLANDI")
    print(f"  Sonuçlar : {RESULTS_DIR}/")
    print(f"  Önbellek : {EMBED_CACHE}/")
    print("=" * 60)
