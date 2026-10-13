"""Dan 1 — alat za IZBOR praga deduplikacije.

Prag se ne pogađa: gleda se (a) tabela koliko podataka koji prag "pojede" i
(b) same slike na različitim nivoima sličnosti. Ovaj skript pravi oboje i pakuje
u jednu samostalnu HTML stranu (slike su ugrađene kao data: URI, nema spoljnih fajlova).

    python scripts/dedup_review.py --emb data/emb/convnext_tiny --labels data/raw/labels.csv

Izlaz: reports/dedup_review.html  +  reports/dedup.json
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dogbreeds.dedup import cross_label_pairs, find_similar_pairs, threshold_sweep  # noqa: E402
from dogbreeds.embed import load as load_emb  # noqa: E402

# Opsezi sličnosti koje pregledamo. Donja granica je namerno niska: treba da VIDIŠ
# gde model počinje da spaja slike koje nisu isti pas, da bi prag imao obrazloženje.
BANDS = [(0.99, 1.01), (0.97, 0.99), (0.95, 0.97), (0.93, 0.95), (0.90, 0.93), (0.85, 0.90)]
THUMB = 150


def thumb_data_uri(path: Path, size: int = THUMB) -> str:
    """Sliku smanjimo i ugradimo kao data: URI -> HTML je samostalan, može se poslati/objaviti."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def sample_band(pairs: pd.DataFrame, lo: float, hi: float, k: int, seed: int = 0) -> pd.DataFrame:
    sub = pairs[(pairs["sim"] >= lo) & (pairs["sim"] < hi)]
    if len(sub) <= k:
        return sub
    return sub.sample(k, random_state=seed).sort_values("sim", ascending=False)


FONTS = ('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         "family=Archivo:wght@500;600;700&family=Source+Sans+3:wght@400;600"
         '&family=IBM+Plex+Mono:wght@400;500;600&display=swap">')

# Vizuelni jezik: KONTAKT-LIST iz mracne komore. Tema strane je forenzika fotografija,
# pa su parovi dva kadra spojena hairline-om (kao dva frejma na filmu), svi brojevi u
# mono fontu, a neutrali blago povuceni u hladno-zelenu stranu (fotografska hemija).
CSS = """
:root{
  --bg:#f7f9f8; --surface:#ecf1f0; --sunk:#e3eae9; --fg:#121a19; --muted:#536361;
  --line:#d3dedc; --line-strong:#b6c6c4; --accent:#0d6b68; --accent-soft:#d4e8e7;
  --warn:#a8430d; --warn-soft:#f7e4d7;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0c1211; --surface:#141d1c; --sunk:#101817; --fg:#e4ecea; --muted:#8d9f9c;
  --line:#243130; --line-strong:#354645; --accent:#48c2bc; --accent-soft:#143331;
  --warn:#ef8d53; --warn-soft:#2f1b0f;
}}
:root[data-theme="dark"]{
  --bg:#0c1211; --surface:#141d1c; --sunk:#101817; --fg:#e4ecea; --muted:#8d9f9c;
  --line:#243130; --line-strong:#354645; --accent:#48c2bc; --accent-soft:#143331;
  --warn:#ef8d53; --warn-soft:#2f1b0f;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:400 16px/1.65 "Source Sans 3","Segoe UI",system-ui,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1140px;margin:0 auto;padding:56px 24px 96px}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.88em;
  background:var(--sunk);padding:.12em .36em;border-radius:3px}

.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500;
  letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 14px}
h1{font-family:Archivo,"Segoe UI",sans-serif;font-weight:700;
  font-size:clamp(30px,4.4vw,42px);line-height:1.08;letter-spacing:-.025em;
  text-wrap:balance;margin:0 0 12px}
h2{font-family:Archivo,"Segoe UI",sans-serif;font-weight:600;font-size:21px;
  letter-spacing:-.015em;text-wrap:balance;margin:60px 0 8px;padding-top:22px;
  border-top:1px solid var(--line)}
h3{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:13px;
  letter-spacing:.04em;margin:34px 0 12px}
.lede{color:var(--muted);font-size:17px;max-width:66ch;margin:0}
.sub{color:var(--muted);max-width:70ch;margin:8px 0 0}
.meta{font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--muted);
  margin:0 0 28px}

.note{background:var(--surface);border:1px solid var(--line);
  border-left:2px solid var(--accent);padding:16px 20px;margin:22px 0;
  color:var(--muted);font-size:15px;max-width:76ch}
.note b{color:var(--fg);font-weight:600}
.note.flag{border-left-color:var(--warn)}

.tablewrap{overflow-x:auto;border:1px solid var(--line);margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:14px;
  font-variant-numeric:tabular-nums lining-nums}
th,td{padding:10px 14px;text-align:right;border-bottom:1px solid var(--line);
  white-space:nowrap}
td{font-family:"IBM Plex Mono",monospace;font-size:13px}
th{background:var(--sunk);font-family:"IBM Plex Mono",monospace;font-weight:500;
  color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.1em}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:0}
tbody tr.pick td{background:var(--accent-soft);color:var(--fg);font-weight:600}
tr.pick td:first-child::after{content:" \\2190 predlog";color:var(--accent);
  font-family:"Source Sans 3",sans-serif;font-weight:600;letter-spacing:0}

/* kontakt-list: dva kadra spojena hairline-om, bez zaobljenja - kao kontakt kopija */
.sheet{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));
  gap:16px;margin:14px 0 4px}
.pair{border:1px solid var(--line);background:var(--surface);display:flex;
  flex-direction:column}
.pair .frames{display:grid;grid-template-columns:1fr 1fr;gap:1px;
  background:var(--line-strong)}
.pair img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;
  background:var(--sunk)}
.pair .cap{padding:8px 10px 9px;font-size:12px;line-height:1.4;color:var(--muted);
  border-top:1px solid var(--line);display:flex;flex-direction:column;gap:2px}
.pair .sim{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:13px;
  color:var(--fg)}
.pair.mismatch{border-color:var(--warn);background:var(--warn-soft)}
.pair.mismatch .cap b{color:var(--warn);font-weight:600}
.band{font-family:"Source Sans 3",sans-serif;color:var(--muted);font-weight:400;
  font-size:12.5px;letter-spacing:0;text-transform:none}
"""


def render(sweep: pd.DataFrame, bands: dict, cross: pd.DataFrame, info: dict,
           suggest: float) -> str:
    def table(df: pd.DataFrame, headers: list[str], pick_col: str | None = None) -> str:
        th = "".join(f"<th>{h}</th>" for h in headers)
        rows = []
        for _, r in df.iterrows():
            cls = ' class="pick"' if pick_col and r[pick_col] == suggest else ""
            tds = "".join(f"<td>{html.escape(str(v))}</td>" for v in r.tolist())
            rows.append(f"<tr{cls}>{tds}</tr>")
        return (f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>')

    sw = sweep.copy()
    sw["pct_affected"] = (100 * sw["images_in_dupe_groups"] / info["n_images"]).round(2)
    sw_disp = sw[["threshold", "pairs", "groups_with_dupes", "images_in_dupe_groups",
                  "pct_affected", "redundant_images", "largest_group"]]
    n_fmt = f"{info['n_images']:,}".replace(",", ".")
    worst = int(sweep["largest_group"].max())
    worst_pct = 100 * worst / info["n_images"]

    parts = [
        "<title>Duplikati u Stanford Dogs</title>",
        FONTS,
        f"<style>{CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">Dan 1 &middot; odluka o pragu</p>',
        "<h1>Isti pas, dva kadra</h1>",
        '<p class="lede">Koliko slika u datasetu je zapravo ista scena snimljena dva puta '
        "&mdash; i gde povu&#263;i granicu da to curenje ne u&#273;e u test skup.</p>",
        f'<p class="meta">{n_fmt} slika &middot; {info["n_classes"]} rasa &middot; '
        f'embedding <code>{html.escape(info["model"])}</code> &middot; {info["dim"]}D</p>',
        '<div class="note"><b>&#352;ta gleda&#353;.</b> Dve slike su &bdquo;isti pas / ista '
        "scena&ldquo; ako im je kosinusna sli&#269;nost embeddinga iznad praga. Prenizak prag "
        "spaja razli&#269;ite pse i skuplja podatke u ogromne grupe; previsok ostavlja curenje "
        "izme&#273;u train i test skupa. Odluka se pravi iz tabele <i>i</i> iz samih slika "
        "ispod &mdash; ne iz pretpostavke.</div>",

        "<h2>1. Koliko dataseta koji prag zahvata</h2>",
        '<p class="sub">Grupe sa duplikatima idu <i>cele</i> u isti split (train '
        "<i>ili</i> val <i>ili</i> test), pa ni&#353;ta ne bri&#353;emo &mdash; samo "
        "spre&#269;avamo curenje. Kolona <i>redundantnih</i> pokazuje koliko bi slika "
        "<i>nestalo</i> da smo umesto grupisanja birali brisanje duplikata.</p>",
        table(sw_disp.round(2), ["prag", "parova", "grupa s dupl.", "slika u njima",
                                 "% dataseta", "redundantnih", "najve&#263;a grupa"],
              "threshold"),
        '<div class="note flag"><b>Klju&#269;na kolona je &bdquo;najve&#263;a grupa&ldquo;.</b> '
        "Union-Find radi tranzitivno zatvaranje: ako je A~B i B~C, sve tri idu zajedno. Na "
        f"najni&#382;em pragu to pravi lan&#269;anu reakciju &mdash; jedna grupa od "
        f"<b>{worst:,}</b> slika ({worst_pct:.1f}% dataseta) koja bi cela morala u jedan split "
        "i prakti&#269;no nam uni&#353;tila stratifikaciju. Prag mora biti dovoljno visok da "
        "lanci ostanu kratki.</div>".replace(",", "."),
        f'<div class="note">Medijana sli&#269;nosti sa najbli&#382;im susedom je '
        f'<b>{info["nn_sim_median"]:.3f}</b>, p99 je <b>{info["nn_sim_p99"]:.3f}</b> &mdash; '
        "tipi&#269;na slika <i>nema</i> blizak par, pa prag se&#269;e samo rep raspodele.</div>",

        "<h2>2. Kako to izgleda &mdash; parovi po nivou sli&#269;nosti</h2>",
        '<p class="sub">Levo i desno su dve <i>razli&#269;ite</i> slike iz dataseta. Tra&#382;i '
        "nivo na kom parovi prestaju da budu <i>isti pas</i> i postaju samo <i>ista rasa</i> "
        "&mdash; prag ide ta&#269;no iznad toga. Narand&#382;asti okvir zna&#269;i da par ima "
        "<b>razli&#269;ite labele</b>.</p>",
    ]

    for (lo, hi), items in bands.items():
        if not items:
            continue
        parts.append(f'<h3>sli&#269;nost {lo:.2f} &ndash; {hi:.2f} '
                     f'<span class="band">&middot; {items[0]["band_total"]} parova ukupno, '
                     f'prikazano {len(items)}</span></h3>')
        cards = []
        for it in items:
            mm = " mismatch" if it["breed_i"] != it["breed_j"] else ""
            label = (f'<b>{html.escape(it["breed_i"])}</b> &nbsp;vs&nbsp; '
                     f'<b>{html.escape(it["breed_j"])}</b>'
                     if mm else html.escape(it["breed_i"]))
            cards.append(
                f'<div class="pair{mm}"><div class="frames">'
                f'<img src="{it["uri_i"]}" alt="prva slika para" loading="lazy">'
                f'<img src="{it["uri_j"]}" alt="druga slika para" loading="lazy"></div>'
                f'<div class="cap"><span class="sim">{it["sim"]:.4f}</span>'
                f"<span>{label}</span></div></div>")
        parts.append(f'<div class="sheet">{"".join(cards)}</div>')

    parts += [
        "<h2>3. Besplatan signal: sli&#269;ne slike, razli&#269;ita labela</h2>",
        f'<p class="sub">{len(cross)} parova ima sli&#269;nost &ge; 0,93 a <b>razli&#269;itu</b> '
        "rasu. To su kandidati za pogre&#353;nu labelu &mdash; na&#273;eni <i>pre</i> nego "
        "&#353;to je ikakav model istreniran. Isti signal potvr&#273;ujemo cleanlab-om u "
        "Danu 3.</p>",
        table(cross.head(25)[["sim", "breed_i", "breed_j", "id_i", "id_j"]].round(4),
              ["sli&#269;nost", "labela A", "labela B", "id A", "id B"]),
        "</div>",
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/emb/convnext_tiny")
    ap.add_argument("--labels", default="data/raw/labels.csv")
    ap.add_argument("--images", default="data/raw/train")
    ap.add_argument("--min-sim", type=float, default=0.85)
    ap.add_argument("--per-band", type=int, default=12)
    ap.add_argument("--suggest", type=float, default=0.95)
    ap.add_argument("--out", default="reports/dedup_review.html")
    ap.add_argument("--json", default="reports/dedup.json")
    a = ap.parse_args()

    emb, ids, meta = load_emb(a.emb)
    df = pd.read_csv(a.labels)
    df["id"] = df["id"].astype(str)
    df = df.set_index("id").loc[ids].reset_index()
    images_dir = Path(a.images)

    print(f"[review] {len(df)} slika, {meta['model']} ({meta['dim']}D)")
    pairs, nn_sim = find_similar_pairs(emb, a.min_sim)
    print(f"[review] {len(pairs)} parova sa sim >= {a.min_sim}")
    sweep = threshold_sweep(pairs, len(df))
    print("\n" + sweep.to_string(index=False))

    breeds = df["breed"].to_numpy()
    ids_arr = df["id"].to_numpy()
    cross = cross_label_pairs(pairs, breeds)
    cross = cross.assign(id_i=ids_arr[cross["i"].to_numpy()], id_j=ids_arr[cross["j"].to_numpy()])
    print(f"\n[review] {len(cross)} parova sim>=0.93 sa razlicitim labelama")

    bands = {}
    for lo, hi in BANDS:
        sub = sample_band(pairs, lo, hi, a.per_band)
        band_total = int(((pairs["sim"] >= lo) & (pairs["sim"] < hi)).sum())
        items = []
        for _, r in sub.iterrows():
            i, j = int(r["i"]), int(r["j"])
            items.append({
                "sim": float(r["sim"]), "band_total": band_total,
                "breed_i": breeds[i], "breed_j": breeds[j],
                "id_i": ids_arr[i], "id_j": ids_arr[j],
                "uri_i": thumb_data_uri(images_dir / f"{ids_arr[i]}.jpg"),
                "uri_j": thumb_data_uri(images_dir / f"{ids_arr[j]}.jpg"),
            })
        bands[(lo, hi)] = items
        print(f"  pojas [{lo}, {hi}): {band_total} parova, uzorkovano {len(items)}")

    info = {
        "n_images": int(len(df)), "n_classes": int(df["breed"].nunique()),
        "model": meta["model"], "dim": int(meta["dim"]),
        "nn_sim_median": float(np.median(nn_sim)),
        "nn_sim_p99": float(np.percentile(nn_sim, 99)),
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(sweep, bands, cross, info, a.suggest), encoding="utf-8")

    Path(a.json).write_text(json.dumps({
        "info": info,
        "sweep": sweep.to_dict("records"),
        "top_pairs": pairs.head(50).assign(
            id_i=ids_arr[pairs.head(50)["i"].to_numpy()],
            id_j=ids_arr[pairs.head(50)["j"].to_numpy()],
            breed_i=breeds[pairs.head(50)["i"].to_numpy()],
            breed_j=breeds[pairs.head(50)["j"].to_numpy()]).to_dict("records"),
        "cross_label_suspects": cross.head(80).to_dict("records"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {out}\n-> {a.json}")


if __name__ == "__main__":
    main()
