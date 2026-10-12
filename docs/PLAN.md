# Plan rada: 7 dana (može i 5)

Cilj nije samo "model koji radi", nego **sistem** koji pokazuje da razmišljaš kao inženjer:
mereš, porediš, obrazlažeš odluke i znaš šta se dešava posle deploy-a.
Na intervjuu će te pitati **zašto**, ne samo **šta** — zato svaki dan ima deo "Nauči" i "Zapiši odluku".

---

## 1. Analiza zadatka (pročitaj pre nego što kreneš)

### Šta su tačno napisali i šta to znači

| Iz oglasa | Inženjerski prevod |
|---|---|
| "više od 10.000 fotografija, 120 rasa" | Gotovo sigurno Kaggle **Dog Breed Identification** (10.222 train slika, 120 rasa), podskup **Stanford Dogs** (20.580 slika) koji je i sam izvučen iz **ImageNet-a**. |
| "prikazuje tri najverovatnija odgovora" | Glavna poslovna metrika je **top-3 accuracy**, ne top-1. Verovatnoće se prikazuju korisniku → moraju biti **kalibrisane**. |
| "dovoljno mali da radi brzo na srednje jakom hardveru" | Definiši **budžet**: npr. p95 latencija < 100 ms na 4 CPU niti, model < 50 MB. Meri na svom laptopu (Ryzen 7 5700U je baš "srednje jak hardver"). |
| "dovoljno veliki za visoku preciznost" | Ovo je **trade-off** → nacrtaj Pareto krivu (tačnost vs latencija) za 3-5 modela i izaberi tačku. To je najjači slajd na intervjuu. |
| labeliranje, trening, evaluacija, HPO, monitoring, kontinualno unapređivanje | Ceo **ML lifecycle / MLOps**. Ne moraš sve napraviti "enterprise", ali za svaku stavku treba da postoji **radeći minimum + objašnjenje kako bi se skaliralo**. |

### Tri zamke koje većina kandidata neće primetiti (tvoja prednost)

1. **Curenje podataka iz ImageNet-a.** Modeli pretrenirani na ImageNet-1k su već "videli" ove pse sa labelama. Uz to, Stanford Dogs ima i interno curenje: [Voxel51 je našao ~11% test slika koje su gotovo identične train slikama](https://voxel51.com/blog/on-leaky-datasets-and-a-clever-horse) (isti pas, slikan u nizu). → Interne metrike su optimistične. Rešenja: deduplikacija preko embeddinga + grupni split, self-supervised backbone (DINOv3, treniran **bez** ImageNet labela), i **spoljni test skup** iz drugog izvora.
2. **Realni korisnici KP-a ne šalju slike iz ImageNet-a.** Šalju slike telefonom, mešance, štence, mačke, slike bez psa. Treba ti: prag sigurnosti ("nisam siguran"), detekcija "nije pas" (out-of-distribution) i plan za mešance.
3. **"Brzo" se meri, ne pretpostavlja.** GMAC-ovi dobro prate latenciju na CPU-u, ali loše na GPU-u ([studija iz 2026, merena na Ryzen 5 5500U](https://arxiv.org/abs/2607.01984)). Zato meriš na ciljnom hardveru.

### Hardver i alati (prilagođeno tebi)

- Tvoj laptop nema NVIDIA GPU → **trening na Kaggle Notebooks** (besplatno T4/P100, sesija do 12h, nedeljna kvota ~30h — proveri u podešavanjima naloga). Colab kao rezerva.
- **Inference na tvom CPU-u** → upravo to je dokaz da radi "na srednje jakom hardveru".
- Stack: PyTorch + **timm** (najbolja biblioteka pretreniranih modela za vid), **W&B** (praćenje eksperimenata, radi iz Kaggle-a), **Optuna** (HPO), **Label Studio** + **cleanlab** (labeliranje i kvalitet labela), **ONNX Runtime** (brz CPU inference), **FastAPI** (servis), **Prometheus + Grafana** + **Evidently** (monitoring).

### Izbor modela — polazna hipoteza (proveri merenjem, Dan 2)

| Uloga | Model (timm) | Param | Zašto |
|---|---|---|---|
| Brzi baseline | `efficientnet_b0.ra4_e3600_r224_in1k` | 5.3M | Konzistentno na Pareto granici tačnost/resursi |
| Kandidat za produkciju | `mobilenetv4_conv_medium.e500_r256_in1k` | 9.7M | Moderna mobilna arhitektura, brza na CPU |
| Kandidat za produkciju | `convnext_tiny.dinov3_lvd1689m` | 28M | Self-supervised (bez ImageNet labela → manje curenja), jake reprezentacije. **Proveri DINOv3 licencu** za komercijalnu upotrebu. |
| "Učitelj" (opciono) | `convnext_base.dinov3_lvd1689m` ili veći | 89M | Za knowledge distillation u mali model |

Orijentaciono merenje u cloud okruženju (2 CPU jezgra, batch 1, FP32 ONNX): MobileNetV3 ~6 ms, EfficientNet-B0 ~10 ms, MobileNetV4-M ~12 ms, ConvNeXt-Tiny ~66 ms. Na tvom laptopu će biti drugačije — zato `scripts/benchmark_candidates.py`.
Zaključak koji ćeš verovatno dobiti: **svi** su dovoljno brzi za web servis, pa se izbor svodi na tačnost i cenu servera po zahtevu. To je odličan uvid za intervju.

---

## 2. Dan po dan

Svaki dan: **Cilj → Radi → Nauči → Zapiši odluku**. Na kraju dana pošalji mi `history.csv` / `metrics.json` / W&B link i zajedno pregledamo pre sledećeg koraka.

### Dan 0 (danas, 1-2h) — Setup
- [ ] Kaggle nalog + verifikacija telefona; prihvati pravila takmičenja.
- [ ] GitHub repo `kp-dog-breeds` (javan — biće ti portfolio), push ovog koda.
- [ ] Lokalno: `venv`, `pip install -r requirements.txt`, `python -m pytest -q` mora proći.
- [ ] `python scripts/benchmark_candidates.py --threads 4` → sačuvaj tabelu.
- **Nauči (1h):** [fast.ai lekcija 1-2](https://course.fast.ai/) — intuicija za transfer learning.

### Dan 1 — Podaci, EDA, labeliranje
- [ ] Kaggle notebook: učitaj `labels.csv`, napravi EDA: broj slika po rasi (distribucija klasa), rezolucije, odnos stranica, primeri po rasi, grayscale slike.
- [ ] `python -m dogbreeds.data --config configs/kaggle.yaml` → train/val/test (80/10/10, stratifikovano).
- [ ] **TVOJ ZADATAK — deduplikacija:** izvuci embedding svake slike (`timm.create_model(..., num_classes=0)`), kosinusna sličnost, parovi > ~0.95 su "isti pas". Napravi kolonu `group` u `labels.csv` (union-find nad parovima) → `data.py` tada automatski radi grupni split. Izmeri koliko duplikata ima.
- [ ] **Label Studio** lokalno (`pip install label-studio`, `label-studio start`): projekat "Image Classification" sa 120 rasa + "nije pas" + "mešanac". Uvezi ~50 slika. Ovo je alat za (a) proveru sumnjivih labela (Dan 3) i (b) labeliranje produkcionih slika (Dan 6).
- **Nauči:** CNN osnove ([CS231n beleške](https://cs231n.github.io/)), [D2L poglavlje baš o ovom datasetu](https://d2l.ai/chapter_computer-vision/kaggle-dog.html), šta je data leakage.
- **Zapiši odluku:** zašto 80/10/10, zašto stratifikacija, koliko duplikata je nađeno.

### Dan 2 — Baseline i izbor modela
- [ ] **Linear probe** (zamrznut backbone): `--set train.freeze_epochs=100 train.epochs=5` — brz baseline, koliko "znaju" pretrenirane reprezentacije.
- [ ] **Full fine-tune** 3 kandidata (tabela gore), 10-15 epoha, W&B uključen (`logging.wandb=true`).
- [ ] Tabela + grafik: val top-1/top-3 vs CPU latencija → **Pareto kriva**. Izaberi 1-2 modela.
- **Nauči:** [Karpathy — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) (obavezno), [timm dokumentacija](https://huggingface.co/docs/timm), šta rade AdamW, cosine LR, warmup, label smoothing, EMA, AMP (svaka linija u `train.py`).
- **Zapiši odluku:** koji model i zašto (brojevi!).

### Dan 3 — Duboka evaluacija i kvalitet podataka
- [ ] `python -m dogbreeds.evaluate ...` → top-1/3/5, log-loss, **ECE pre/posle temperature scaling-a**, najslabije rase, najčešće zamene (husky↔malamut, terijeri...). Pogledaj 20 pogrešnih slika očima — da li je greška modela ili labele?
- [ ] **cleanlab:** `cleanlab.filter.find_label_issues(labels, pred_probs)` nad out-of-sample verovatnoćama (idealno 5-fold cross-val na linear probe-u, jeftino je) → top 50 sumnjivih → pregled u Label Studio → ispravi/izbaci → ponovo treniraj i uporedi.
- [ ] **Spoljni test skup:** 100-200 slika iz drugog izvora — [Tsinghua Dogs](https://cg.cs.tsinghua.edu.cn/ThuDogs/) (130 rasa, slike sa telefona, drugi izvor) za rase koje se poklapaju, + tvoje slike pasa iz okoline. Ovo je **pošteni** broj koji prijavljuješ.
- **Nauči:** [MIT Data-Centric AI — Label Errors](https://dcai.csail.mit.edu/2024/label-errors/), [Guo et al. 2017 — On Calibration](https://arxiv.org/abs/1706.04599), precision/recall/F1 po klasi, zašto accuracy nije dovoljan.

### Dan 4 — Pronalaženje najboljih hiperparametara
- [ ] **TVOJ ZADATAK — `dogbreeds/hpo.py`** (zajedno ga pišemo): Optuna study, `TPESampler` + `MedianPruner`, `train.run(cfg, trial)` već ima pruning hook.
  Prostor: `lr` (log 1e-4..3e-3), `weight_decay` (log 1e-3..1e-1), `label_smoothing` (0..0.2), `drop_path_rate` (0..0.3), `mixup_alpha` (0 / 0.2), `img_size` (224/256/288), `backbone_lr_mult` (0.05..1).
  Budžet: 20-30 trial-a × 4-5 epoha → najbolji config treniraš "do kraja". Storage: `sqlite:///optuna.db` da preživi prekid Kaggle sesije.
- [ ] Grafici: `optuna.visualization.plot_param_importances`, `plot_optimization_history`.
- [ ] (Opciono, ako ima vremena) **Knowledge distillation:** veliki "učitelj" → mali "učenik" (KL loss nad soft labelama). Često +1-3% tačnosti za mali model, bez gubitka brzine.
- **Nauči:** [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) (obavezno), [Optuna tutorijal](https://optuna.readthedocs.io/en/stable/tutorial/index.html), [Hinton — Distillation](https://arxiv.org/abs/1503.02531). Zašto Bayesian/TPE > grid search, šta je pruning.

### Dan 5 — Optimizacija za CPU + servis
- [ ] `python -m dogbreeds.export ... --int8` → `scripts/eval_onnx.py` za FP32 i INT8 (tačnost mora ostati blizu!) → benchmark na laptopu.
- [ ] **TVOJ ZADATAK — `service/` (FastAPI):** `POST /predict` (upload → top-3 + `max_prob` + `model_version`), `POST /feedback` (korisnik kaže tačnu rasu), `GET /health`, `GET /metrics`. Validacija ulaza (tip, veličina fajla), timeout. `Dockerfile` (python:slim + onnxruntime, **bez torch-a**).
- [ ] Prag sigurnosti: ako `max_prob` < τ (izaberi τ sa val skupa) → "Nisam siguran, evo 3 moguće rase". **"Nije pas" gate:** najjednostavnije — ImageNet-1k klasifikator (dog klase su 151-268) ili CLIP zero-shot "photo of a dog" vs ostalo.
- [ ] Mini frontend (može i Gradio/Streamlit, ili jednostavna HTML strana) za demo.
- **Nauči:** [ONNX Runtime kvantizacija](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), static vs dynamic kvantizacija, batch vs online inference, p50 vs p95 latencija.

### Dan 6 — Monitoring i kontinualno unapređivanje
- [ ] Logovanje svake predikcije u bazu (SQLite → u stvarnosti Postgres): vreme, hash slike, top-3, `max_prob`, `entropy`, latencija, `model_version`, feedback. (Ovde koristiš SQL iz oglasa.)
- [ ] **Prometheus** metrike iz servisa (`prometheus-fastapi-instrumentator` + svoje: histogram `max_prob`, brojač po predviđenoj rasi) + **Grafana** dashboard (docker-compose).
- [ ] **Drift:** dnevni [Evidently](https://www.evidentlyai.com/blog/embedding-drift-detection) izveštaj — distribucija sigurnosti, distribucija predviđenih rasa i **embedding drift** u odnosu na referentni (val) skup. Labele u produkciji kasne/ne postoje → zato pratimo proxy signale.
- [ ] **Petlja unapređivanja** (dokumentuj dijagramom, implementiraj minimalno):
  niska sigurnost + negativan feedback + nasumični uzorak → Label Studio red → nove labele → retrain → **champion/challenger** na fiksnom "golden" test skupu → promocija samo ako je bolji na top-3 i nije sporiji → verzionisan model (W&B Model Registry) → postepeno puštanje (shadow / A-B).
- [ ] GitHub Actions: `pytest` na svaki push.
- **Nauči:** [Made With ML (MLOps kurs)](https://madewithml.com/), [Evidently — embedding drift](https://www.evidentlyai.com/blog/embedding-drift-detection), data drift vs concept drift, active learning.

### Dan 7 — Završnica i priprema za razgovor
- [ ] README: rezultati, Pareto grafik, arhitektura, ograničenja, "šta bih radio sa više vremena".
- [ ] **Model card** (namena, podaci, metrike po grupama, ograničenja, etika — npr. ne koristiti za procenu cene psa ili "čistokrvnosti").
- [ ] Kratak demo (GIF/video 60 s) + 5-6 slajdova za prezentaciju na razgovoru.
- [ ] Prođi [`INTERVIEW.md`](INTERVIEW.md) naglas, pa uradimo **probni intervju** zajedno.

### Ako imaš samo 5 dana
Dan 0+1 spoji; Dan 3 skrati (bez spoljnog test skupa, samo cleanlab); Dan 6+7 spoji (monitoring kao dijagram + minimalni Prometheus). HPO i servis ne skraćuj — to su najvidljiviji delovi.

---

## 3. Izvori (filtrirano — ovo vredi tvog vremena)

**Osnove (prioritet):**
- [Practical Deep Learning for Coders — fast.ai](https://course.fast.ai/) — najbrži put do praktičnog razumevanja
- [Dive into Deep Learning — poglavlje "Dog Breed Identification"](https://d2l.ai/chapter_computer-vision/kaggle-dog.html) — isti dataset
- [Karpathy — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- [Google — Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)

**Alati:**
- [timm dokumentacija](https://huggingface.co/docs/timm) · [Optuna](https://optuna.readthedocs.io/) · [Weights & Biases](https://docs.wandb.ai/) · [Label Studio](https://labelstud.io/) · [cleanlab](https://docs.cleanlab.ai/) · [ONNX Runtime](https://onnxruntime.ai/docs/) · [FastAPI](https://fastapi.tiangolo.com/) · [Evidently](https://docs.evidentlyai.com/)

**MLOps / sistemi (za "kompletan sistem" deo):**
- [Made With ML](https://madewithml.com/) — besplatan, praktičan MLOps kurs
- Chip Huyen — *Designing Machine Learning Systems* (O'Reilly) — pogl. 8-9 (monitoring, continual learning) su skoro 1:1 tvoj zadatak
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/) — predavanja o deploy-u i monitoringu

**Radovi / članci za "duboko znanje" na intervjuu:**
- [Guo et al. — On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [Hinton et al. — Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Voxel51 — curenje u Stanford Dogs](https://voxel51.com/blog/on-leaky-datasets-and-a-clever-horse)
- [MIT DCAI — Label Errors & Confident Learning](https://dcai.csail.mit.edu/2024/label-errors/)
- [Studija lakih CNN-ova pod ograničenim resursima (2026)](https://arxiv.org/abs/2607.01984)

**Podaci:**
- [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) · [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) · [Tsinghua Dogs](https://cg.cs.tsinghua.edu.cn/ThuDogs/)

---

## 4. Dnevnik odluka (popunjavaj usput — ovo je zlato za intervju)

| Datum | Odluka | Alternative | Zašto (brojevi) |
|12.10|Budžet: model ≤ 50 ms p95, kandidati B0 / MNv4-M / ConvNeXt-T (DINOv3)|MNv3, B2, ViT-S|Na Ryzen 7 5700U svi < 70 ms; usko grlo je propusnost (B0 ≈ 95 img/s naspram ConvNeXt-T ≈ 15 img/s), ne latencija. ViT-S je brži od ConvNeXt-T na mom CPU-u, a sporiji u cloud-u, pa redosled zavisi od hardvera.|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | Trening na Kaggle GPU, inference na CPU | Colab, lokalni CPU trening | Laptop bez NVIDIA GPU; CPU inference je i cilj zadatka |
| | timm + PyTorch | Keras/TF | Najveći izbor pretreniranih modela, isti preprocessing config za trening i produkciju |
| | | | |
