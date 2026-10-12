# Priprema za intervju

Odgovori su kratki namerno — nauči **suštinu**, pa je objasni svojim rečima na primeru iz projekta.
Najjači odgovor uvek ima strukturu: **šta sam uradio → zašto → kako sam izmerio → šta bih sledeće**.

## A. Tvoja priča (2 minuta, nauči napamet strukturu)

1. Problem: 120 rasa, ~10k slika, top-3, mali i brz model.
2. Ključni uvid: dataset je iz ImageNet-a → metrike su optimistične; zato deduplikacija, self-supervised backbone i spoljni test skup.
3. Odluka o modelu: Pareto kriva tačnost/latencija na ciljnom CPU-u → izabran X (top-3 = __%, p95 = __ ms).
4. Sistem: Label Studio + cleanlab → timm trening na Kaggle GPU → Optuna → ONNX INT8 → FastAPI → Prometheus/Evidently → feedback petlja.
5. Šta bih sledeće: podaci iz realne produkcije, mešanci, distilacija, A/B test.

## B. Podaci i labeliranje

**Kako deliš podatke i zašto?** Train/val/test stratifikovano po rasi (80/10/10). Val za izbor modela i HP, test samo jednom na kraju. Ako postoje gotovo identične slike (isti pas), grupni split — inače je test "lak" jer je model video skoro istu sliku.

**Šta je data leakage? Primer iz projekta.** Informacija iz test skupa koja procuri u trening. Ovde: (1) duplikati između splitova, (2) pretrenirani ImageNet model je već video ove slike sa labelama. Posledica: precenjena tačnost u odnosu na produkciju.

**Kako bi labelirao nove podatke?** Label Studio, jasna uputstva + klase "nije pas"/"mešanac"/"ne znam". Prvo labeliramo ono što model ne zna (niska sigurnost) — active learning. Kontrola kvaliteta: dupla anotacija na uzorku, meri se slaganje anotatora (Cohen's kappa).

**Kako nalaziš pogrešne labele?** Confident learning (cleanlab): primeri gde model sa visokom sigurnošću tvrdi drugu klasu na out-of-sample predikcijama. Pregledam ručno, ne brišem automatski.

**Neuravnotežene klase?** Ovde je umerena (≈60-130 po rasi). Opcije: stratifikacija, praćenje per-class metrika, weighted sampler ili class weights ako slabe klase zaostaju, više podataka za njih.

## C. Model i trening

**Zašto transfer learning?** 10k slika je premalo za trening od nule; pretrenirani model već zna ivice, teksture, oblike. Fine-tuning prilagođava te reprezentacije rasama. (U studiji iz 2026. pretrening je dao +15-20 poena na težim datasetima.)

**Linear probe vs fine-tuning?** Linear probe: zamrznut backbone, uči se samo poslednji sloj — brz baseline i mera kvaliteta reprezentacija. Fine-tuning: uči se sve, obično bolje, ali rizik od overfittinga i "zaboravljanja" → manji LR za backbone.

**Zašto discriminative LR i zamrzavanje u prvoj epohi?** Nova glava je nasumična; njeni veliki gradijenti bi na početku pokvarili dobre pretrenirane težine. Prvo se glava "smiri", pa backbone uči sporije.

**AdamW vs Adam?** AdamW odvaja weight decay od adaptivnog gradijenta → regularizacija radi kako treba.

**Cosine LR + warmup?** Warmup sprečava nestabilan početak; cosine postepeno smanjuje LR da bi model "sleteo" u dobar minimum.

**Augmentacije i zašto baš te?** RandomResizedCrop (pas na različitim pozicijama/veličinama), flip, RandAugment, Random Erasing (delimično zaklonjen pas). Ne koristim vertikalni flip niti jake promene boje — boja krzna je bitna osobina rase.

**Label smoothing / Mixup / CutMix?** Regularizacija: model ne sme biti 100% siguran; Mixup/CutMix mešaju slike i labele. Pomaže generalizaciji, ali utiče na kalibraciju — zato kalibrišem posle.

**EMA težina?** Pokretni prosek težina tokom treninga; stabilniji i često bolji model za evaluaciju.

**Mixed precision (AMP)?** FP16 računanje na GPU-u: ~2x brže, manje memorije; GradScaler sprečava underflow gradijenata.

**Kako znaš da li je overfitting?** Train loss pada, val loss raste / val tačnost stagnira. Rešenja: više augmentacije, regularizacija, early stopping, više podataka.

**CNN vs Vision Transformer?** ViT treba više podataka i sporiji je na CPU-u za istu tačnost na malim rezolucijama; CNN-ovi (EfficientNet, MobileNet, ConvNeXt) imaju korisne induktivne pretpostavke (lokalnost) i bolji su izbor za brz CPU inference. Pokazujem merenjem, ne verovanjem.

**Šta je knowledge distillation?** Veliki model (učitelj) daje "meke" verovatnoće; mali model (učenik) uči da ih imitira. Učenik dobija znanje o sličnosti klasa (husky je sličniji malamutu nego pudli) → bolji od istog modela treniranog samo na tvrdim labelama.

## D. Evaluacija

**Koje metrike i zašto?** Top-3 accuracy (poslovna — prikazujemo 3 odgovora), top-1, log-loss, per-class accuracy (gde smo slabi), ECE (da li su verovatnoće poštene), latencija p50/p95 i veličina modela.

**Šta je kalibracija i zašto je bitna ovde?** Ako model kaže 80%, treba da je u pravu ~80% puta. Prikazujemo procente korisniku i koristimo prag "nisam siguran" → moraju značiti nešto. Temperature scaling: jedan parametar T fitovan na val skupu; ne menja redosled (top-k ostaje isti).

**Zašto ne koristiš test skup za izbor hiperparametara?** Onda test više nije nepristrasna procena — "prilagodio" bih se testu.

**Kako bi uporedio dva modela pošteno?** Isti split, isti test skup, više seed-ova ako je razlika mala, interval poverenja (bootstrap) ili McNemar test za razliku u tačnosti.

## E. Hiperparametri

**Grid vs random vs Bayesian (TPE)?** Grid eksplodira kombinatorno; random je iznenađujuće dobar; TPE uči iz prethodnih pokušaja gde su dobri regioni. Pruning (MedianPruner) gasi loše trial-ove posle par epoha → ista GPU kvota, više pokušaja.

**Koji HP su najvažniji?** Gotovo uvek learning rate, pa weight decay i augmentacija/regularizacija. Proveravam `plot_param_importances`.

**Kako da HPO ne traje večno?** Kratki trial-ovi (manje epoha, manja rezolucija), pruning, logaritamska skala za LR, pa samo najbolji config "do kraja".

## F. Produkcija i monitoring

**Kako postižeš brzinu na CPU-u?** Mala arhitektura izabrana merenjem → ONNX Runtime (grafovske optimizacije) → INT8 kvantizacija (statička, kalibrisana na val slikama) → provera da tačnost nije pala. Servis bez PyTorch-a.

**Šta je training/serving skew i kako si ga sprečio?** Razlika između obrade u treningu i produkciji (npr. drugačiji resize) → tiho pada tačnost. Preprocessing čitam iz istog timm config-a i imam test koji proverava da je NumPy preprocessing identičan treningu.

**Šta pratiš u produkciji kad nemaš labele?** Operativno: latencija, greške, throughput. Model: distribucija sigurnosti (`max_prob`), udeo "nisam siguran", distribucija predviđenih rasa, embedding drift. Korisnički feedback kao delimične labele.

**Data drift vs concept drift?** Data drift: menjaju se ulazi (nove kamere, sezonski štenci). Concept drift: menja se veza ulaz→labela (npr. nova popularna rasa ili mešanac koji se počne zvati određenim imenom).

**Kako bi izgledalo kontinualno unapređivanje?** Signal (drift / feedback / niska sigurnost) → uzorkovanje → labeliranje → retrain → champion/challenger na golden setu → shadow ili A/B → promocija ili rollback. Sve verzionisano (podaci, kod, model).

**Šta ako korisnik pošalje mačku?** Gate "nije pas" pre klasifikatora + prag sigurnosti. Bolje "nisam siguran" nego samouvereno pogrešno.

**Mešanci?** Najrealniji slučaj na KP-u. Top-3 sa procentima već nagoveštava mešavinu; u UI-ju pisati "liči na…" umesto "ovo je…"; dugoročno klasa "mešanac" ili multi-label pristup.

## G. Pitanja koja TI postaviš njima (pokazuje zrelost)

- Gde bi ovakav model živeo na platformi — auto-popunjavanje kategorije/atributa oglasa ili moderacija?
- Kakav je njihov ML stack (trening, serving, monitoring)? Koriste li GPU u produkciji ili CPU?
- Kako mere uticaj AI funkcionalnosti na korisničko iskustvo — A/B testovi, koje metrike?
- Na čemu bi praktikant radio prvih mesec dana i kako izgleda mentorstvo?
