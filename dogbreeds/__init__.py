"""kp-dog-breeds: prepoznavanje rase psa sa fotografije (120 rasa, top-3 odgovora)."""

__version__ = "0.1.0"

# Windows: podrazumevana konzolna kodna strana je cp1252 i ruši svaki `print` sa
# našim dijakriticima (č, ć, š, ž, đ) uz UnicodeEncodeError. Pošto su svi ispisi u
# projektu na srpskom, stdout/stderr prebacujemo na UTF-8 čim se paket uveze.
# errors="replace" je pojas i tregeri: radije nečitak znak nego pad CLI alata.
def _force_utf8_stdio() -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
                reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # presretnut/zatvoren stream (npr. pod wandb-om)
            pass


_force_utf8_stdio()
