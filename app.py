import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import joblib
import requests
import os
import time
import math
from datetime import datetime, timedelta
from io import StringIO
from zoneinfo import ZoneInfo
from statsmodels.tsa.seasonal import STL
import holidays
from dotenv import load_dotenv
load_dotenv()

app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}])
app.title = "OkosMérő"
server = app.server

BASE = os.path.dirname(os.path.abspath(__file__))
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY","")
VISUAL_CROSSING_KEY = os.environ.get("VISUAL_CROSSING_KEY","")
BUDAPEST_TZ = ZoneInfo("Europe/Budapest")
hu_holidays = holidays.Hungary(years=list(range(2015,2028)))

C = {'bg':'#050d1a','sb':'#070f1e','card':'#0a1628','card2':'#0f1923','brd':'#1a2d42',
     'txt':'#cbd5e1','mut':'#64748b','or':'#FF6600','gr':'#10b981','bl':'#0066CC',
     'rd':'#ef4444','yw':'#f59e0b','cy':'#4b9cd3','wh':'#f1f5f9'}

CHART = dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=C['txt'],family='Inter,sans-serif',size=10),
    margin=dict(l=40,r=15,t=35,b=35),
    xaxis=dict(gridcolor=C['brd'],showline=False,color=C['mut'],zeroline=False),
    yaxis=dict(gridcolor=C['brd'],showline=False,color=C['mut'],zeroline=False),
    showlegend=False)

# ============================================================
# MODELL: CatBoost V10 — validált: MAE 108.48 MWh | MAPE 2.50%
# (MAVIR hivatalos napelőtti előrejelzés u.azon teszten: 244.45)
# ============================================================
bundle = None
MODELL_HIBA = None
try:
    bundle = joblib.load(f"{BASE}/catboost_model_final.pkl")
    MODEL = bundle["model"]
    FEATURES = bundle["features"]
    CAT_FEATS = bundle.get("cat_features") or []
except Exception as e:
    MODELL_HIBA = str(e)

def model_predict(X_df):
    """Kategorikus konverzió a bundle előírása szerint (int64→string),
    majd CatBoost jóslás + 2000 MWh alsó levágás."""
    X = X_df[FEATURES].copy()
    for c in CAT_FEATS:
        X[c] = X[c].astype('int64').astype(str)
    return np.maximum(MODEL.predict(X), 2000.0)

# ============================================================
# GYORSÍTÓTÁR
# ============================================================
CACHE = {}
def cachelt(kulcs, ttl_sec, fn, ok_index, force=False):
    most = time.time()
    rec = CACHE.get(kulcs)
    if rec and not force and (most - rec["ido"]) < ttl_sec:
        return rec["ertek"]
    ertek = fn()
    if ertek[ok_index]:
        CACHE[kulcs] = {"ido": most, "ertek": ertek}
        return ertek
    if rec:
        print(f"[CACHE] {kulcs}: friss hívás sikertelen, korábbi jó adat "
              f"({int((most-rec['ido'])/60)} perce)", flush=True)
        return rec["ertek"]
    return ertek

def _helyi_most():
    """Budapesti falióra, a szerver saját időzónájától függetlenül."""
    return datetime.now(BUDAPEST_TZ).replace(tzinfo=None)

def _ma():
    return _helyi_most().replace(hour=0,minute=0,second=0,microsecond=0)

def _entsoe():
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=ENTSOE_API_KEY, timeout=30)

def _helyi(sorozat):
    s = sorozat.copy()
    s.index = s.index.tz_convert('Europe/Budapest').tz_localize(None)
    return s

def _naponkent_oras(sorozat):
    """Helyi idejű sorozat → {date: [24 órás átlag]}."""
    s = sorozat.resample('h').mean()
    ki = {}
    for d, g in s.groupby(s.index.date):
        if len(g) >= 20:
            teljes = g.reindex(pd.date_range(pd.Timestamp(d), periods=24, freq='h'))
            ki[d] = teljes.interpolate(limit_direction='both').tolist()
    return ki

# ============================================================
# ÉLŐ ADATFORRÁSOK
# ============================================================
def get_eur_huf():
    try:
        r = requests.get("https://data-api.ecb.europa.eu/service/data/EXR/D.HUF.EUR.SP00.A",
            params={"startPeriod":(_helyi_most()-timedelta(days=7)).strftime("%Y-%m-%d"),
                    "endPeriod":_helyi_most().strftime("%Y-%m-%d"),"format":"csvdata"},timeout=10)
        df = pd.read_csv(StringIO(r.text))[["TIME_PERIOD","OBS_VALUE"]].dropna()
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"],errors="coerce")
        return float(df["OBS_VALUE"].dropna().iloc[-1]),True
    except Exception as e:
        print(f"[HIBA] ECB árfolyam: {e}", flush=True)
        return None,False

def get_ho():
    for kiserlet in range(2):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude":47.5,"longitude":19.0,"current_weather":"true","timezone":"Europe/Budapest"},timeout=10)
            d = r.json()
            if "current_weather" not in d:
                print(f"[HIBA] Open-Meteo (hőmérséklet) {kiserlet+1}. — HTTP {r.status_code}: {str(d)[:150]}", flush=True)
                time.sleep(3); continue
            return float(d["current_weather"]["temperature"]),True
        except Exception as e:
            print(f"[HIBA] Open-Meteo (hőmérséklet) {kiserlet+1}.: {e}", flush=True)
            time.sleep(3)
    return None,False

def get_ho_vc():
    if not VISUAL_CROSSING_KEY: return None,False
    try:
        r = requests.get("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/47.5,19.0/today",
            params={"key":VISUAL_CROSSING_KEY,"unitGroup":"metric","include":"current","elements":"temp"},timeout=15)
        d = r.json() if r.status_code==200 else {}
        if "currentConditions" in d and d["currentConditions"].get("temp") is not None:
            return float(d["currentConditions"]["temp"]),True
        print(f"[HIBA] Visual Crossing (hőmérséklet): HTTP {r.status_code}", flush=True)
        return None,False
    except Exception as e:
        print(f"[HIBA] Visual Crossing (hőmérséklet): {e}", flush=True)
        return None,False

def get_ho_barmelyik():
    t,ok = get_ho()
    if ok: return t,ok
    return get_ho_vc()

def get_idojaras():
    """Órás időjárás TEGNAPTÓL +4 napig — a delta feature-ökhöz a
    célnap előtti nap is kell. daily: holnaptól 4 nap a panelhez."""
    ma = _ma()
    for kiserlet in range(2):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude":47.5,"longitude":19.0,
                    "hourly":"temperature_2m,relative_humidity_2m,direct_radiation,wind_speed_10m,precipitation",
                    "daily":"temperature_2m_max,temperature_2m_min,weathercode",
                    "timezone":"Europe/Budapest",
                    "start_date":(ma-timedelta(days=1)).strftime("%Y-%m-%d"),
                    "end_date":(ma+timedelta(days=4)).strftime("%Y-%m-%d")},timeout=15)
            d = r.json()
            if "hourly" not in d:
                print(f"[HIBA] Open-Meteo (előrejelzés) {kiserlet+1}. — HTTP {r.status_code}: {str(d)[:150]}", flush=True)
                time.sleep(3); continue
            hourly = pd.DataFrame({"Datum":pd.to_datetime(d["hourly"]["time"]),
                "Homerseklet_C":d["hourly"]["temperature_2m"],
                "Paratartalom_szazalek":d["hourly"]["relative_humidity_2m"],
                "Napsugarzas_W_m2":d["hourly"]["direct_radiation"],
                "Szelsebesseg_kmh":d["hourly"]["wind_speed_10m"],
                "Csapadek_mm":d["hourly"]["precipitation"]}).dropna(subset=["Homerseklet_C"])
            daily = {"max":d["daily"]["temperature_2m_max"][2:6],
                     "min":d["daily"]["temperature_2m_min"][2:6],
                     "code":d["daily"]["weathercode"][2:6]}
            if len(hourly) < 48:
                time.sleep(3); continue
            return hourly,daily,True
        except Exception as e:
            print(f"[HIBA] Open-Meteo (előrejelzés) {kiserlet+1}.: {e}", flush=True)
            time.sleep(3)
    return None,None,False

def get_idojaras_vc():
    if not VISUAL_CROSSING_KEY: return None,None,False
    try:
        ma = _ma()
        kezd = (ma-timedelta(days=1)).strftime("%Y-%m-%d")
        veg = (ma+timedelta(days=4)).strftime("%Y-%m-%d")
        r = requests.get(
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/47.5,19.0/{kezd}/{veg}",
            params={"key":VISUAL_CROSSING_KEY,"unitGroup":"metric","include":"hours,days",
                    "elements":"datetime,temp,humidity,solarradiation,windspeed,precip,tempmax,tempmin"},timeout=20)
        d = r.json() if r.status_code==200 else {}
        if "days" not in d or len(d["days"])<3:
            print(f"[HIBA] Visual Crossing (előrejelzés): HTTP {r.status_code}", flush=True)
            return None,None,False
        sorok = []
        for nap in d["days"]:
            for h in nap.get("hours",[]):
                sorok.append({"Datum":pd.to_datetime(f"{nap['datetime']} {h['datetime']}"),
                    "Homerseklet_C":h.get("temp"),
                    "Paratartalom_szazalek":h.get("humidity") or 0,
                    "Napsugarzas_W_m2":h.get("solarradiation") or 0,
                    "Szelsebesseg_kmh":h.get("windspeed") or 0,
                    "Csapadek_mm":h.get("precip") or 0})
        hourly = pd.DataFrame(sorok).dropna(subset=["Homerseklet_C"])
        daily = {"max":[n.get("tempmax") for n in d["days"][2:6]],
                 "min":[n.get("tempmin") for n in d["days"][2:6]],
                 "code":[None]*len(d["days"][2:6])}
        return hourly,daily,True
    except Exception as e:
        print(f"[HIBA] Visual Crossing (előrejelzés): {e}", flush=True)
        return None,None,False

def get_idojaras_barmelyik():
    h,dl,ok = get_idojaras()
    if ok: return h,dl,ok,"Open-Meteo"
    h,dl,ok = get_idojaras_vc()
    if ok:
        print("[INFO] Időjárás: Visual Crossing tartalék aktív", flush=True)
        return h,dl,ok,"Visual Crossing"
    return None,None,False,None

def get_dam():
    """DAM árak tegnaptól holnapig. A modell célnapja a holnap, ha már
    publikált; egyébként a ma. A 'Mikor tölts ma?' kártyához a MAI
    negyedórás árgörbe is visszajön (a piac 15 perces felbontású)."""
    if not ENTSOE_API_KEY:
        print("[HIBA] ENTSO-E: nincs API kulcs", flush=True)
        return None,False
    try:
        c = _entsoe(); ma = _ma()
        s = pd.Timestamp((ma-timedelta(days=1)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        e = pd.Timestamp((ma+timedelta(days=2)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        arak = c.query_day_ahead_prices("HU",start=s,end=e)
        if isinstance(arak,pd.DataFrame): arak = arak.iloc[:,0]
        arak = _helyi(arak)
        napok = _naponkent_oras(arak)
        holnap = (ma+timedelta(days=1)).date(); ma_d = ma.date(); tegnap = (ma-timedelta(days=1)).date()
        if ma_d not in napok:
            print("[HIBA] ENTSO-E (DAM): a mai árak nem érhetők el", flush=True)
            return None,False
        if holnap in napok:
            target, target_nap, prev = napok[holnap], "holnap", napok[ma_d]
        else:
            if tegnap not in napok:
                print("[HIBA] ENTSO-E (DAM): tegnapi árak hiányoznak a deltához", flush=True)
                return None,False
            target, target_nap, prev = napok[ma_d], "ma", napok[tegnap]
        # A mai nap natív (negyedórás) felbontású görbéje a hero-kártyához
        mai = arak[arak.index.date == ma_d]
        # A holnapi negyedórás görbe a 2. oldal menetrend-szalagjához (ha már publikált)
        holnapi = arak[arak.index.date == holnap]
        holnap_negyed = ({"ido":[t.isoformat() for t in holnapi.index],
                          "ar":[float(x) for x in holnapi.values]}
                         if len(holnapi) > 0 else None)
        return {"target":target,"prev":prev,"ma":napok[ma_d],"target_nap":target_nap,
                "holnap_negyed":holnap_negyed,
                "ma_negyed":{"ido":[t.isoformat() for t in mai.index],
                             "ar":[float(x) for x in mai.values]}},True
    except Exception as e:
        print(f"[HIBA] ENTSO-E (DAM árak): {e}", flush=True)
        return None,False

def get_load():
    """Elmúlt 17 nap mért fogyasztása (lag336h-hoz) + legutolsó mért érték."""
    if not ENTSOE_API_KEY: return None,False
    try:
        c = _entsoe(); ma = _ma()
        s = pd.Timestamp((ma-timedelta(days=17)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        e = pd.Timestamp(_helyi_most(),tz="Europe/Budapest") + pd.Timedelta(hours=1)
        load = c.query_load("HU",start=s,end=e)
        if isinstance(load,pd.DataFrame): load = load.iloc[:,0]
        load = load.resample('h').mean().dropna()
        load = _helyi(load)
        if len(load) < 15*24:
            print(f"[HIBA] ENTSO-E (fogyasztás): kevés adat ({len(load)} óra)", flush=True)
            return None,False
        return load,True
    except Exception as e:
        print(f"[HIBA] ENTSO-E (fogyasztás): {e}", flush=True)
        return None,False

def get_naposzel_fc():
    """Hivatalos napelőtti nap/szél termelés-előrejelzés naponként
    (célnap + előző nap a deltákhoz)."""
    if not ENTSOE_API_KEY: return None,False
    try:
        c = _entsoe(); ma = _ma()
        s = pd.Timestamp((ma-timedelta(days=1)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        e = pd.Timestamp((ma+timedelta(days=2)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        g = c.query_wind_and_solar_forecast("HU",start=s,end=e)
        g.index = g.index.tz_convert('Europe/Budapest').tz_localize(None)
        napo = [x for x in g.columns if 'Solar' in str(x)]
        szelo = [x for x in g.columns if 'Wind' in str(x)]
        nap_s = g[napo].sum(axis=1) if napo else pd.Series(0.0, index=g.index)
        szel_s = g[szelo].sum(axis=1) if szelo else pd.Series(0.0, index=g.index)
        return {"nap":_naponkent_oras(nap_s),"szel":_naponkent_oras(szel_s)},True
    except Exception as e:
        print(f"[HIBA] ENTSO-E (nap/szél előrejelzés): {e}", flush=True)
        return None,False

# ============================================================
# ELŐREJELZÉS — direkt 24 órás, láncolás nélkül (V10)
# ============================================================
def elorejelez(target_datum, dam, ido_df, load, fcs, eur_huf):
    ido_map = {d: r for d, r in zip(ido_df["Datum"], ido_df.to_dict("records"))}
    load_d = load.to_dict()
    utolso_mert = load.index.max()

    def mert(dt):
        """Mért fogyasztás dt-kor. Ha az az óra még nem mért (jövő),
        ugyanezen óra legutóbbi MÉRT napját adja — dokumentált,
        őszinte kezelés, nem jóslat-a-jóslatra."""
        t = dt
        while t > utolso_mert:
            t -= pd.Timedelta(hours=24)
        while t not in load_d and t > load.index.min():
            t -= pd.Timedelta(hours=24)
        return float(load_d.get(t, float(load.iloc[-1])))

    def same_hour_7(dt):
        vals = []
        t = dt - pd.Timedelta(hours=24)
        while len(vals) < 7 and t >= load.index.min():
            if t <= utolso_mert and t in load_d:
                vals.append(float(load_d[t]))
            t -= pd.Timedelta(hours=24)
        return vals if vals else [float(load.iloc[-1])]

    td = target_datum.date()
    prev_d = (target_datum - timedelta(days=1)).date()
    fc_nap_t = fcs["nap"].get(td); fc_szel_t = fcs["szel"].get(td)
    fc_nap_p = fcs["nap"].get(prev_d, fc_nap_t)
    fc_szel_p = fcs["szel"].get(prev_d, fc_szel_t)
    if fc_nap_t is None:
        fc_nap_t, fc_szel_t = fc_nap_p, fc_szel_p
        print("[INFO] Nap/szél fc: célnapi hiányzik, előző napi perzisztencia", flush=True)
    if fc_nap_t is None:
        raise ValueError("nap/szél előrejelzés nem elérhető")

    sorok = []
    for h in range(24):
        dt = pd.Timestamp(target_datum) + pd.Timedelta(hours=h)
        w = ido_map.get(dt)
        w_prev = ido_map.get(dt - pd.Timedelta(hours=24), w)
        if w is None:
            raise ValueError(f"hiányzó időjárás-óra: {dt}")
        l24 = mert(dt - pd.Timedelta(hours=24))
        l48 = mert(dt - pd.Timedelta(hours=48))
        l168 = mert(dt - pd.Timedelta(hours=168))
        sh = same_hour_7(dt)
        temp = float(w["Homerseklet_C"])
        sorok.append({
            "DAM_EUR_MWh": dam["target"][h],
            "Homerseklet_C": temp,
            "Paratartalom_szazalek": float(w["Paratartalom_szazalek"]),
            "Napsugarzas_W_m2": float(w["Napsugarzas_W_m2"]),
            "Szelsebesseg_kmh": float(w["Szelsebesseg_kmh"]),
            "Csapadek_mm": float(w["Csapadek_mm"]),
            "EUR_HUF": eur_huf,
            "Ora": h, "Het_napja": dt.weekday()+1, "Honap": dt.month,
            "Unnepnap": 1 if dt.date() in hu_holidays else 0,
            "Hetvege": 1 if dt.weekday()>=5 else 0,
            "Extrem_hideg": 1 if temp < -5 else 0,
            "Extrem_meleg": 1 if temp > 30 else 0,
            "Fogyasztas_lag24h": l24,
            "Fogyasztas_lag48h": l48,
            "Fogyasztas_lag72h": mert(dt - pd.Timedelta(hours=72)),
            "Fogyasztas_lag96h": mert(dt - pd.Timedelta(hours=96)),
            "Fogyasztas_lag120h": mert(dt - pd.Timedelta(hours=120)),
            "Fogyasztas_lag144h": mert(dt - pd.Timedelta(hours=144)),
            "Fogyasztas_lag168h": l168,
            "Fogyasztas_lag336h": mert(dt - pd.Timedelta(hours=336)),
            "Fogyasztas_same_hour_mean7": float(np.mean(sh)),
            "Fogyasztas_same_hour_median7": float(np.median(sh)),
            "Fogyasztas_same_hour_min7": float(np.min(sh)),
            "Fogyasztas_same_hour_max7": float(np.max(sh)),
            "Fogyasztas_trend_24_168": l24 - l168,
            "Fogyasztas_trend_24_48": l24 - l48,
            "Homerseklet_delta24": temp - float(w_prev["Homerseklet_C"]),
            "Napsugarzas_delta24": float(w["Napsugarzas_W_m2"]) - float(w_prev["Napsugarzas_W_m2"]),
            "DAM_delta24": dam["target"][h] - dam["prev"][h],
            "Nap_fc_MW": fc_nap_t[h], "Szel_fc_MW": fc_szel_t[h],
            "Nap_fc_delta24": fc_nap_t[h] - (fc_nap_p[h] if fc_nap_p else fc_nap_t[h]),
            "Szel_fc_delta24": fc_szel_t[h] - (fc_szel_p[h] if fc_szel_p else fc_szel_t[h]),
            "Ora_sin": float(np.sin(2*np.pi*h/24)), "Ora_cos": float(np.cos(2*np.pi*h/24)),
            "Het_sin": float(np.sin(2*np.pi*(dt.weekday()+1)/7)),
            "Het_cos": float(np.cos(2*np.pi*(dt.weekday()+1)/7)),
            "Ev_sin": float(np.sin(2*np.pi*dt.dayofyear/365.25)),
            "Ev_cos": float(np.cos(2*np.pi*dt.dayofyear/365.25)),
            "Cooling_degree": max(temp-21, 0.0), "Heating_degree": max(16-temp, 0.0),
            "Cooling_x_hour": max(temp-21, 0.0)*h,
        })
    X = pd.DataFrame(sorok)
    josolt = model_predict(X)
    return [{"ora":h,"datum":(pd.Timestamp(target_datum)+pd.Timedelta(hours=h)).isoformat(),
             "homerseklet":float(X["Homerseklet_C"].iloc[h]),
             "fogyasztas":float(josolt[h]),"dam_ar":float(dam["target"][h]),
             "koltseg_mft":float(josolt[h])*dam["target"][h]*eur_huf/1_000_000}
            for h in range(24)]

# ============================================================
# "MIKOR TÖLTS MA?" — ajánlási logika a mai negyedórás árakból
# ============================================================
def toltes_ajanlas(ma_negyed):
    """A mostantól éjfélig tartó árakból: a legolcsóbb 1 órás ablak,
    2 alternatíva, és az állapot (TÖLTS MOST / NEGATÍV / VÁRJ)."""
    most = _helyi_most()
    idok = [datetime.fromisoformat(t) for t in ma_negyed["ido"]]
    arak = ma_negyed["ar"]
    # negyedórás vagy órás a felbontás?
    lepes_perc = 15 if len(idok) > 30 else 60
    ablak = max(1, 60 // lepes_perc)  # 1 órányi lépés

    jovo = [(t, a) for t, a in zip(idok, arak) if t + timedelta(minutes=lepes_perc) > most]
    if not jovo:
        return None
    # A nap utolsó órájában kevesebb negyedóra marad, mint egy teljes ablak —
    # ilyenkor a maradékkal számolunk, nem tűnik el a kártya éjfélig.
    ablak = min(ablak, len(jovo))
    t_lista = [x[0] for x in jovo]; a_lista = [x[1] for x in jovo]

    # gördülő 1 órás átlagok
    atlagok = [(i, float(np.mean(a_lista[i:i+ablak])))
               for i in range(len(a_lista)-ablak+1)]
    fo_i, fo_ar = min(atlagok, key=lambda x: x[1])
    fo_kezd = t_lista[fo_i]
    fo_veg = fo_kezd + timedelta(hours=1)
    fo_min = float(np.min(a_lista[fo_i:fo_i+ablak]))

    # alternatívák: egész órás kezdetű ablakok, a főtől legalább 2 órára
    altok = []
    for i, atl in sorted(atlagok, key=lambda x: x[1]):
        t = t_lista[i]
        if t.minute != 0: continue
        if abs((t - fo_kezd).total_seconds()) < 2*3600: continue
        if any(abs((t - m).total_seconds()) < 2*3600 for m, _ in altok): continue
        altok.append((t, atl))
        if len(altok) == 2: break

    # negatív blokk (összefüggő, a fő ablak körül/tól)
    negativ = any(a < 0 for a in a_lista[fo_i:fo_i+ablak])
    neg_kezd = neg_veg = None
    if negativ:
        i0 = fo_i
        while i0 > 0 and a_lista[i0-1] < 0: i0 -= 1
        i1 = fo_i + ablak - 1
        while i1 < len(a_lista)-1 and a_lista[i1+1] < 0: i1 += 1
        neg_kezd = t_lista[i0]
        neg_veg = t_lista[i1] + timedelta(minutes=lepes_perc)

    # Most éppen kedvező? A visszaszámlálás mindig az AKTUÁLIS
    # kedvező/negatív blokk végét mutassa, ne egy későbbi blokkét.
    akt_ar = a_lista[0]
    optimalis_most = fo_kezd <= most < fo_veg
    akt_neg_veg = None
    if akt_ar < 0:
        akt_i1 = 0
        while akt_i1 < len(a_lista)-1 and a_lista[akt_i1+1] < 0:
            akt_i1 += 1
        akt_neg_veg = t_lista[akt_i1] + timedelta(minutes=lepes_perc)

    most_jo = optimalis_most or akt_ar < 0
    akt_veg = akt_neg_veg if akt_ar < 0 else (fo_veg if optimalis_most else None)
    hatra_perc = None
    if akt_veg:
        hatra_perc = max(1, math.ceil((akt_veg - most).total_seconds() / 60))

    return {"fo_kezd":fo_kezd.isoformat(),"fo_veg":fo_veg.isoformat(),
            "fo_ar":fo_ar,"fo_min":fo_min,
            "altok":[(t.isoformat(),a) for t,a in altok],
            "negativ":negativ,
            "neg_kezd":neg_kezd.isoformat() if neg_kezd else None,
            "neg_veg":neg_veg.isoformat() if neg_veg else None,
            "most_jo":most_jo,"hatra_perc":hatra_perc,
            "akt_veg":akt_veg.isoformat() if akt_veg else None,
            "akt_ar":float(akt_ar),
            "grafikon":{"ido":[t.isoformat() for t in idok],"ar":arak}}

def visszaszamlalo(cel):
    """'1 óra 24 perc múlva' formátum."""
    delta = math.ceil((cel - _helyi_most()).total_seconds() / 60)
    if delta <= 0: return "most"
    ora, perc = divmod(delta, 60)
    if ora == 0: return f"{perc} perc múlva"
    return f"{ora} óra {perc} perc múlva"

def dam_szin(ar, atlag):
    if ar < 0: return C['gr']
    elif ar < atlag * 0.7: return '#a3e635'
    elif ar > atlag * 1.3: return C['rd']
    return C['yw']

def ido_ikon(code):
    if code is None: return "☀️"
    code = int(code)
    if code == 0: return "☀️"
    elif code <= 3: return "⛅"
    elif code <= 48: return "🌫️"
    elif code <= 67: return "🌧️"
    elif code <= 77: return "❄️"
    else: return "⛈️"

CS = {"background":C['card'],"border":f"1px solid {C['brd']}","borderRadius":"14px","padding":"18px","height":"100%"}

def kpi(cim,val,sub,szin,trend=None):
    ch = []
    if trend:
        fig = go.Figure(data=[go.Scatter(y=trend,mode="lines",line=dict(color=szin,width=2),hoverinfo="skip")],
            layout=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=0,b=0),height=24,
                xaxis=dict(visible=False),yaxis=dict(visible=False),showlegend=False))
        ch = [dcc.Graph(figure=fig,config={"displayModeBar":False},style={"height":"24px","marginTop":"4px"})]
    return html.Div([
        html.Div(cim,className="kpi-label"),
        html.Div(val,className="kpi-value"),
        html.Div(sub,className="kpi-sub",style={"color":szin}),
        *ch
    ],className="kpi-card")

def src_sor(nev,ok):
    return html.Div([
        html.Span(nev,style={"fontSize":"10px","color":C['mut']}),
        html.Span("● Élő" if ok else "○ Nem elérhető",style={"fontSize":"10px","color":C['gr'] if ok else C['rd']})
    ],style={"display":"flex","justifyContent":"space-between","padding":"2px 0"})

def hiba_panel(hianyzo, modell_hiba=None):
    sorok = []
    if modell_hiba:
        sorok.append(html.Div([
            html.Div("⚠ A modell nem tölthető be",style={"fontSize":"14px","fontWeight":"600","color":C['rd']}),
            html.Div(f"Részletek: {modell_hiba}",style={"fontSize":"11px","color":C['mut'],"marginTop":"4px"}),
            html.Div("Ellenőrizd, hogy a catboost_model_final.pkl az app.py mellett van-e.",
                style={"fontSize":"11px","color":C['txt'],"marginTop":"4px"})
        ],style={"marginBottom":"16px"}))
    if hianyzo:
        sorok.append(html.Div([
            html.Div("⚠ Élő adatforrás nem elérhető",style={"fontSize":"14px","fontWeight":"600","color":C['rd']}),
            html.Div(f"Érintett: {', '.join(hianyzo)}",style={"fontSize":"12px","color":C['txt'],"marginTop":"6px"}),
            html.Div("Az alkalmazás kizárólag élő adatokkal működik. Ellenőrizd az API kulcsokat és a "
                     "hálózati kapcsolatot — az oldal 30 percen belül automatikusan újrapróbálkozik.",
                style={"fontSize":"11px","color":C['mut'],"marginTop":"6px"})
        ]))
    return html.Div(sorok,style={"background":C['card'],"border":f"1px solid {C['rd']}",
        "borderRadius":"14px","padding":"24px","maxWidth":"640px"})

def hianyzo_panel(cim, uzenet):
    return html.Div([
        html.Div(cim,style={"fontSize":"11px","fontWeight":"600","color":C['wh'],"marginBottom":"12px"}),
        html.Div([
            html.Div("⚠",style={"fontSize":"22px","color":C['yw'],"textAlign":"center","marginBottom":"8px"}),
            html.Div(uzenet,style={"fontSize":"11px","color":C['mut'],"textAlign":"center"})
        ],style={"display":"flex","flexDirection":"column","justifyContent":"center","flex":"1","padding":"20px 10px"})
    ],style={**CS,"display":"flex","flexDirection":"column"})

HEADER = html.Header([
    html.Div([
        html.Div([
            html.Div("OM",className="brand-mark"),
            html.Div([
                html.Div("OkosMérő",className="brand-name"),
                html.Div("ENERGIAPIACI IRÁNYÍTÓPULT",className="brand-subtitle")
            ])
        ],className="brand-lockup"),
        html.Div([
            html.Span("ÉLŐ ADAT",className="live-badge"),
            html.Div(id="statusz",className="header-status"),
            html.Button("Frissítés",id="manual-refresh",n_clicks=0,className="refresh-button")
        ],className="header-meta")
    ],className="app-header-inner")
],className="app-header")

NAV_TABS = html.Div([
    html.Div("Főoldal",id="nav-fooldal",n_clicks=0,
        style={"color":C['gr'],"borderBottom":f"2px solid {C['gr']}"},
        className="nav-tab"),
    html.Div("Energiaelemzés",id="nav-elemzes",n_clicks=0,
        style={"color":C['mut'],"borderBottom":"2px solid transparent"},
        className="nav-tab"),
    html.Div("ML Modell Labor",id="nav-mllabor",n_clicks=0,
        style={"color":C['mut'],"borderBottom":"2px solid transparent"},
        className="nav-tab")
],className="nav-tabs-row")

app.layout = html.Div([
    HEADER,
    html.Main([
        html.Div(id="kpi-sor",className="kpi-strip"),
        NAV_TABS,
        html.Div(id="oldal-content"),
        html.Div([
            html.Div(id="src-panel"),
            html.Div(id="modell-panel")
        ],style={"display":"none"}),
        dcc.Interval(id="refresh",interval=1800*1000,n_intervals=0),
        dcc.Interval(id="clock",interval=15*1000,n_intervals=0),
        dcc.Store(id="oldal",data="fooldal"),
        dcc.Store(id="adatok",data=None),
    ],className="app-main")
],className="app-shell")

@callback(Output("adatok","data"),
    [Input("refresh","n_intervals"),Input("manual-refresh","n_clicks")],
    running=[
        (Output("manual-refresh","disabled"),True,False),
        (Output("manual-refresh","children"),"Frissítés…","Frissítés"),
    ])
def fetch(n,_manual):
    manual = dash.ctx.triggered_id == "manual-refresh"

    if bundle is None:
        return {"kritikus_hiba":True,"hianyzo":[],"modell_hiba":MODELL_HIBA}

    eur_huf,eur_ok = cachelt("ecb", 6*3600, get_eur_huf, 1, force=manual)
    ido_df,daily,ido_ok,ido_forras = cachelt(
        "idojaras3", 3600, get_idojaras_barmelyik, 2, force=manual)
    dam,dam_ok = cachelt("dam2", 3600, get_dam, 1, force=manual)
    load,load_ok = cachelt("load17", 3600, get_load, 1, force=manual)
    fcs,fc_ok = cachelt("napszelfc", 3600, get_naposzel_fc, 1, force=manual)
    aho,ho_ok = cachelt("homerseklet", 3600, get_ho_barmelyik, 1, force=manual)

    hianyzo = []
    if not dam_ok: hianyzo.append("ENTSO-E (DAM árak)")
    if not load_ok: hianyzo.append("ENTSO-E (fogyasztás)")
    if not fc_ok: hianyzo.append("ENTSO-E (nap/szél előrejelzés)")
    if not ido_ok: hianyzo.append("Időjárás (Open-Meteo és Visual Crossing)")
    if not eur_ok: hianyzo.append("ECB (árfolyam)")

    if not dam_ok:
        return {"kritikus_hiba":True,"hianyzo":hianyzo,"modell_hiba":None}

    ma = _ma()
    target_datum = ma + timedelta(days=1) if dam["target_nap"]=="holnap" else ma

    eredm = None
    if ido_ok and load_ok and eur_ok and fc_ok:
        try:
            eredm = elorejelez(target_datum, dam, ido_df, load, fcs, eur_huf)
        except Exception as e:
            print(f"[HIBA] Előrejelzés: {e}", flush=True)
            hianyzo.append("Előrejelzés")

    # Célnapi nap/szél termelés-sor a 2. oldal grafikonjához
    target_fc = None
    if fc_ok and fcs:
        td_ = target_datum.date()
        if fcs["nap"].get(td_) is not None and fcs["szel"].get(td_) is not None:
            target_fc = {"nap":fcs["nap"][td_],"szel":fcs["szel"][td_]}

    # Óránkénti időjárás-ikonok a célnapra (napsugárzás + csapadék alapján)
    target_ikonok = None
    if ido_ok and ido_df is not None:
        try:
            sor = ido_df[ido_df["Datum"].dt.date == target_datum.date()].iloc[:24]
            if len(sor) >= 24:
                target_ikonok = []
                for _, r_ in sor.iterrows():
                    o_ = int(r_["Datum"].hour)
                    if float(r_["Csapadek_mm"] or 0) > 0.3:
                        target_ikonok.append("☂")
                    elif o_ < 6 or o_ >= 21:
                        target_ikonok.append("☾")
                    elif float(r_["Napsugarzas_W_m2"] or 0) > 120:
                        target_ikonok.append("☀")
                    else:
                        target_ikonok.append("☁")
        except Exception as e:
            print(f"[HIBA] Időjárás-ikonok: {e}", flush=True)

    stl_data = None
    if load_ok:
        try:
            s = load.tail(720) if len(load)>=720 else load
            res = STL(s,period=24,seasonal=25,robust=True).fit()
            std=float(res.resid.std()); mean=float(res.resid.mean()); kuszob=2.5*std
            mask=abs(res.resid-mean)>kuszob
            stl_data={"trend":res.trend.tolist(),"seasonal":res.seasonal.tolist(),
                "residual":res.resid.tolist(),"original":[float(x) for x in s],
                "anomalia_db":int(mask.sum()),
                "stat":{"std":std,"mean":mean,"kuszob":kuszob,
                    "irany":"emelkedő" if res.trend.iloc[-1]>res.trend.iloc[-24] else "csökkenő"}}
        except Exception as e:
            print(f"[HIBA] STL: {e}", flush=True)

    mert = None
    if load_ok:
        mert = {"ertek":float(load.iloc[-1]),"idopont":load.index.max().strftime("%H:%M")}

    heti_atlag = None
    if load_ok:
        try:
            u7 = load.tail(7*24)
            heti_atlag = [float(u7[u7.index.hour == o].mean()) for o in range(24)]
        except Exception as e:
            print(f"[HIBA] Heti átlag: {e}", flush=True)

    return {"kritikus_hiba":False,
        "eredm":eredm,
        "eur_huf":eur_huf if eur_ok else None,
        "dam_target":dam["target"],"dam_ma":dam["ma"],
        "dam_target_nap":dam["target_nap"],
        "ma_negyed":dam["ma_negyed"],
        "holnap_negyed":dam.get("holnap_negyed"),
        "ido_forras":ido_forras if ido_ok else None,
        "aho":aho if ho_ok else None,
        "mert_fogyasztas":mert,
        "heti_atlag":heti_atlag,
        "target_fc":target_fc,
        "target_ikonok":target_ikonok,
        "stl":stl_data,
        "daily":daily if ido_ok else None,
        "frissites":_helyi_most().strftime("%H:%M:%S"),
        "frissites_tipus":"kézi" if manual else "automatikus",
        "fb":{"ENTSO-E":not (dam_ok and load_ok and fc_ok),"Időjárás":not ido_ok,"ECB":not eur_ok},
        "hianyzo":hianyzo}

@callback(Output("oldal","data"),
    [Input(f"nav-{x}","n_clicks") for x in ["fooldal","elemzes","mllabor"]],
    prevent_initial_call=True)
def nav(*_):
    ctx=dash.callback_context
    if not ctx.triggered: return "fooldal"
    return ctx.triggered[0]["prop_id"].split(".")[0].replace("nav-","")

NB = {"display":"flex","alignItems":"center","justifyContent":"center",
    "padding":"14px 12px","cursor":"pointer","transition":"all 0.2s",
    "background":"transparent"}

@callback([Output("statusz","children"),Output("kpi-sor","children"),
    Output("oldal-content","children"),Output("src-panel","children"),
    Output("modell-panel","children"),
    Output("nav-fooldal","style"),Output("nav-elemzes","style"),Output("nav-mllabor","style")],
    [Input("adatok","data"),Input("oldal","data"),Input("clock","n_intervals")])
def render(data,oldal,_clock):
    ns=[{**NB,"color":C['gr'],"borderBottom":f"2px solid {C['gr']}"} if oldal==x
        else {**NB,"color":C['mut'],"borderBottom":"2px solid transparent"}
        for x in ["fooldal","elemzes","mllabor"]]

    mk = (bundle or {}).get("metrikak",{})
    modell_info = html.Div([
        html.Div("Modell",style={"fontSize":"9px","color":C['cy'],"fontWeight":"bold"}),
        html.Div("CatBoost V10 — direkt 24h",style={"fontSize":"10px","color":C['mut']}),
        html.Div(f"MAE {mk.get('mae',0):.1f} MWh | MAPE {mk.get('mape',0):.2f}%"
                 if bundle else "Modell nem elérhető",
            style={"fontSize":"9px","color":C['gr'] if bundle else C['rd'],"marginTop":"2px"})
    ])

    if data is None:
        return (html.Div("Élő adatok betöltése...",style={"color":C['yw']}),
            html.Div(),html.Div(),html.Div(),modell_info,*ns)

    if data.get("kritikus_hiba"):
        statusz = html.Div([
            html.Span("● ",style={"color":C['rd']}),
            html.Span("Éles adatkapcsolat megszakadt",style={"fontSize":"12px","color":C['rd'],"fontWeight":"500"})
        ])
        src = html.Div([src_sor(k, k not in " ".join(data.get("hianyzo",[])))
                        for k in ["ENTSO-E","Időjárás","ECB"]])
        return (statusz,html.Div(),
            hiba_panel(data.get("hianyzo",[]),data.get("modell_hiba")),
            src,modell_info,*ns)

    dam_ma = data["dam_ma"]; dam_target = data["dam_target"]
    edf = pd.DataFrame(data["eredm"]) if data.get("eredm") else None
    eur_huf=data.get("eur_huf"); aho=data.get("aho")
    fb=data["fb"]; hianyzo=data.get("hianyzo",[])

    stl_db=data["stl"]["anomalia_db"] if data["stl"] else 0
    stl_tot=len(data["stl"]["trend"]) if data["stl"] else 0

    ora=_helyi_most().hour
    ma_atlag = float(np.mean(dam_ma))
    ma_negyed = data.get("ma_negyed")
    aj = toltes_ajanlas(ma_negyed) if ma_negyed else None
    dam_most = float(aj["akt_ar"]) if aj and aj.get("akt_ar") is not None else (
        dam_ma[ora] if ora < len(dam_ma) else ma_atlag)
    dam_sz = dam_szin(dam_most, ma_atlag)

    legolcs_txt = "–"
    if aj:
        fk = datetime.fromisoformat(aj["fo_kezd"])
        legolcs_txt = f"{fk:%H:%M} – {aj['fo_ar']:.0f} €"

    nap_cimke = "Holnap" if data["dam_target_nap"]=="holnap" else "Ma"
    dam_cimke = "Holnapi árak elérhetők" if data["dam_target_nap"]=="holnap" else "Holnapi árak publikálás előtt"
    ido_txt = f" — Időjárás: {data['ido_forras']}" if data.get("ido_forras") else ""

    if hianyzo:
        statusz=html.Div([
            html.Span("● ",style={"color":C['yw']}),
            html.Span(f"Részleges élő adatok — nem elérhető: {', '.join(hianyzo)} — "
                f"frissítve: {data['frissites']} (Budapest)",
                style={"fontSize":"12px","color":C['yw'],"fontWeight":"500"})
        ])
    else:
        statusz=html.Div([
            html.Span("● ",style={"color":C['gr']}),
            html.Span(f"Élő adatok frissítve: {data['frissites']} (Budapest) — {dam_cimke}{ido_txt}",
                style={"fontSize":"12px","color":C['txt'],"fontWeight":"500"})
        ])

    src=html.Div([src_sor(k,not v) for k,v in fb.items()])

    kat = "Negatív" if dam_most<0 else ("Olcsó" if dam_most<ma_atlag*0.7 else ("Drága" if dam_most>ma_atlag*1.3 else "Átlagos"))
    mert = data.get("mert_fogyasztas")
    ksor=html.Div([
        kpi("Jelenlegi DAM ár",f"{dam_most:.0f} €/MWh",kat,dam_sz,dam_ma),
        kpi("Mért fogyasztás",
            f"{mert['ertek']:,.0f} MWh" if mert else "–",
            f"ENTSO-E mérés, {mert['idopont']}" if mert else "Nem elérhető",
            C['bl'], edf["fogyasztas"].tolist() if edf is not None else None),
        kpi("Budapest",f"{aho:.0f} °C" if aho is not None else "–","Most",C['yw']),
        kpi("EUR/HUF",f"{eur_huf:.1f} Ft" if eur_huf is not None else "–","Árfolyam",C['txt']),
        kpi("Legolcsóbb ma",legolcs_txt,"Hátralévő órákból",C['gr']),
        kpi("Adatminőség-őr",f"{stl_db} / {stl_tot}" if data["stl"] else "–",
            "Anomália, elmúlt 30 nap mérés",C['or'] if stl_db > 0 else C['gr']),
    ],className="kpi-grid")

    if oldal=="fooldal":
        page=fooldal(data,aj)
    elif oldal=="elemzes":
        page=elemzes(dam_target,edf,data,float(np.mean(dam_target)),nap_cimke)
    else:
        page=mllabor(data)
    return statusz,ksor,page,src,modell_info,*ns

# ============================================================
# FŐOLDAL — egyetlen hero-kártya: "Mikor tölts ma?"
# ============================================================
def fooldal(data,aj):
    if not aj:
        return hianyzo_panel("MIKOR TÖLTS MA?",
            "A mai napból nincs hátra értékelhető időszak — éjfél után frissül.")

    most = _helyi_most()
    fo_kezd = datetime.fromisoformat(aj["fo_kezd"])
    fo_veg = datetime.fromisoformat(aj["fo_veg"])
    toltheto = bool(aj["most_jo"])

    if aj["negativ"] and aj.get("neg_kezd") and aj.get("neg_veg"):
        ajanlott_kezd = datetime.fromisoformat(aj["neg_kezd"])
        ajanlott_veg = datetime.fromisoformat(aj["neg_veg"])
        ajanlott_ar = float(aj["fo_min"])
    else:
        ajanlott_kezd = fo_kezd
        ajanlott_veg = fo_veg
        ajanlott_ar = float(aj["fo_ar"])

    def hatralevo(cel):
        perc = max(0, math.ceil((cel - most).total_seconds() / 60))
        ora, perc = divmod(perc, 60)
        return f"{ora} óra {perc} perc" if ora else f"{perc} perc"

    if toltheto:
        aktiv_veg = datetime.fromisoformat(aj["akt_veg"])
        dontes = "IGEN"
        dontes_seged = "Még"
        dontes_ido = hatralevo(aktiv_veg)
        idoszak_cimke = "Kedvező időszak vége"
        idoszak = f"{aktiv_veg:%H:%M}"
        dontes_ar = float(aj["akt_ar"])
    else:
        dontes = "NEM"
        dontes_seged = "Várj még"
        dontes_ido = hatralevo(ajanlott_kezd)
        idoszak_cimke = "Következő optimális időszak"
        idoszak = f"{ajanlott_kezd:%H:%M} – {ajanlott_veg:%H:%M}"
        dontes_ar = ajanlott_ar

    terv = [(ajanlott_kezd, ajanlott_veg, ajanlott_ar)]
    for ido, ar in aj.get("altok", []):
        kezd = datetime.fromisoformat(ido)
        terv.append((kezd, kezd + timedelta(hours=1), float(ar)))

    terv_sorok = [
        html.Div([
            html.Span(f"{kezd:%H:%M} – {veg:%H:%M}",className="charge-plan-time"),
            html.Span(f"{ar:.0f} €/MWh",className="charge-plan-price")
        ],className="charge-plan-row")
        for kezd,veg,ar in terv[:3]
    ]

    # --- Mai DAM oszlopdiagram: a negatív árak a nullavonal alá futnak ---
    g = aj["grafikon"]
    idok = [datetime.fromisoformat(t) for t in g["ido"]]
    arak = [float(a) for a in g["ar"]]
    most_i = int(np.argmin([abs((t-most).total_seconds()) for t in idok]))
    arak[most_i] = float(aj["akt_ar"])

    def ar_szin(ar):
        if ar < -10:
            return "#00bfae"
        if ar < 0:
            return "#00e0c2"
        if ar < 50:
            return "#c9df16"
        if ar < 100:
            return "#ff9800"
        return "#ff3b30"

    fig = go.Figure()
    slot_ms = (abs((idok[1]-idok[0]).total_seconds())*1000
               if len(idok)>1 else 60*60*1000)
    bar_width = slot_ms * 0.66
    glow_width = slot_ms * 0.94
    pozitiv_y = [max(a,0.0) for a in arak]
    negativ_y = [min(a,0.0) for a in arak]

    if any(a < 0 for a in arak):
        fig.add_trace(go.Bar(x=idok,y=negativ_y,width=glow_width,
            marker=dict(color="rgba(0,224,194,0.16)",line=dict(width=0)),
            hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Bar(x=idok,y=[a if a<0 else 0 for a in arak],width=bar_width,
        marker=dict(color=[ar_szin(a) for a in arak],line=dict(width=0)),
        customdata=arak,
        hovertemplate="%{x|%H:%M}<br>%{customdata:.0f} €/MWh<extra></extra>",
        showlegend=False))
    fig.add_trace(go.Bar(x=idok,y=pozitiv_y,width=bar_width,
        marker=dict(color=[ar_szin(a) for a in arak],line=dict(width=0)),
        customdata=arak,
        hovertemplate="%{x|%H:%M}<br>%{customdata:.0f} €/MWh<extra></extra>",
        showlegend=False))

    fig.add_trace(go.Scatter(x=idok,y=arak,mode="markers",
        marker=dict(size=12,color="rgba(0,0,0,0)"),
        hovertemplate="%{x|%H:%M}<br>%{y:.0f} €/MWh<extra></extra>"))
    fig.add_hline(y=0,line=dict(color=C['mut'],width=1))
    fig.add_shape(type="line",x0=most,x1=most,y0=0,y1=1,yref="paper",
        line=dict(color=C['wh'],width=1,dash="dot"))
    fig.add_trace(go.Scatter(x=[idok[most_i]],y=[arak[most_i]],mode="markers",
        marker=dict(size=9,color=C['wh'],line=dict(width=2,color=ar_szin(arak[most_i]))),
        hoverinfo="skip",showlegend=False))
    fig.add_annotation(x=idok[most_i],y=arak[most_i],
        text=f"<b>{arak[most_i]:.0f} €</b><br>{idok[most_i]:%H:%M}",
        showarrow=True,arrowhead=0,ax=34,ay=-42,
        bgcolor="#0f1d31",bordercolor=C['brd'],borderwidth=1,borderpad=7,
        font=dict(color=C['wh'],size=10))
    ar_min = min(arak)
    ar_max = max(arak)
    ar_tartomany = max(ar_max - ar_min, 1.0)
    y_padding = max(5.0, ar_tartomany * 0.08)
    y_min = min(0.0, ar_min - y_padding) if ar_min < 0 else 0.0
    y_max = max(0.0, ar_max + y_padding)
    if y_max <= y_min:
        y_max = y_min + 10.0
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=C['mut'],family='Inter,sans-serif',size=10),
        margin=dict(l=48,r=18,t=12,b=30),height=200,showlegend=False,
        hovermode="closest",barmode="overlay",bargap=0.18,
        xaxis=dict(type="date",showgrid=False,color=C['mut'],
            tickformat="%H:%M",dtick=3*60*60*1000,fixedrange=True,
            range=[idok[0],idok[0]+timedelta(days=1)]),
        yaxis=dict(gridcolor=C['brd'],color=C['mut'],
            zeroline=False,fixedrange=True,range=[y_min,y_max]))

    state_class = "is-negative" if float(aj["akt_ar"]) < 0 else (
        "is-ready" if toltheto else "is-waiting")
    egy_soros_idoszak = f"{idoszak} · {dontes_ar:.0f} €/MWh"

    return html.Div([
        html.Div([
            html.Div([
                html.Div("MIKOR TÖLTS MA?",className="charge-eyebrow"),
                html.Div([
                    html.Div([
                        html.Img(src="/assets/sports-coupe-board.png",
                            alt="Elektromos sportautó",className="charge-car-source"),
                        html.Div(className="charge-headlight charge-headlight-bal"),
                        html.Div(className="charge-headlight charge-headlight-jobb"),
                        html.Div(className="charge-floor-glow")
                    ],className="charge-car-stage")
                ],className="charge-car-frame")
            ],className=f"charge-car-showcase {state_class}"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(dontes,className="charge-decision"),
                        html.Div(dontes_seged,className="charge-decision-helper"),
                        html.Div(dontes_ido,className="charge-countdown")
                    ],className="charge-gauge-face")
                ],className=f"charge-gauge {state_class}"),
                html.Div([
                    html.Div(idoszak_cimke,className="charge-next-label"),
                    html.Div(egy_soros_idoszak,className="charge-next-period")
                ],className="charge-next"),
                html.Details([
                    html.Summary("Töltési terv",className="charge-plan-button"),
                    html.Div(terv_sorok,className="charge-plan-list")
                ],className="charge-plan")
            ],className=f"charge-decision-panel {state_class}")
        ],className="charge-summary"),
        html.Div([
            html.Div([
                html.Div("MAI DAM ÁRAK",className="charge-chart-title"),
                html.Div("€/MWh",className="charge-unit")
            ],className="charge-chart-header"),
            dcc.Graph(figure=fig,config={"displayModeBar":False},
                className="charge-chart-graph"),
            html.Div([
                html.Div("< –10 €",className="price-band price-band-deep-negative"),
                html.Div("–10 – 0 €",className="price-band price-band-negative"),
                html.Div("0 – 50 €",className="price-band price-band-low"),
                html.Div("50 – 100 €",className="price-band price-band-medium"),
                html.Div("> 100 €",className="price-band price-band-high")
            ],className="charge-price-scale")
        ],className="charge-chart")
    ],className="charge-hero-card")

def elemzes(dam_target,edf,data,target_atlag,nap_cimke):
    orak=[f"{h:02d}:00" for h in range(24)]

    if edf is not None:
        y = [float(v) for v in edf["fogyasztas"].tolist()]
        y_lo, y_hi = min(y), max(y)

        def terheles_szin(v):
            """A terhelés szintje szerinti szín: hűvös kék → sárga → izzó piros."""
            t = 0.0 if y_hi == y_lo else (v - y_lo) / (y_hi - y_lo)
            allomasok = [(0.0,(47,127,214)),(0.35,(77,163,255)),(0.6,(230,223,0)),
                         (0.8,(255,152,0)),(1.0,(255,59,48))]
            for (t0,c0),(t1,c1) in zip(allomasok, allomasok[1:]):
                if t <= t1:
                    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                    return (f"rgb({int(c0[0]+(c1[0]-c0[0])*a)},"
                            f"{int(c0[1]+(c1[1]-c0[1])*a)},"
                            f"{int(c0[2]+(c1[2]-c0[2])*a)})")
            return "rgb(255,59,48)"

        fig = go.Figure()
        # napszak-zónák a háttérben, felirattal
        for zx0,zx1,zszin,znev,ztszin in [
                (0,6,"#0a1a33","Éjszaka","#4b6a94"),
                (6,9,"#10233c","Reggel","#6f8fb8"),
                (9,16,"#0d1f30","Napközben","#7fa3c4"),
                (16,21,"#2a1a0e","Esti csúcs","#d99a5b"),
                (21,23.99,"#0a1a33","Éjszaka","#4b6a94")]:
            fig.add_vrect(x0=zx0,x1=zx1,fillcolor=zszin,opacity=0.5,line_width=0,layer="below")
            fig.add_annotation(x=(zx0+zx1)/2,y=1.07,yref="paper",text=znev,
                showarrow=False,font=dict(size=10,color=ztszin))
        # neon-izzás: több áttetsző réteg + éles fő vonal, szakaszonként színezve
        for i in range(len(y)-1):
            szin = terheles_szin((y[i]+y[i+1])/2)
            for w,op in [(12,0.05),(8,0.09),(5,0.15)]:
                fig.add_trace(go.Scatter(x=[i,i+1],y=[y[i],y[i+1]],mode="lines",
                    line=dict(color=szin,width=w),opacity=op,
                    hoverinfo="skip",showlegend=False))
            fig.add_trace(go.Scatter(x=[i,i+1],y=[y[i],y[i+1]],mode="lines",
                line=dict(color=szin,width=2.6),
                hoverinfo="skip",showlegend=False))
        # hover-pontok óránként
        fig.add_trace(go.Scatter(x=list(range(len(y))),y=y,mode="markers",
            marker=dict(size=10,color="rgba(0,0,0,0)"),
            hovertemplate="%{x}:00<br>%{y:,.0f} MWh<extra></extra>",showlegend=False))
        # csúcs és minimum kiemelése
        i_max = int(np.argmax(y)); i_min = int(np.argmin(y))
        fig.add_trace(go.Scatter(x=[i_max],y=[y[i_max]],mode="markers",
            marker=dict(size=9,color=C['wh'],line=dict(width=2,color="#ff3b30")),
            hoverinfo="skip",showlegend=False))
        fig.add_annotation(x=i_max,y=y[i_max],
            text=f"<b>Csúcs: {i_max:02d}:00 — {y[i_max]:,.0f} MWh</b>".replace(","," "),
            showarrow=True,arrowhead=0,ax=-64,ay=-36,bgcolor="#0f1d31",
            bordercolor="#ff3b30",borderwidth=1,borderpad=6,
            font=dict(color=C['wh'],size=11))
        fig.add_trace(go.Scatter(x=[i_min],y=[y[i_min]],mode="markers",
            marker=dict(size=8,color=C['wh'],line=dict(width=2,color="#4da3ff")),
            hoverinfo="skip",showlegend=False))
        fig.add_annotation(x=i_min,y=y[i_min],
            text=f"Minimum: {i_min:02d}:00 — {y[i_min]:,.0f} MWh".replace(","," "),
            showarrow=True,arrowhead=0,ax=64,ay=36,bgcolor="#0f1d31",
            bordercolor="#2a3a4c",borderwidth=1,borderpad=5,
            font=dict(color="#94a3b8",size=10))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=C['mut'],family='Inter,sans-serif',size=10),
            margin=dict(l=46,r=16,t=34,b=30),height=300,showlegend=False,
            xaxis=dict(range=[0,len(y)-1],tickvals=list(range(0,24,3)),
                ticktext=[f"{x:02d}:00" for x in range(0,24,3)],
                gridcolor="#101f35",color=C['mut'],zeroline=False,fixedrange=True),
            yaxis=dict(title="MWh",gridcolor="#101f35",color=C['mut'],
                zeroline=False,fixedrange=True))
        # kontextus-sor: a heti azonos órás átlaghoz viszonyítva
        heti = data.get("heti_atlag")
        if heti and all(v == v for v in heti):
            d = (float(np.mean(y)) / float(np.mean(heti)) - 1) * 100
            irany = "magasabb" if d >= 0 else "alacsonyabb"
            kontextus = (f"Az elmúlt 7 nap átlagánál ~{abs(d):.0f}%-kal {irany} "
                         f"fogyasztás várható · Delta V10 · CatBoost ML")
        else:
            kontextus = "Delta V10 · CatBoost ML"
        if nap_cimke == "Ma":
            kontextus += " · a holnapi előrejelzés a ~14:00-s DAM-aukció után készül el"
        fogy_panel=html.Div([
            html.Div(f"{nap_cimke.upper()}I FOGYASZTÁS-ELŐREJELZÉS",
                style={"fontSize":"13px","fontWeight":"700","color":C['wh']}),
            html.Div(kontextus,style={"fontSize":"11px","color":"#94a3b8",
                "marginTop":"3px","marginBottom":"6px"}),
            dcc.Graph(figure=fig,config={"displayModeBar":False},
                style={"height":"300px"})
        ],style=CS)
    else:
        fogy_panel=hianyzo_panel("Fogyasztás-előrejelzés",
            "Az előrejelzéshez szükséges élő adatforrások egyike jelenleg nem elérhető. "
            "A panel automatikusan megjelenik, amint minden forrás él.")

    # ================= HŐMÉRSÉKLET — izzó pontokkal =================
    if edf is not None:
        tmp = [float(v) for v in edf["homerseklet"].tolist()]
        t_lo, t_hi = min(tmp), max(tmp)
        fig_t = go.Figure()
        # izzás: több áttetsző pont-réteg a fő pontok alatt
        for meret, opac in [(22,0.08),(15,0.14),(10,0.25)]:
            fig_t.add_trace(go.Scatter(x=list(range(24)),y=tmp,mode="markers",
                marker=dict(size=meret,color="#4da3ff",opacity=opac),
                hoverinfo="skip",showlegend=False))
        fig_t.add_trace(go.Scatter(x=list(range(24)),y=tmp,mode="lines",
            line=dict(color="#4da3ff",width=2),hoverinfo="skip",showlegend=False))
        fig_t.add_trace(go.Scatter(x=list(range(24)),y=tmp,mode="markers+text",
            marker=dict(size=6,color="#bfe0ff",line=dict(width=1.4,color="#4da3ff")),
            text=[f"{tmp[i]:.0f}°" if i % 2 == 0 else "" for i in range(24)],
            textposition="top center",textfont=dict(size=10,color=C['wh']),
            hovertemplate="%{x}:00<br>%{y:.1f} °C<extra></extra>",showlegend=False))
        ikonok = data.get("target_ikonok")
        if ikonok and len(ikonok) >= 24:
            iy = t_lo - max((t_hi-t_lo)*0.30, 2.0)
            fig_t.add_trace(go.Scatter(x=list(range(0,24,3)),y=[iy]*8,mode="text",
                text=[ikonok[i] for i in range(0,24,3)],
                textfont=dict(size=18,color=[
                    "#f59e0b" if ikonok[i]=="☀" else "#94a3b8"
                    for i in range(0,24,3)]),
                hoverinfo="skip",showlegend=False))
        fig_t.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=C['mut'],family='Inter,sans-serif',size=10),
            margin=dict(l=36,r=14,t=18,b=28),height=250,showlegend=False,
            xaxis=dict(range=[-0.6,23.6],tickvals=list(range(0,24,3)),
                ticktext=[f"{x:02d}:00" for x in range(0,24,3)],
                gridcolor="#101f35",color=C['mut'],zeroline=False,fixedrange=True),
            yaxis=dict(range=[t_lo-max((t_hi-t_lo)*0.55,3.5), t_hi+max((t_hi-t_lo)*0.30,2.5)],
                gridcolor="#101f35",color=C['mut'],zeroline=False,fixedrange=True))
        homerseklet_resz = [
            html.Div(f"{nap_cimke.upper()}I HŐMÉRSÉKLET (°C)",
                style={"fontSize":"13px","fontWeight":"700","color":C['wh']}),
            html.Div("A Delta V10 kulcs-feature-je: hűtési igény",
                style={"fontSize":"11px","color":"#94a3b8","marginTop":"3px"}),
            dcc.Graph(figure=fig_t,config={"displayModeBar":False},
                style={"height":"250px"})
        ]
    else:
        homerseklet_resz = [html.Div(
            "Az órás hőmérséklet-előrejelzés jelenleg nem elérhető.",
            style={"fontSize":"11px","color":C['mut']})]

    # ================= MEGÚJULÓ TERMELÉS — modern nap/szél panel =================
    tfc = data.get("target_fc")
    if tfc and tfc.get("nap") and tfc.get("szel"):
        nap_mw = [float(v) for v in tfc["nap"]]
        szel_mw = [float(v) for v in tfc["szel"]]
        x = list(range(24))

        nap_max_i = int(np.argmax(nap_mw))
        szel_max_i = int(np.argmax(szel_mw))
        nap_max = float(max(nap_mw))
        szel_max = float(max(szel_mw))
        nap_mwh = float(np.sum(nap_mw))
        szel_mwh = float(np.sum(szel_mw))
        mix_total = max(nap_mwh + szel_mwh, 1.0)
        nap_resz = nap_mwh / mix_total * 100
        szel_resz = szel_mwh / mix_total * 100

        fig_r = make_subplots(rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.07, row_heights=[0.58,0.42])

        # háttér napszak-zónák mindkét panelen
        for row in [1, 2]:
            for x0, x1, szin in [(0,6,"#071426"),(6,9,"#10233c"),(9,16,"#1f2718"),(16,21,"#261a10"),(21,24,"#071426")]:
                fig_r.add_vrect(x0=x0, x1=x1, fillcolor=szin, opacity=0.32,
                    line_width=0, layer="below", row=row, col=1)

        # NAPENERGIA: glow + terület + fő vonal
        fig_r.add_trace(go.Scatter(x=x, y=nap_mw, mode="lines",
            line=dict(color="rgba(255,183,77,0.18)", width=14),
            hoverinfo="skip", showlegend=False), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=x, y=nap_mw, mode="lines",
            name="Napenergia", line=dict(color="#ffb74d", width=3.2),
            fill="tozeroy", fillcolor="rgba(255,183,77,0.22)",
            hovertemplate="%{x:02d}:00<br>%{y:,.0f} MW<extra>Napenergia</extra>"), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=[nap_max_i], y=[nap_max], mode="markers",
            marker=dict(size=15, color="#fff7cc", line=dict(width=3, color="#ffb74d")),
            hoverinfo="skip", showlegend=False), row=1, col=1)
        fig_r.add_annotation(x=nap_max_i, y=nap_max,
            text=f"<b>Nap csúcs</b><br>{nap_max:,.0f} MW".replace(","," "),
            showarrow=True, arrowhead=0, ax=58, ay=-28,
            bgcolor="#101b2d", bordercolor="#ffb74d", borderwidth=1, borderpad=6,
            font=dict(color=C['wh'], size=10), row=1, col=1)

        # SZÉLENERGIA: neon vonal + terület
        fig_r.add_trace(go.Scatter(x=x, y=szel_mw, mode="lines",
            line=dict(color="rgba(77,208,225,0.16)", width=13),
            hoverinfo="skip", showlegend=False), row=2, col=1)
        fig_r.add_trace(go.Scatter(x=x, y=szel_mw, mode="lines+markers",
            name="Szélenergia", line=dict(color="#4dd0e1", width=2.8),
            marker=dict(size=5, color="#b2f5ff", line=dict(width=1.5, color="#4dd0e1")),
            fill="tozeroy", fillcolor="rgba(77,208,225,0.18)",
            hovertemplate="%{x:02d}:00<br>%{y:,.0f} MW<extra>Szélenergia</extra>"), row=2, col=1)

        # Minimalista szélturbina-markerek: nem minden pontra, csak ritmusosan + csúcspont
        szel_range = max(max(szel_mw) - min(szel_mw), 1.0)
        shaft = max(szel_range * 0.11, max(szel_mw) * 0.035, 20.0)
        blade_y = max(szel_range * 0.065, 12.0)
        blade_x = 0.23
        turbina_idx = sorted(set([3,6,9,12,15,18,21, szel_max_i]))
        for i in turbina_idx:
            y0 = szel_mw[i]
            y1 = y0 + shaft
            # oszlop
            fig_r.add_shape(type="line", x0=i, x1=i, y0=y0, y1=y1,
                line=dict(color="#d9fbff", width=1.4), row=2, col=1)
            # három lapát
            fig_r.add_shape(type="line", x0=i, x1=i+blade_x, y0=y1, y1=y1+blade_y,
                line=dict(color="#d9fbff", width=1.25), row=2, col=1)
            fig_r.add_shape(type="line", x0=i, x1=i-blade_x, y0=y1, y1=y1+blade_y*0.55,
                line=dict(color="#d9fbff", width=1.25), row=2, col=1)
            fig_r.add_shape(type="line", x0=i, x1=i, y0=y1, y1=y1-blade_y*0.9,
                line=dict(color="#d9fbff", width=1.25), row=2, col=1)
            fig_r.add_trace(go.Scatter(x=[i], y=[y1], mode="markers",
                marker=dict(size=5, color="#ffffff", line=dict(width=1, color="#4dd0e1")),
                hoverinfo="skip", showlegend=False), row=2, col=1)

        fig_r.add_annotation(x=szel_max_i, y=szel_mw[szel_max_i] + shaft,
            text=f"<b>Szél csúcs</b><br>{szel_max:,.0f} MW".replace(","," "),
            showarrow=True, arrowhead=0, ax=-58, ay=-28,
            bgcolor="#101b2d", bordercolor="#4dd0e1", borderwidth=1, borderpad=6,
            font=dict(color=C['wh'], size=10), row=2, col=1)

        fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=C['mut'], family='Inter,sans-serif', size=10),
            margin=dict(l=48, r=18, t=20, b=34), height=390, showlegend=False,
            hovermode="x unified")
        fig_r.update_xaxes(range=[-0.4,23.4], tickvals=list(range(0,24,3)),
            ticktext=[f"{h:02d}:00" for h in range(0,24,3)],
            gridcolor="#101f35", color=C['mut'], zeroline=False, fixedrange=True, row=2, col=1)
        fig_r.update_xaxes(showticklabels=False, gridcolor="#101f35", color=C['mut'],
            zeroline=False, fixedrange=True, row=1, col=1)
        fig_r.update_yaxes(title="Nap MW", gridcolor="#101f35", color=C['mut'],
            zeroline=False, fixedrange=True, row=1, col=1)
        fig_r.update_yaxes(title="Szél MW", gridcolor="#101f35", color=C['mut'],
            zeroline=False, fixedrange=True, row=2, col=1)

        termeles_resz = [
            html.Div([
                html.Div([
                    html.Div("MEGÚJULÓ ENERGIA", style={"fontSize":"13px","fontWeight":"700","color":C['wh']}),
                    html.Div(f"{nap_cimke} · hivatalos napelőtti nap/szél előrejelzés",
                        style={"fontSize":"11px","color":"#94a3b8","marginTop":"3px"})
                ]),
                html.Div([
                    html.Div([html.Span("☀", style={"color":"#ffb74d","marginRight":"6px"}),
                              html.Span(f"Nap {nap_resz:.0f}% · {nap_max:,.0f} MW csúcs".replace(","," "))],
                        className="renewable-pill"),
                    html.Div([html.Span("◇", style={"color":"#4dd0e1","marginRight":"6px"}),
                              html.Span(f"Szél {szel_resz:.0f}% · {szel_max:,.0f} MW csúcs".replace(","," "))],
                        className="renewable-pill")
                ], style={"display":"flex","gap":"8px","flexWrap":"wrap"})
            ], style={"display":"flex","justifyContent":"space-between","gap":"12px","alignItems":"flex-start","marginBottom":"8px"}),
            dcc.Graph(figure=fig_r, config={"displayModeBar":False}, style={"height":"390px"})
        ]
    else:
        termeles_resz = [html.Div(
            "A nap/szél termelés-előrejelzés jelenleg nem elérhető.",
            style={"fontSize":"11px","color":C['mut'],"marginTop":"16px"})]

    # 2. oldal: Fogyasztás + Időjárás felül, Megújuló energia alul.
    idojaras_panel = html.Div(homerseklet_resz, style=CS)
    megujulo_panel = html.Div(termeles_resz, style=CS)

    return html.Div([
        html.Div("Energiaelemzés",style={"fontSize":"16px","fontWeight":"600","color":C['wh'],"marginBottom":"14px"}),
        dbc.Row([
            dbc.Col(fogy_panel, lg=8, md=12),
            dbc.Col(idojaras_panel, lg=4, md=12),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(megujulo_panel, md=12),
        ], className="g-3")
    ])

def mllabor(data):
    mk = (bundle or {}).get("metrikak",{})
    fi_n=[]; fi_v=[]
    try:
        fi = MODEL.feature_importances_
        srt = sorted(zip(FEATURES, fi), key=lambda x:x[1])
        fi_n=[x[0] for x in srt[-12:]]; fi_v=[float(x[1]) for x in srt[-12:]]
    except Exception: pass
    fig_fi=go.Figure()
    if fi_n:
        fig_fi.add_trace(go.Bar(x=fi_v,y=fi_n,orientation="h",
            marker=dict(color=[C['or'] if v==max(fi_v) else C['bl'] for v in fi_v],opacity=0.85),
            hovertemplate="%{y}<br>%{x:.2f}<extra></extra>"))
    lay_fi=dict(**CHART); lay_fi["height"]=320; lay_fi["margin"]=dict(l=220,r=20,t=35,b=40)
    lay_fi["title"]=dict(text="Feature importance (CatBoost V10, top 12)",font=dict(size=12,color=C['wh']))
    fig_fi.update_layout(**lay_fi)

    if data["stl"]:
        stl=data["stl"]; idx=list(range(len(stl["trend"])))
        fig_stl=go.Figure()
        for vals,name,color,dash in [
            (stl["original"],"Eredeti",C['txt'],"solid"),
            (stl["trend"],"Trend",C['or'],"solid"),
            (stl["seasonal"],"Szezonális",C['gr'],"solid"),
            (stl["residual"],"Maradék",C['bl'],"dot")]:
            fig_stl.add_trace(go.Scatter(x=idx,y=vals,name=name,mode="lines",
                line=dict(color=color,width=1.5,dash=dash)))
        k=stl["stat"]["kuszob"]; a=stl["stat"]["mean"]
        fig_stl.add_hline(y=a+k,line=dict(color=C['rd'],width=1,dash="dash"))
        fig_stl.add_hline(y=a-k,line=dict(color=C['rd'],width=1,dash="dash"))
        lay_s=dict(**CHART); lay_s["height"]=300; lay_s["showlegend"]=True
        lay_s["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=C['txt']))
        lay_s["title"]=dict(text="Adatminőség-őr: STL dekompozíció az elmúlt ~17 nap MÉRT fogyasztásán "
                                 "(anomália-figyelés, a predikciótól független)",font=dict(size=11,color=C['wh']))
        fig_stl.update_layout(**lay_s)
        stl_panel=html.Div([dcc.Graph(figure=fig_stl,config={"displayModeBar":False})],style=CS)
    else:
        stl_panel=hianyzo_panel("STL dekompozíció","Kevés élő mérési adat az elemzéshez.")

    info=[
        ("Modell","CatBoost V10 — direkt 24h",C['wh']),
        ("Validált MAE (720h teszt)",f"{mk.get('mae',0):.2f} MWh",C['gr']),
        ("MAPE",f"{mk.get('mape',0):.2f}%",C['gr']),
        ("R²",f"{mk.get('r2',0):.4f}",C['gr']),
        ("MAVIR benchmark (u.azon teszt)",f"{mk.get('mavir_benchmark_mae',0):.1f} MWh",C['yw']),
        ("Gördülő validáció (3 ablak)",f"{mk.get('valid_3ablak_atlag',0):.1f} MWh",C['bl']),
        ("Feature-ök",f"{len(FEATURES) if bundle else 0} — szivárgásmentes",C['txt']),
        ("Mintasúlyozás","exponenciális, 2 év felezési idő",C['txt']),
        ("Élő adatforrás","ENTSO-E + Open-Meteo/VC + ECB",C['txt']),
    ]

    return html.Div([
        html.Div("Gépi Tanulás Modell Labor",style={"fontSize":"16px","fontWeight":"600","color":C['wh'],"marginBottom":"14px"}),
        dbc.Row([
            dbc.Col(stl_panel,md=8),
            dbc.Col(html.Div([
                html.Div("Modell-kártya (validációs eredmények)",style={"fontSize":"11px","fontWeight":"500","color":C['wh'],"marginBottom":"10px"}),
                *[html.Div([html.Div(l,style={"fontSize":"8px","color":C['mut'],"textTransform":"uppercase"}),
                    html.Div(v,style={"fontSize":"13px","fontWeight":"500","color":c,"marginTop":"2px"})],
                    style={"background":C['card2'],"borderRadius":"8px","padding":"9px","marginBottom":"5px"})
                for l,v,c in info],
            ],style=CS),md=4)
        ],className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_fi,config={"displayModeBar":False})],style=CS),md=12)
        ],className="g-3")
    ])

if __name__=="__main__":
    # Lokális futtatás. Élesben (Render) a gunicorn indítja: gunicorn app:server
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)