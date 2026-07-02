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
import base64
from datetime import datetime, timedelta
from io import StringIO
from statsmodels.tsa.seasonal import STL
import holidays
from dotenv import load_dotenv
load_dotenv()
app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}])
app.title = "OkosMérő.hu"
server = app.server

BASE = os.path.dirname(os.path.abspath(__file__))
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY","")
hu_holidays = holidays.Hungary(years=[2025,2026,2027])

# ============================================================
# GYORSÍTÓTÁR (CACHE)
# Minden API-eredményt időbélyeggel tárolunk. Újrahívás csak a
# lejárat után történik. Ha egy API épp nem elérhető, a legutóbbi
# jó adatot szolgáljuk ki — az app nem hal meg egy kimaradástól.
# ============================================================
CACHE = {}

def cachelt(kulcs, ttl_sec, fn, ok_index):
    """kulcs: cache azonosító | ttl_sec: érvényesség másodpercben
    fn: a tényleges API-hívó függvény | ok_index: a visszatérési
    tuple hányadik eleme a sikerjelző (True/False)"""
    most = time.time()
    rec = CACHE.get(kulcs)
    # Van friss, jó adatunk? — nem hívunk API-t
    if rec and (most - rec["ido"]) < ttl_sec:
        return rec["ertek"]
    # Lejárt vagy nincs: tényleges hívás
    ertek = fn()
    if ertek[ok_index]:
        CACHE[kulcs] = {"ido": most, "ertek": ertek}
        return ertek
    # A hívás sikertelen: ha van korábbi jó adat, azt adjuk vissza
    if rec:
        print(f"[CACHE] {kulcs}: friss hívás sikertelen, korábbi jó adat kiszolgálva "
              f"({int((most-rec['ido'])/60)} perce frissült)", flush=True)
        return rec["ertek"]
    return ertek

FEATURES = ['DAM_EUR_MWh','Homerseklet_C','Paratartalom_szazalek','Napsugarzas_W_m2',
    'Szelsebesseg_kmh','Csapadek_mm','EUR_HUF','Ora','Het_napja','Honap','Unnepnap','Hetvege',
    'Extrem_hideg','Extrem_meleg','Fogyasztas_lag1h','Fogyasztas_lag24h','Fogyasztas_lag168h',
    'Nap_termeles_MW','Szel_termeles_MW']

# Prémium sötét téma színpaletta
C = {'bg':'#050d1a','sb':'#070f1e','card':'#0a1628','card2':'#0f1923','brd':'#1a2d42',
     'txt':'#cbd5e1','mut':'#64748b','or':'#FF6600','gr':'#10b981','bl':'#0066CC',
     'rd':'#ef4444','yw':'#f59e0b','cy':'#4b9cd3','wh':'#f1f5f9'}

CHART = dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=C['txt'],family='Inter,sans-serif',size=10),
    margin=dict(l=40,r=15,t=35,b=35),
    xaxis=dict(gridcolor=C['brd'],showline=False,color=C['mut'],zeroline=False),
    yaxis=dict(gridcolor=C['brd'],showline=False,color=C['mut'],zeroline=False),
    showlegend=False)

# Ensemble modell betöltése — élesben kötelező, hiba esetén az app jelzi
ensemble = None
MODELL_HIBA = None
try:
    ensemble = joblib.load(f"{BASE}/ensemble_model.pkl")
except Exception as e:
    MODELL_HIBA = str(e)

def ep(X):
    s = ensemble["sulyok"]
    return (s["xgboost"]*ensemble["xgboost"].predict(X) +
            s["lightgbm"]*ensemble["lightgbm"].predict(X) +
            s["catboost"]*ensemble["catboost"].predict(X))

def tesla_img(szin):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 90">
      <path d="M20 62 Q20 70 28 70 L192 70 Q200 70 200 62 L200 51 Q200 46 196 43 L178 37 Q167 15 146 11 L82 10 Q61 10 50 26 L34 37 L20 43 Q18 46 18 50Z" fill="{szin}"/>
      <path d="M54 70 Q54 80 65 80 Q76 80 76 70" fill="rgba(0,0,0,0.3)"/>
      <path d="M144 70 Q144 80 155 80 Q166 80 166 70" fill="rgba(0,0,0,0.3)"/>
      <path d="M58 37 L67 18 Q71 11 80 11 L108 11 L108 37Z" fill="rgba(0,0,0,0.2)"/>
      <path d="M112 37 L112 11 L142 11 Q153 11 163 26 L172 37Z" fill="rgba(0,0,0,0.2)"/>
      <line x1="110" y1="11" x2="110" y2="37" stroke="rgba(0,0,0,0.15)" stroke-width="1"/>
      <rect x="164" y="28" width="16" height="6" rx="3" fill="#f97316"/>
    </svg>'''
    b64 = base64.b64encode(svg.encode()).decode()
    return html.Img(src=f"data:image/svg+xml;base64,{b64}",
        style={"width":"85%","maxWidth":"160px","display":"block","margin":"2px auto"})

def dam_szin(ar, atlag):
    if ar < 0: return C['gr']
    elif ar < atlag * 0.7: return '#a3e635'
    elif ar > atlag * 1.3: return C['rd']
    return C['yw']

def ido_ikon(code):
    if not code: return "☀️"
    code = int(code)
    if code == 0: return "☀️"
    elif code <= 3: return "⛅"
    elif code <= 48: return "🌫️"
    elif code <= 67: return "🌧️"
    elif code <= 77: return "❄️"
    else: return "⛈️"

# ============================================================
# ÉLŐ API LEKÉRDEZÉSEK — nincs szimulált adat.
# Minden függvény (érték, ok) párost ad vissza; ha ok=False,
# az app hibapanelt mutat az érintett forrásra.
# ============================================================

def get_eur_huf():
    try:
        r = requests.get("https://data-api.ecb.europa.eu/service/data/EXR/D.HUF.EUR.SP00.A",
            params={"startPeriod":(datetime.now()-timedelta(days=7)).strftime("%Y-%m-%d"),
                    "endPeriod":datetime.now().strftime("%Y-%m-%d"),"format":"csvdata"},timeout=10)
        df = pd.read_csv(StringIO(r.text))[["TIME_PERIOD","OBS_VALUE"]].dropna()
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"],errors="coerce")
        return float(df["OBS_VALUE"].dropna().iloc[-1]),True
    except Exception as e:
        print(f"[HIBA] ECB árfolyam: {e}", flush=True)
        return None,False

def get_ho():
    for kiserlet in range(3):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude":47.5,"longitude":19.0,"current_weather":"true","timezone":"Europe/Budapest"},timeout=10)
            d = r.json()
            if "current_weather" not in d:
                print(f"[HIBA] Open-Meteo (hőmérséklet) {kiserlet+1}. kísérlet — HTTP {r.status_code}, válasz: {str(d)[:200]}", flush=True)
                time.sleep(3)
                continue
            return float(d["current_weather"]["temperature"]),True
        except Exception as e:
            print(f"[HIBA] Open-Meteo (hőmérséklet) {kiserlet+1}. kísérlet: {e}", flush=True)
            time.sleep(3)
    return None,False

def get_idojaras():
    for kiserlet in range(3):
        try:
            ma = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude":47.5,"longitude":19.0,
                    "hourly":"temperature_2m,relative_humidity_2m,direct_radiation,wind_speed_10m,precipitation",
                    "daily":"temperature_2m_max,temperature_2m_min,weathercode",
                    "timezone":"Europe/Budapest",
                    "start_date":(ma+timedelta(days=1)).strftime("%Y-%m-%d"),
                    "end_date":(ma+timedelta(days=4)).strftime("%Y-%m-%d")},timeout=15)
            d = r.json()
            if "hourly" not in d:
                print(f"[HIBA] Open-Meteo (előrejelzés) {kiserlet+1}. kísérlet — HTTP {r.status_code}, válasz: {str(d)[:200]}", flush=True)
                time.sleep(3)
                continue
            hourly = pd.DataFrame({"Datum":pd.to_datetime(d["hourly"]["time"]),
                "Homerseklet_C":d["hourly"]["temperature_2m"],
                "Paratartalom_szazalek":d["hourly"]["relative_humidity_2m"],
                "Napsugarzas_W_m2":d["hourly"]["direct_radiation"],
                "Szelsebesseg_kmh":d["hourly"]["wind_speed_10m"],
                "Csapadek_mm":d["hourly"]["precipitation"]})
            holnap = (ma+timedelta(days=1)).date()
            hourly = hourly[hourly["Datum"].dt.date==holnap].reset_index(drop=True)
            daily = {"max":d["daily"]["temperature_2m_max"][:4],
                     "min":d["daily"]["temperature_2m_min"][:4],
                     "code":d["daily"]["weathercode"][:4]}
            if len(hourly)==0:
                print("[HIBA] Open-Meteo (előrejelzés): üres válasz a holnapi napra", flush=True)
                time.sleep(3)
                continue
            return hourly,daily,True
        except Exception as e:
            print(f"[HIBA] Open-Meteo (előrejelzés) {kiserlet+1}. kísérlet: {e}", flush=True)
            time.sleep(3)
    return None,None,False

def get_dam():
    """Holnapi DAM árak. Publikálás előtt (kb. 14:00-ig) a MAI valós
    árakat adja vissza — az is élő ENTSO-E adat, a felület jelzi."""
    if not ENTSOE_API_KEY:
        return None,None,False,"nincs_kulcs"
    try:
        from entsoe import EntsoePandasClient
        c = EntsoePandasClient(api_key=ENTSOE_API_KEY, timeout=30)
        ma = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
        # 1. próba: holnapi árak
        hs = pd.Timestamp((ma+timedelta(days=1)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        try:
            ho = c.query_day_ahead_prices("HU",start=hs,end=hs+pd.Timedelta(days=1))
            if ho is not None and len(ho)>=20:
                return float(ho.mean()),ho,True,"holnap"
        except:
            pass
        # 2. próba: mai árak (publikálás előtti időszakban)
        ms = pd.Timestamp(ma.strftime("%Y-%m-%d"),tz="Europe/Budapest")
        ho = c.query_day_ahead_prices("HU",start=ms,end=ms+pd.Timedelta(days=1))
        if ho is not None and len(ho)>=20:
            return float(ho.mean()),ho,True,"ma"
        print("[HIBA] ENTSO-E (DAM): sem holnapi, sem mai ár nem érkezett", flush=True)
        return None,None,False,"nincs_adat"
    except Exception as e:
        print(f"[HIBA] ENTSO-E (DAM árak): {e}", flush=True)
        return None,None,False,"api_hiba"

def get_load():
    """Elmúlt 8 nap valós fogyasztása — lag feature-ökhöz és STL-hez."""
    if not ENTSOE_API_KEY:
        return None,False
    try:
        from entsoe import EntsoePandasClient
        c = EntsoePandasClient(api_key=ENTSOE_API_KEY, timeout=30)
        ma = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
        s = pd.Timestamp((ma-timedelta(days=8)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        e = pd.Timestamp(ma.strftime("%Y-%m-%d"),tz="Europe/Budapest")
        load = c.query_load("HU",start=s,end=e)
        if isinstance(load,pd.DataFrame): load=load.iloc[:,0]
        if load is None or len(load)<168:
            print(f"[HIBA] ENTSO-E (fogyasztás): kevés adat érkezett ({0 if load is None else len(load)} sor, minimum 168 kell)", flush=True)
            return None,False
        return load,True
    except Exception as e:
        print(f"[HIBA] ENTSO-E (fogyasztás): {e}", flush=True)
        return None,False

def get_megujulo():
    """Elmúlt 7 nap átlagos nap- és szélenergia-termelése — ÉLŐ ENTSO-E.
    Szűkített lekérdezés: csak a nap (B16) és a szárazföldi szél (B19)
    termeléstípus jön le — a többi erőműtípus adata felesleges lenne."""
    if not ENTSOE_API_KEY:
        print("[HIBA] Megújuló: nincs ENTSOE_API_KEY beállítva", flush=True)
        return None,None,False
    try:
        from entsoe import EntsoePandasClient
        c = EntsoePandasClient(api_key=ENTSOE_API_KEY, timeout=30)
        ma = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
        s = pd.Timestamp((ma-timedelta(days=7)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        e = pd.Timestamp(ma.strftime("%Y-%m-%d"),tz="Europe/Budapest")
        nap = None; szel = None
        try:
            g_nap = c.query_generation("HU",start=s,end=e,psr_type="B16")
            if isinstance(g_nap,pd.DataFrame): g_nap = g_nap.sum(axis=1)
            nap = float(g_nap.mean())
        except Exception as e1:
            print(f"[HIBA] Megújuló (nap, B16): {e1}", flush=True)
        try:
            g_szel = c.query_generation("HU",start=s,end=e,psr_type="B19")
            if isinstance(g_szel,pd.DataFrame): g_szel = g_szel.sum(axis=1)
            szel = float(g_szel.mean())
        except Exception as e2:
            print(f"[HIBA] Megújuló (szél, B19): {e2}", flush=True)
        if nap is None and szel is None:
            return None,None,False
        return (nap if nap is not None else 0.0),(szel if szel is not None else 0.0),True
    except Exception as e:
        print(f"[HIBA] Megújuló termelés: {e}", flush=True)
        return None,None,False

def elorejelez(ido_df,dam_atlag,eur_huf,load_hist,nap,szel,dam_oras):
    lag_buf = list(load_hist.values[-168:])
    eredm=[]
    for i,row in ido_df.iterrows():
        datum=row["Datum"]; ora=datum.hour
        dam_ar = float(dam_oras.iloc[ora]) if dam_oras is not None and ora<len(dam_oras) else dam_atlag
        l1=lag_buf[-1]; l24=lag_buf[-24]; l168=lag_buf[-168]
        X=pd.DataFrame([{'DAM_EUR_MWh':dam_ar,'Homerseklet_C':row["Homerseklet_C"],
            'Paratartalom_szazalek':row["Paratartalom_szazalek"],'Napsugarzas_W_m2':row["Napsugarzas_W_m2"],
            'Szelsebesseg_kmh':row["Szelsebesseg_kmh"],'Csapadek_mm':row["Csapadek_mm"],'EUR_HUF':eur_huf,
            'Ora':ora,'Het_napja':datum.weekday()+1,'Honap':datum.month,
            'Unnepnap':1 if datum.date() in hu_holidays else 0,'Hetvege':1 if datum.weekday()>=5 else 0,
            'Extrem_hideg':1 if row["Homerseklet_C"]<-5 else 0,'Extrem_meleg':1 if row["Homerseklet_C"]>30 else 0,
            'Fogyasztas_lag1h':l1,'Fogyasztas_lag24h':l24,'Fogyasztas_lag168h':l168,
            'Nap_termeles_MW':nap if row["Napsugarzas_W_m2"]>50 else 0,'Szel_termeles_MW':szel
        }],columns=FEATURES)
        josolt=max(float(ep(X)[0]),2000)
        lag_buf.append(josolt)
        eredm.append({"datum":datum,"ora":int(ora),"homerseklet":float(row["Homerseklet_C"]),
            "fogyasztas":josolt,"dam_ar":float(dam_ar),"koltseg_mft":josolt*dam_ar*eur_huf/1_000_000})
    return eredm

def negativ_ablak(edf):
    """Az első összefüggő negatív árú időablak (kezdet, vég, min ár).
    Ha nincs negatív ár: a legolcsóbb összefüggő 2 órás ablakot adja."""
    neg = edf[edf["dam_ar"]<0]
    if len(neg)>0:
        orak = sorted(neg["ora"].tolist())
        start = orak[0]; end = start
        for o in orak[1:]:
            if o == end+1: end = o
            else: break
        blokk = edf[(edf["ora"]>=start)&(edf["ora"]<=end)]
        return start, end+1, float(blokk["dam_ar"].min()), True
    # nincs negatív: legolcsóbb óra körüli ablak
    legolcs = edf.loc[edf["dam_ar"].idxmin()]
    o = int(legolcs["ora"])
    return o, min(o+2,24), float(legolcs["dam_ar"]), False

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
        html.Div(cim,style={"fontSize":"10px","color":C['mut'],"textTransform":"uppercase","letterSpacing":".8px","marginBottom":"2px"}),
        html.Div(val,style={"fontSize":"20px","fontWeight":"600","color":C['wh']}),
        html.Div(sub,style={"fontSize":"11px","color":szin,"marginTop":"1px","fontWeight":"500"}),
        *ch
    ],style={"background":C['card'],"border":f"1px solid {C['brd']}","borderRadius":"12px","padding":"14px 16px","flex":"1"})

def src_sor(nev,ok):
    return html.Div([
        html.Span(nev,style={"fontSize":"10px","color":C['mut']}),
        html.Span("● Élő" if ok else "○ Nem elérhető",style={"fontSize":"10px","color":C['gr'] if ok else C['rd']})
    ],style={"display":"flex","justifyContent":"space-between","padding":"2px 0"})

def hiba_panel(hianyzo, modell_hiba=None):
    sorok = []
    if modell_hiba:
        sorok.append(html.Div([
            html.Div("⚠ Ensemble modell nem tölthető be",style={"fontSize":"14px","fontWeight":"600","color":C['rd']}),
            html.Div(f"Részletek: {modell_hiba}",style={"fontSize":"11px","color":C['mut'],"marginTop":"4px"}),
            html.Div("Ellenőrizd, hogy az ensemble_model.pkl az app.py mellett van-e.",
                style={"fontSize":"11px","color":C['txt'],"marginTop":"4px"})
        ],style={"marginBottom":"16px"}))
    if hianyzo:
        sorok.append(html.Div([
            html.Div("⚠ Élő adatforrás nem elérhető",style={"fontSize":"14px","fontWeight":"600","color":C['rd']}),
            html.Div(f"Érintett: {', '.join(hianyzo)}",style={"fontSize":"12px","color":C['txt'],"marginTop":"6px"}),
            html.Div("Az alkalmazás kizárólag élő adatokkal működik. "
                     "Ellenőrizd az API kulcsot (ENTSOE_API_KEY környezeti változó) és a hálózati kapcsolatot, "
                     "majd az oldal 5 percen belül automatikusan újrapróbálkozik.",
                style={"fontSize":"11px","color":C['mut'],"marginTop":"6px"})
        ]))
    return html.Div(sorok,style={"background":C['card'],"border":f"1px solid {C['rd']}",
        "borderRadius":"14px","padding":"24px","maxWidth":"640px"})

SIDEBAR = html.Div([
    html.Div([
        html.Span("⚡",style={"fontSize":"24px"}),
        html.Span("Okos",style={"fontSize":"18px","fontWeight":"800","color":C['wh']}),
        html.Span("Mérő",style={"fontSize":"18px","fontWeight":"800","color":C['gr']}),
        html.Span(".hu",style={"fontSize":"18px","fontWeight":"800","color":C['wh']}),
    ],style={"display":"flex","alignItems":"center","gap":"6px","marginBottom":"2px"}),
    html.Div("ENERGIAPIACI ASSZISZTENS",style={"fontSize":"9px","color":C['cy'],"letterSpacing":"1.5px","marginBottom":"30px"}),

    *[html.Div([
        html.I(className=f"fa-solid {ikon}",style={"width":"18px","fontSize":"14px"}),
        html.Span(nev,style={"marginLeft":"12px","fontSize":"13px","fontWeight":"500"})
    ],id=f"nav-{oid}",n_clicks=0,style={"display":"flex","alignItems":"center","padding":"12px 16px",
        "borderRadius":"10px","cursor":"pointer","marginBottom":"6px","transition":"all 0.2s",
        "color":C['or'] if oid=="fooldal" else C['mut'],
        "background":"rgba(255,102,0,0.1)" if oid=="fooldal" else "transparent"})
    for ikon,nev,oid in [("fa-house","Főoldal","fooldal"),
                          ("fa-chart-line","Energiaelemzés","elemzes"),
                          ("fa-flask","ML Modell Labor","mllabor")]],

    html.Div(style={"flex":"1"}),

    html.Div([
        html.Div("Adatkapcsolat",style={"fontSize":"11px","fontWeight":"600","color":C['wh'],"marginBottom":"8px"}),
        html.Div(id="src-panel")
    ],style={"background":"rgba(26,45,66,0.2)","border":f"1px solid {C['brd']}","borderRadius":"10px","padding":"12px"}),

    html.Div(id="modell-panel",style={"marginTop":"12px","padding":"0 4px"})
],style={"width":"230px","minWidth":"230px","background":C['sb'],
    "borderRight":f"1px solid {C['brd']}","padding":"24px 16px",
    "display":"flex","flexDirection":"column","minHeight":"100vh",
    "position":"fixed","top":"0","left":"0","bottom":"0","zIndex":"100"})

app.layout = html.Div([
    SIDEBAR,
    html.Div([
        html.Div(id="statusz",style={"marginBottom":"16px"}),
        html.Div(id="kpi-sor",style={"marginBottom":"16px"}),
        html.Div(id="oldal-content"),
        dcc.Interval(id="refresh",interval=1800*1000,n_intervals=0),
        dcc.Store(id="oldal",data="fooldal"),
        dcc.Store(id="adatok",data=None),
    ],style={"marginLeft":"230px","padding":"24px 28px","background":C['bg'],"minHeight":"100vh"})
],style={"fontFamily":"Inter,sans-serif","background":C['bg']})

@callback(Output("adatok","data"),Input("refresh","n_intervals"))
def fetch(n):
    if ensemble is None:
        return {"kritikus_hiba":True,"hianyzo":[],"modell_hiba":MODELL_HIBA}

    eur_huf,eur_ok = cachelt("ecb", 6*3600, get_eur_huf, 1)
    ido_df,daily,ido_ok = cachelt("idojaras", 3600, get_idojaras, 2)
    dam_atlag,dam_oras,dam_ok,dam_forras = cachelt("dam", 3600, get_dam, 2)
    load,load_ok = cachelt("load", 3600, get_load, 1)
    nap,szel,gen_ok = cachelt("megujulo", 3600, get_megujulo, 2)
    aho,ho_ok = cachelt("homerseklet", 3600, get_ho, 1)

    hianyzo = []
    if not dam_ok: hianyzo.append("ENTSO-E (DAM árak)")
    if not load_ok: hianyzo.append("ENTSO-E (fogyasztás)")
    if not gen_ok: hianyzo.append("ENTSO-E (megújuló termelés)")
    if not ido_ok: hianyzo.append("Open-Meteo (időjárás)")
    if not eur_ok: hianyzo.append("ECB (árfolyam)")

    # Kritikus források nélkül nincs előrejelzés
    if not (dam_ok and load_ok and ido_ok and eur_ok and gen_ok):
        return {"kritikus_hiba":True,"hianyzo":hianyzo,"modell_hiba":None}

    eredm = elorejelez(ido_df,dam_atlag,eur_huf,load,nap,szel,dam_oras)

    stl_data = None
    try:
        s = load.tail(720) if len(load)>=720 else load
        res = STL(s,period=24,seasonal=25,robust=True).fit()
        std=float(res.resid.std()); mean=float(res.resid.mean()); kuszob=2.5*std
        mask=abs(res.resid-mean)>kuszob
        stl_data={"trend":res.trend.tolist(),"seasonal":res.seasonal.tolist(),
            "residual":res.resid.tolist(),"original":[float(x) for x in s],
            "anomalia_db":int(mask.sum()),
            "stat":{"std":std,"mean":mean,"kuszob":kuszob,
                "trend_utolso":float(res.trend.iloc[-1]),
                "irany":"emelkedő" if res.trend.iloc[-1]>res.trend.iloc[-24] else "csökkenő"}}
    except: pass

    return {"kritikus_hiba":False,"eredm":eredm,"eur_huf":eur_huf,"dam_atlag":dam_atlag,
        "dam_oras":[float(x) for x in dam_oras],
        "dam_forras":dam_forras,
        "aho":aho,"stl":stl_data,"daily":daily,
        "frissites":datetime.now().strftime("%H:%M"),
        "fb":{"ENTSO-E":not dam_ok,"Open-Meteo":not ido_ok,"ECB":not eur_ok},
        "hianyzo":hianyzo}

@callback(Output("oldal","data"),
    [Input(f"nav-{x}","n_clicks") for x in ["fooldal","elemzes","mllabor"]],
    prevent_initial_call=True)
def nav(*_):
    ctx=dash.callback_context
    if not ctx.triggered: return "fooldal"
    return ctx.triggered[0]["prop_id"].split(".")[0].replace("nav-","")

NB = {"display":"flex","alignItems":"center","padding":"12px 16px","borderRadius":"10px","cursor":"pointer","marginBottom":"6px","transition":"all 0.2s"}

@callback([Output("statusz","children"),Output("kpi-sor","children"),
    Output("oldal-content","children"),Output("src-panel","children"),
    Output("modell-panel","children"),
    Output("nav-fooldal","style"),Output("nav-elemzes","style"),Output("nav-mllabor","style")],
    [Input("adatok","data"),Input("oldal","data")])
def render(data,oldal):
    ns=[{**NB,"color":C['or'],"background":"rgba(255,102,0,0.1)"} if oldal==x
        else {**NB,"color":C['mut'],"background":"transparent"} for x in ["fooldal","elemzes","mllabor"]]

    modell_info = html.Div([
        html.Div("Modell Futómű",style={"fontSize":"9px","color":C['cy'],"fontWeight":"bold"}),
        html.Div("FLAML AutoML Ensemble",style={"fontSize":"10px","color":C['mut']}),
        html.Div(f"MAE {ensemble['metrikak']['ensemble']['mae']:.2f} MWh | R² {ensemble['metrikak']['ensemble']['r2']:.4f}"
                 if ensemble else "Modell nem elérhető",
            style={"fontSize":"9px","color":C['gr'] if ensemble else C['rd'],"marginTop":"2px"})
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
                        for k in ["ENTSO-E","Open-Meteo","ECB"]])
        return (statusz,html.Div(),
            hiba_panel(data.get("hianyzo",[]),data.get("modell_hiba")),
            src,modell_info,*ns)

    edf=pd.DataFrame(data["eredm"])
    eur_huf=data["eur_huf"]; dam_atlag=data["dam_atlag"]; aho=data["aho"]
    fb=data["fb"]
    legolcs=edf.loc[edf["dam_ar"].idxmin()]

    stl_db=data["stl"]["anomalia_db"] if data["stl"] else 0
    stl_tot=len(data["stl"]["trend"]) if data["stl"] else 0

    ora=datetime.now().hour
    dam_most=data["dam_oras"][ora] if ora<len(data["dam_oras"]) else dam_atlag
    dam_sz=dam_szin(dam_most,dam_atlag)
    t_dam=[float(x) for x in data["dam_oras"]]
    t_fogy=edf["fogyasztas"].tolist()

    dam_cimke = "Holnapi árak" if data["dam_forras"]=="holnap" else "Mai árak (holnapi publikálás előtt)"

    statusz=html.Div([
        html.Span("● ",style={"color":C['gr']}),
        html.Span(f"Élő adatok frissítve: {data['frissites']} — {dam_cimke}",
            style={"fontSize":"12px","color":C['txt'],"fontWeight":"500"})
    ])

    src=html.Div([src_sor(k,not v) for k,v in fb.items()])

    kat = "Negatív" if dam_most<0 else ("Olcsó" if dam_most<dam_atlag*0.7 else ("Drága" if dam_most>dam_atlag*1.3 else "Átlagos"))
    ksor=html.Div([
        kpi("Jelenlegi DAM ár",f"{dam_most:.0f} €/MWh",kat,dam_sz,t_dam),
        kpi("Most (Fogyasztás)",f"{edf['fogyasztas'].iloc[min(ora,len(edf)-1)]:,.0f} MWh","Aktuális fogyasztás",C['bl'],t_fogy),
        kpi("Budapest",f"{aho:.0f} °C","Most",C['yw']),
        kpi("EUR/HUF",f"{eur_huf:.1f} Ft","Árfolyam",C['txt']),
        kpi("Legolcsóbb időszak",f"{int(legolcs['ora']):02d}:00 – {legolcs['dam_ar']:.0f} €","Holnap" if data["dam_forras"]=="holnap" else "Ma",C['gr']),
        kpi("STL Állapot",f"{stl_db} / {stl_tot}","Óra",C['or'] if stl_db > 0 else C['gr']),
    ],style={"display":"flex","gap":"12px"})

    if oldal=="fooldal":
        page=fooldal(edf,data,dam_atlag,eur_huf,legolcs,dam_most,dam_sz)
    elif oldal=="elemzes":
        page=elemzes(edf,data,dam_atlag)
    else:
        page=mllabor(data)
    return statusz,ksor,page,src,modell_info,*ns

def fooldal(edf,data,dam_atlag,eur_huf,legolcs,dam_most,dam_sz):
    orak=[f"{int(r['ora']):02d}:00" for _,r in edf.iterrows()]
    dam_vals=edf["dam_ar"].tolist()

    fig_dam=go.Figure()
    fig_dam.add_trace(go.Scatter(x=orak,y=dam_vals,mode="lines+markers",
        line=dict(color=C['or'],width=3),marker=dict(size=4,color=C['or']),
        hovertemplate="%{x}<br>%{y:.0f} €/MWh<extra></extra>"))
    fig_dam.add_hline(y=0,line=dict(color=C['brd'],width=1,dash="dot"))
    lay=dict(**CHART); lay["height"]=240
    fig_dam.update_layout(**lay)

    olcso_e = dam_most < dam_atlag * 0.7 or dam_most < 0
    t_szin = C['gr'] if olcso_e else C['rd']

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = dam_most,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': " €/MWh", 'font': {'size': 18, 'color': C['wh']}},
        gauge = {
            'axis': {'range': [-100, 150], 'tickwidth': 1, 'tickcolor': C['mut']},
            'bar': {'color': t_szin, 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [-100, 0], 'color': 'rgba(16,185,129,0.2)'},
                {'range': [0, 100], 'color': 'rgba(245,158,11,0.2)'},
                {'range': [100, 150], 'color': 'rgba(239,68,68,0.2)'}],
        }))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=20,r=20,t=20,b=20), height=140)

    varj=(int(legolcs['ora'])-datetime.now().hour)%24
    top3=edf.nsmallest(3,"dam_ar")
    rk_c=[C['gr'], '#a3e635', C['yw']]

    fig_f=go.Figure()
    fig_f.add_trace(go.Scatter(x=orak,y=dam_vals,mode="lines",fill="tozeroy",
        line=dict(color=C['or'],width=2),fillcolor="rgba(255,102,0,0.06)"))
    lay2=dict(**CHART); lay2["height"]=150; lay2["margin"]=dict(l=30,r=10,t=10,b=25)
    fig_f.update_layout(**lay2)

    # ---- ÉLŐ számítások (korábban beégetett értékek) ----
    n_start, n_end, n_min, van_negativ = negativ_ablak(edf)
    ar_min = float(edf["dam_ar"].min())
    ar_max = float(edf["dam_ar"].max())
    ar_atlag = float(edf["dam_ar"].mean())
    ar_szoras = float(edf["dam_ar"].std())
    volatilitas = "Magas" if ar_szoras > 40 else ("Közepes" if ar_szoras > 20 else "Alacsony")
    vol_szin = C['or'] if ar_szoras > 40 else (C['yw'] if ar_szoras > 20 else C['gr'])
    nap_cimke = "Holnap" if data["dam_forras"]=="holnap" else "Ma"

    m = ensemble.get("metrikak",{}).get("ensemble",{})

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(f"⭐ LEGJOBB TÖLTÉSI IDŐSZAKOK ({nap_cimke.upper()})",style={"fontSize":"11px","fontWeight":"600","color":C['wh'],"marginBottom":"16px"}),
                *[html.Div([
                    html.Div(str(i+1),style={"width":"24px","height":"24px","borderRadius":"50%",
                        "background":C['card2'],"color":rk_c[i],"display":"flex","alignItems":"center",
                        "justifyContent":"center","fontSize":"11px","fontWeight":"700","border":f"1px solid {C['brd']}"}),
                    html.Div([
                        html.Div(f"{int(r['ora']):02d}:00 – {int(r['ora'])+1:02d}:00",style={"fontSize":"13px","fontWeight":"600","color":C['wh']}),
                        html.Div("Nagyon olcsó" if r['dam_ar']<0 else "Kedvező",style={"fontSize":"10px","color":C['mut']})
                    ],style={"flex":"1","marginLeft":"10px"}),
                    html.Span(f"{r['dam_ar']:.0f} €/MWh",style={"fontSize":"13px","fontWeight":"700","color":rk_c[i]})
                ],style={"display":"flex","alignItems":"center","padding":"10px 12px",
                    "background":C['card2'],"border":f"1px solid {C['brd']}","borderRadius":"10px","marginBottom":"10px"})
                for i,(_,r) in enumerate(top3.iterrows())],
                html.Div("Összes időszak megtekintése ›",style={"textAlign":"center","fontSize":"11px","color":C['mut'],"cursor":"pointer","marginTop":"14px"})
            ],style=CS),md=3),

            dbc.Col(html.Div([
                html.Div("24 ÓRÁS DAM ÁRATLAG (€/MWh)",style={"fontSize":"11px","fontWeight":"600","color":C['wh'],"marginBottom":"4px"}),
                dcc.Graph(figure=fig_dam,config={"displayModeBar":False}),
                html.Div([
                    html.Div([html.Div(style={"width":"10px","height":"10px","borderRadius":"50%","background":color}),
                              html.Span(label,style={"fontSize":"10px","color":C['mut']})],
                             style={"display":"flex","alignItems":"center","gap":"6px"})
                    for color,label in [(C['gr'],"Negatív ár"),('#a3e635',"Olcsó"),(C['yw'],"Átlagos"),(C['or'],"Drága"),(C['rd'],"Nagyon drága")]
                ],style={"display":"flex","gap":"16px","justifyContent":"center","marginTop":"4px"})
            ],style=CS),md=6),

            dbc.Col(html.Div([
                html.Div("⛽ TÖLTHETEK MOST?",style={"fontSize":"11px","fontWeight":"600","color":C['wh'],"marginBottom":"10px","textAlign":"center"}),
                dcc.Graph(figure=fig_gauge,config={"displayModeBar":False}),
                html.Div("IGEN" if olcso_e else "NEM",
                    style={"fontSize":"28px","fontWeight":"800","color":t_szin,"textAlign":"center","lineHeight":"1"}),
                html.Div(f"Várj {varj} órát — {int(legolcs['ora']):02d}:00-kor {legolcs['dam_ar']:.0f} €/MWh" if not olcso_e else "Optimális feltételek!",
                    style={"fontSize":"10px","color":C['mut'],"textAlign":"center","marginTop":"6px"})
            ],style=CS),md=3),
        ],className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(html.Div([
                html.Div("🧮 MEGTAKARÍTÁS KALKULÁTOR",style={"fontSize":"11px","fontWeight":"600","color":C['wh'],"marginBottom":"12px"}),
                html.Div([html.Span("Töltendő energia",style={"fontSize":"11px","color":C['mut']}),
                          html.Span(id="kwh-txt",style={"fontSize":"11px","fontWeight":"bold","color":C['wh']})],
                          style={"display":"flex","justifyContent":"space-between"}),
                dcc.Slider(id="kwh-sl",min=5,max=80,value=40,step=5,marks=None),

                html.Div([html.Span("Akkumulátor kapacitás",style={"fontSize":"11px","color":C['mut']}),
                          html.Span("60 kWh",style={"fontSize":"11px","fontWeight":"bold","color":C['wh']})],
                          style={"display":"flex","justifyContent":"space-between","marginTop":"8px"}),
                dcc.Slider(min=20,max=100,value=60,step=10,marks=None,disabled=True),

                html.Div(id="calc-out",style={"marginTop":"12px"})
            ],style=CS),md=3),

            dbc.Col(html.Div([
                html.Div("🎯 KÖVETKEZŐ NEGATÍV ÁR" if van_negativ else "🎯 LEGOLCSÓBB IDŐABLAK",
                    style={"fontSize":"11px","fontWeight":"600","color":C['wh'],"marginBottom":"16px","textAlign":"center"}),
                html.Div(nap_cimke,style={"fontSize":"16px","fontWeight":"600","color":C['cy'],"textAlign":"center"}),
                html.Div(f"{n_start:02d}:00 – {n_end:02d}:00",style={"fontSize":"32px","fontWeight":"800","color":C['gr'],"textAlign":"center","margin":"8px 0"}),
                html.Div([
                    html.Span("Minimum ár: ",style={"color":C['mut']}),
                    html.Span(f"{n_min:.0f} € /MWh",style={"color":C['gr'],"fontWeight":"700"})
                ],style={"textAlign":"center","fontSize":"13px"}),
                html.Div("Nincs negatív ár a napon" if not van_negativ else "Negatív árú órák — a hálózat fizet a fogyasztásért",
                    style={"fontSize":"9px","color":C['mut'],"textAlign":"center","marginTop":"20px"})
            ],style=CS),md=3),

            dbc.Col(html.Div([
                html.Div(f"📈 {nap_cimke.upper()}I ÁRSTATISZTIKA",style={"fontSize":"11px","fontWeight":"600","color":C['wh'],"marginBottom":"4px"}),
                dcc.Graph(figure=fig_f,config={"displayModeBar":False}),
                dbc.Row([
                    *[dbc.Col(html.Div([
                        html.Div(lbl,style={"fontSize":"9px","color":C['mut']}),
                        html.Div(val,style={"fontSize":"12px","fontWeight":"700","color":col})
                    ],style={"textAlign":"center"}),md=3)
                    for lbl,val,col in [("Minimum",f"{ar_min:.0f} €",C['gr']),
                                        ("Maximum",f"{ar_max:.0f} €",C['rd']),
                                        ("Átlag",f"{ar_atlag:.0f} €",C['yw']),
                                        ("Volatilitás",volatilitas,vol_szin)]]
                ],className="g-1",style={"marginTop":"8px"})
            ],style=CS),md=6),
        ],className="g-3"),

        html.Div(f"Ensemble (XGBoost + LightGBM + CatBoost) — MAE: {m.get('mae',0):.2f} MWh — R²: {m.get('r2',0):.4f} — lag24h + lag168h mindig valós ENTSO-E adat",
            style={"textAlign":"center","color":C['mut'],"fontSize":"10px","marginTop":"24px","letterSpacing":".5px"})
    ])

def elemzes(edf,data,dam_atlag):
    orak=[f"{int(r['ora']):02d}:00" for _,r in edf.iterrows()]

    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=orak,y=edf["fogyasztas"].tolist(),name="Fogyasztás (MWh)",
        mode="lines",fill="tozeroy",line=dict(color=C['bl'],width=2),
        fillcolor="rgba(0,102,204,0.1)"),secondary_y=False)
    fig.add_trace(go.Scatter(x=orak,y=edf["homerseklet"].tolist(),name="Hőmérséklet (°C)",
        mode="lines",line=dict(color=C['yw'],width=2,dash="dot")),secondary_y=True)
    lay=dict(**CHART); lay["height"]=280; lay["showlegend"]=True
    lay["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=C['txt']))
    lay["title"]=dict(text="Fogyasztás + hőmérséklet",font=dict(size=12,color=C['wh']))
    fig.update_layout(**lay)
    fig.update_yaxes(title_text="MWh",gridcolor=C['brd'],color=C['mut'],secondary_y=False)
    fig.update_yaxes(title_text="°C",gridcolor="rgba(0,0,0,0)",color=C['mut'],secondary_y=True)

    daily=data.get("daily",{}); n=min(4,len(daily.get("max",[])))
    ido_panel=html.Div([
        html.Div("Időjárás előrejelzés",style={"fontSize":"11px","fontWeight":"500","color":C['wh'],"marginBottom":"12px"}),
        html.Div([html.Div([
            html.Div(ido_ikon(daily["code"][i] if n>0 and i<len(daily.get("code",[])) else None),
                style={"fontSize":"22px","textAlign":"center"}),
            html.Div("Holnap" if i==0 else f"+{i} nap",style={"fontSize":"8px","color":C['mut'],"textAlign":"center"}),
            html.Div(f"{daily['max'][i]:.0f}°C" if n>0 and i<len(daily.get("max",[])) else "–",
                style={"fontSize":"14px","fontWeight":"500","color":C['or'],"textAlign":"center"}),
            html.Div(f"{daily['min'][i]:.0f}°C" if n>0 and i<len(daily.get("min",[])) else "–",
                style={"fontSize":"9px","color":C['mut'],"textAlign":"center"})
        ],style={"background":C['card2'],"borderRadius":"8px","padding":"10px","flex":"1"})
        for i in range(max(n,1))],style={"display":"flex","gap":"8px"})
    ],style=CS)

    dam_vals=edf["dam_ar"].tolist()
    szinek=[dam_szin(a,dam_atlag) for a in dam_vals]
    fig_d=go.Figure()
    for i in range(len(dam_vals)-1):
        fig_d.add_trace(go.Scatter(x=[orak[i],orak[i+1]],y=[dam_vals[i],dam_vals[i+1]],
            mode="lines",line=dict(color=szinek[i],width=3),showlegend=False,hoverinfo="skip"))
    fig_d.add_hline(y=0,line=dict(color=C['brd'],width=1,dash="dot"))
    lay_d=dict(**CHART); lay_d["height"]=220
    lay_d["title"]=dict(text="24 órás DAM árgörbe — gradient",font=dict(size=12,color=C['wh']))
    fig_d.update_layout(**lay_d)

    fig_h=go.Figure()
    fig_h.add_trace(go.Histogram(x=dam_vals,nbinsx=12,marker=dict(color=C['bl'],opacity=0.8),
        hovertemplate="%{x:.0f} €/MWh — %{y} óra<extra></extra>"))
    lay_h=dict(**CHART); lay_h["height"]=220
    lay_h["title"]=dict(text="DAM ár eloszlás",font=dict(size=12,color=C['wh']))
    lay_h["xaxis"]["title"]="€/MWh"; lay_h["yaxis"]["title"]="Órák száma"
    fig_h.update_layout(**lay_h)

    return html.Div([
        html.Div("Energiaelemzés",style={"fontSize":"16px","fontWeight":"600","color":C['wh'],"marginBottom":"14px"}),
        dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig,config={"displayModeBar":False})],style=CS),md=8),
            dbc.Col(ido_panel,md=4)
        ],className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_d,config={"displayModeBar":False})],style=CS),md=6),
            dbc.Col(html.Div([dcc.Graph(figure=fig_h,config={"displayModeBar":False})],style=CS),md=6)
        ],className="g-3")
    ])

def mllabor(data):
    m=ensemble.get("metrikak",{}); s=ensemble.get("sulyok",{})
    em=m.get("ensemble",{}); xm=m.get("xgboost",{}); lm=m.get("lightgbm",{}); cm=m.get("catboost",{})
    fi_n=[]; fi_v=[]
    try:
        fi=ensemble["xgboost"].feature_importances_
        srt=sorted(zip(FEATURES,fi),key=lambda x:x[1])
        fi_n=[x[0] for x in srt[-10:]]; fi_v=[float(x[1]) for x in srt[-10:]]
    except: pass
    fig_fi=go.Figure()
    if fi_n:
        fig_fi.add_trace(go.Bar(x=fi_v,y=fi_n,orientation="h",
            marker=dict(color=[C['or'] if v==max(fi_v) else C['bl'] for v in fi_v],opacity=0.85),
            hovertemplate="%{y}<br>%{x:.3f}<extra></extra>"))
    lay_fi=dict(**CHART); lay_fi["height"]=280; lay_fi["margin"]=dict(l=140,r=20,t=35,b=40)
    lay_fi["title"]=dict(text="Feature importance (XGBoost, top 10)",font=dict(size=12,color=C['wh']))
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
        fig_stl.add_hline(y=a+k,line=dict(color=C['rd'],width=1,dash="dash"),
            annotation_text="+2.5σ",annotation_font=dict(color=C['rd'],size=9))
        fig_stl.add_hline(y=a-k,line=dict(color=C['rd'],width=1,dash="dash"),
            annotation_text="-2.5σ",annotation_font=dict(color=C['rd'],size=9))
        lay_s=dict(**CHART); lay_s["height"]=300; lay_s["showlegend"]=True
        lay_s["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=C['txt']))
        lay_s["title"]=dict(text="STL dekompozíció — élő ENTSO-E fogyasztási adat",font=dict(size=12,color=C['wh']))
        fig_stl.update_layout(**lay_s)
        stl_panel=html.Div([dcc.Graph(figure=fig_stl,config={"displayModeBar":False})],style=CS)
    else:
        stl_panel=html.Div("STL számítás nem elérhető — kevés élő adat",
            style={**CS,"color":C['yw'],"textAlign":"center","padding":"40px"})

    info=[
        ("Ensemble MAE",f"{em.get('mae',0):.2f} MWh",C['gr']),
        ("Ensemble R²",f"{em.get('r2',0):.4f}",C['gr']),
        ("XGBoost MAE",f"{xm.get('mae',0):.2f} MWh",C['bl']),
        ("LightGBM MAE",f"{lm.get('mae',0):.2f} MWh",C['bl']),
        ("CatBoost MAE",f"{cm.get('mae',0):.2f} MWh",C['bl']),
        ("STL anomáliák",f"{data['stl']['anomalia_db'] if data['stl'] else '–'} / {len(data['stl']['trend']) if data['stl'] else 0}",
         C['gr'] if (data['stl'] and data['stl']['anomalia_db']==0) else C['or']),
        ("Trend irány",data['stl']['stat']['irany'].capitalize() if data['stl'] else '–',
         C['gr'] if data['stl'] and 'emelkedő' in str(data['stl']['stat'].get('irany','')) else C['rd']),
    ]

    return html.Div([
        html.Div("Gépi Tanulás Modell Labor",style={"fontSize":"16px","fontWeight":"600","color":C['wh'],"marginBottom":"14px"}),
        dbc.Row([
            dbc.Col(stl_panel,md=8),
            dbc.Col(html.Div([
                html.Div("Modell teljesítmény",style={"fontSize":"11px","fontWeight":"500","color":C['wh'],"marginBottom":"10px"}),
                *[html.Div([html.Div(l,style={"fontSize":"8px","color":C['mut'],"textTransform":"uppercase"}),
                    html.Div(v,style={"fontSize":"15px","fontWeight":"500","color":c,"marginTop":"2px"})],
                    style={"background":C['card2'],"borderRadius":"8px","padding":"9px","marginBottom":"5px"})
                for l,v,c in info],
                html.Div("Ensemble súlyok",style={"fontSize":"9px","fontWeight":"500","color":C['txt'],"marginTop":"10px","marginBottom":"6px"}),
                *[html.Div([
                    html.Div([html.Span(nev,style={"fontSize":"9px","color":C['mut']}),
                        html.Span(f"{pct:.1%}",style={"fontSize":"9px","color":szin})],
                        style={"display":"flex","justifyContent":"space-between","marginBottom":"3px"}),
                    html.Div([html.Div(style={"height":"100%","width":f"{pct*100:.0f}%","background":szin,"borderRadius":"2px"})],
                        style={"background":C['card2'],"borderRadius":"2px","height":"5px","marginBottom":"6px"})
                ]) for nev,pct,szin in [
                    ("XGBoost",s.get("xgboost",0.33),C['or']),
                    ("LightGBM",s.get("lightgbm",0.33),C['bl']),
                    ("CatBoost",s.get("catboost",0.34),C['gr'])]]
            ],style=CS),md=4)
        ],className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_fi,config={"displayModeBar":False})],style=CS),md=6),
            dbc.Col(html.Div([
                html.Div("Modell részletek",style={"fontSize":"11px","fontWeight":"500","color":C['wh'],"marginBottom":"10px"}),
                *[html.Div([html.Div(l,style={"fontSize":"9px","color":C['mut']}),
                    html.Div(v,style={"fontSize":"12px","fontWeight":"500","color":C['wh'],"marginTop":"2px"})],
                    style={"background":C['card2'],"borderRadius":"8px","padding":"9px","marginBottom":"5px"})
                for l,v in [("Algoritmus","XGBoost + LightGBM + CatBoost"),
                    ("Tanítóminta","100 505 sor | 2015–2026"),("Feature-ök","19"),
                    ("Teszt méret","720 óra (utolsó 30 nap)"),("Súlyozás","1/MAE arányos"),
                    ("Élő adatforrás","ENTSO-E + Open-Meteo + ECB")]]
            ],style=CS),md=6)
        ],className="g-3")
    ])

@callback([Output("kwh-txt","children"),Output("calc-out","children")],
    [Input("kwh-sl","value"),Input("adatok","data")])
def kalk(kwh,data):
    if data is None or data.get("kritikus_hiba"): return f"{kwh} kWh",html.Div()
    edf=pd.DataFrame(data["eredm"]); eur=data["eur_huf"]; atl=data["dam_atlag"]
    ora=datetime.now().hour
    most=data["dam_oras"][ora] if ora<len(data["dam_oras"]) else atl
    olcs=float(edf["dam_ar"].min())
    m_ft=kwh*most*eur/1000; a_ft=kwh*olcs*eur/1000; meg=m_ft-a_ft

    return f"{kwh} kWh",html.Div([
        html.Div([
            html.Div([html.Div("Ha most töltesz",style={"fontSize":"9px","color":C['mut']}),
                html.Div(f"{m_ft:,.0f} Ft",style={"fontSize":"14px","fontWeight":"700","color":C['rd']})],
                style={"background":C['card2'],"borderRadius":"8px","padding":"8px","textAlign":"center","flex":"1","border":f"1px solid {C['brd']}"}),
            html.Div("➔",style={"color":C['mut'],"fontSize":"14px","padding":"0 4px"}),
            html.Div([html.Div("Aranyórában",style={"fontSize":"9px","color":C['mut']}),
                html.Div(f"{a_ft:,.0f} Ft",style={"fontSize":"14px","fontWeight":"700","color":C['gr']})],
                style={"background":C['card2'],"borderRadius":"8px","padding":"8px","textAlign":"center","flex":"1","border":f"1px solid {C['brd']}"})
        ],style={"display":"flex","alignItems":"center","gap":"6px"}),
        html.Div([html.Span("Várható megtakarítás",style={"fontSize":"11px","color":C['gr'],"fontWeight":"500"}),
            html.Span(f"{meg:,.0f} Ft",style={"fontSize":"16px","fontWeight":"800","color":C['gr']})],
            style={"display":"flex","justifyContent":"space-between","alignItems":"center",
                "marginTop":"12px","paddingTop":"10px","borderTop":f"1px solid {C['brd']}"})
    ])

if __name__=="__main__":
    # Lokális futtatás. Élesben (Render) a gunicorn indítja: gunicorn app:server
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
