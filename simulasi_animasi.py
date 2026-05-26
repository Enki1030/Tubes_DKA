"""
╔══════════════════════════════════════════════════════════════════╗
║      SIMULASI MANAJEMEN PENGUMPULAN SAMPAH  – VERSI 3.0         ║
║   100 Rumah  |  5 Gerobak (06:00-15:00)  |  2 Truk (08:00-17:00)║
╚══════════════════════════════════════════════════════════════════╝
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from matplotlib.widgets import Button
from scipy.spatial import distance_matrix as scipy_dm
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 0. KONSTANTA
# ═══════════════════════════════════════════════════════════════════
np.random.seed(2026)

JUMLAH_RUMAH   = 100
JUMLAH_GEROBAK = 5
JUMLAH_TRUK    = 2
KAP_GEROBAK    = 15.0   # kg
KAP_TRUK       = 200.0   # kg

T_G_MULAI = 6  * 60
T_G_AKHIR = 15 * 60
T_T_MULAI = 8  * 60
T_T_AKHIR = 17 * 60

T_ANIM_MULAI = T_G_MULAI
T_ANIM_AKHIR = T_T_AKHIR

V_GEROBAK = 1.0
V_TRUK    = 2.5

FPS    = 20
SIM_DT = 0.6

# ═══════════════════════════════════════════════════════════════════
# 1. PALET WARNA PREMIUM
# ═══════════════════════════════════════════════════════════════════
BG_FIG   = '#06090f'
BG_MAP   = '#0b1120'
BG_PANEL = '#06090f'

C_TXT1   = '#e2e8f0'
C_TXT2   = '#4a5568'
C_BORDER = '#1a2744'
C_GRID   = '#0d1526'
C_EDGE   = '#1a3354'

C_BELUM    = '#2d3a4a'   
C_SEBAGIAN = '#b45309'   
C_SELESAI  = '#065f46'   
C_BORDER_B = '#475569'   
C_BORDER_S = '#f59e0b'   
C_BORDER_D = '#10b981'   
C_TPS      = '#dc2626'   

C_G_LIGHT  = '#fecaca'   
C_G_DARK   = '#7f1d1d'   
C_T_LIGHT  = '#ddd6fe'   
C_T_DARK   = '#3b0764'   

C_OK    = '#22c55e'
C_WARN  = '#f97316'
C_ERROR = '#ef4444'
C_GOLD  = '#f1c40f'

G_EDGE_COLORS = ['#fca5a5','#fdba74','#fde68a','#86efac','#93c5fd']
T_EDGE_COLORS = ['#c4b5fd', '#a78bfa']

# ═══════════════════════════════════════════════════════════════════
# 2. LAYOUT GRAPH
# ═══════════════════════════════════════════════════════════════════
PUSAT   = [(18,18), (18,82), (82,18), (82,82)]
RADIUS  = 16.5
PHI     = (1 + np.sqrt(5)) / 2

x_all, y_all = [], []
for (cx, cy) in PUSAT:
    for k in range(25):
        r  = RADIUS * np.sqrt((k + 0.5) / 25)
        th = 2 * np.pi * k / PHI
        x_all.append(round(cx + r*np.cos(th) + np.random.uniform(-1.0, 1.0), 2))
        y_all.append(round(cy + r*np.sin(th) + np.random.uniform(-1.0, 1.0), 2))

posisi_rumah = {i: (x_all[i], y_all[i]) for i in range(JUMLAH_RUMAH)}
master_koordinat = {f"Rumah_{i}": (x_all[i], y_all[i]) for i in range(JUMLAH_RUMAH)}
sampah_rumah  = {i: np.random.randint(1, 8) for i in range(JUMLAH_RUMAH)}

posisi_tps = {
    'TPS_1': (2.0,  2.0),
    'TPS_2': (2.0,  98.0),
    'TPS_3': (98.0, 2.0),
}
master_koordinat.update(posisi_tps)

G = nx.Graph()
for r, pos in posisi_rumah.items():
    G.add_node(f"Rumah_{r}", pos=pos, type='rumah', sampah=sampah_rumah[r])
for tid, pos in posisi_tps.items():
    G.add_node(tid, pos=pos, type='tps', sampah=0)

coords = np.array(list(posisi_rumah.values()))
dm     = scipy_dm(coords, coords)

for i in range(JUMLAH_RUMAH):
    order = sorted(range(JUMLAH_RUMAH), key=lambda j: dm[i][j])
    for j in order[1:3]:
        G.add_edge(f"Rumah_{i}", f"Rumah_{j}", weight=dm[i][j])
    for j in order[3:]:
        d = dm[i][j]
        if   d <= 13:
            G.add_edge(f"Rumah_{i}", f"Rumah_{j}", weight=d)
        elif d <= 38 and np.random.rand() < 0.07:
            G.add_edge(f"Rumah_{i}", f"Rumah_{j}", weight=d)

for tid, (tx, ty) in posisi_tps.items():
    nearest = sorted(range(JUMLAH_RUMAH),
                     key=lambda i: np.hypot(x_all[i]-tx, y_all[i]-ty))
    for r in nearest[:5]:
        G.add_edge(tid, f"Rumah_{r}", weight=np.hypot(x_all[r]-tx, y_all[r]-ty))

pos_all     = nx.get_node_attributes(G, 'pos')
nodes_rumah = [n for n, d in G.nodes(data=True) if d['type'] == 'rumah']
nodes_tps   = [n for n, d in G.nodes(data=True) if d['type'] == 'tps']
total_sampah = sum(sampah_rumah.values())

def hitung_jarak_euclidean(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ═══════════════════════════════════════════════════════════════════
# 3. ZONASI K-MEANS
# ═══════════════════════════════════════════════════════════════════
X_coords = np.array([posisi_rumah[i] for i in range(JUMLAH_RUMAH)])
kmeans_gerobak = KMeans(n_clusters=JUMLAH_GEROBAK, random_state=2026, n_init=10).fit(X_coords)
zona_gerobak_rumah = kmeans_gerobak.labels_

kmeans_truk = KMeans(n_clusters=JUMLAH_TRUK, random_state=2026, n_init=10).fit(X_coords)
zona_truk_rumah = kmeans_truk.labels_

# ═══════════════════════════════════════════════════════════════════
# 4. ENGINE LOGIC + TRACKING RUTE
# ═══════════════════════════════════════════════════════════════════
sisa_sampah_rumah = {f"Rumah_{i}": sampah_rumah[i] for i in range(JUMLAH_RUMAH)}

kapasitas_maksimal_tps = {
    "TPS_1": float(np.random.randint(400, 501)),
    "TPS_2": float(np.random.randint(400, 501)),
    "TPS_3": float(np.random.randint(400, 501))
}
tampungan_tps = {"TPS_1": 0.0, "TPS_2": 0.0, "TPS_3": 0.0}
booking_tps = {"TPS_1": 0.0, "TPS_2": 0.0, "TPS_3": 0.0}

status_gerobak = {}
gerobak_routes = {i: [] for i in range(JUMLAH_GEROBAK)}
truk_routes = {i: [] for i in range(JUMLAH_TRUK)}

for i in range(JUMLAH_GEROBAK):
    titik_rumah_zona = [posisi_rumah[r] for r in range(JUMLAH_RUMAH) if zona_gerobak_rumah[r] == i]
    if titik_rumah_zona:
        centroid_zona = (np.mean([p[0] for p in titik_rumah_zona]), np.mean([p[1] for p in titik_rumah_zona]))
    else:
        centroid_zona = posisi_tps["TPS_1"]
    tps_awal = min(posisi_tps.keys(), key=lambda t: hitung_jarak_euclidean(posisi_tps[t], centroid_zona))
    status_gerobak[i] = {
        "posisi": tps_awal, "koordinat": posisi_tps[tps_awal], "muatan": 0.0,
        "total_jarak": 0.0, "sibuk_sampai_menit": 0, "status_aktif": True,
        "tps_tujuan_booking": None, "muatan_booked": 0.0
    }
    gerobak_routes[i].append({"pos": posisi_tps[tps_awal], "t": float(T_G_MULAI), "act": "start", "muatan": 0.0})

status_truk = {}
for i in range(JUMLAH_TRUK):
    tps_awal = "TPS_2" if i == 0 else "TPS_3"
    status_truk[i] = {
        "posisi": tps_awal, "koordinat": posisi_tps[tps_awal], "muatan": 0.0,
        "total_jarak": 0.0, "sibuk_sampai_menit": int(T_T_MULAI) - 360, "status_aktif": True,
        "tps_tujuan_booking": None, "muatan_booked": 0.0
    }
    truk_routes[i].append({"pos": posisi_tps[tps_awal], "t": float(T_T_MULAI), "act": "start", "muatan": 0.0})

def cari_tps_tersedia_terdekat(koordinat_sekarang, muatan_armada):
    tps_valid = [t for t, cap in kapasitas_maksimal_tps.items() if booking_tps[t] + muatan_armada <= cap]
    if not tps_valid: return None, None
    tps_terpilih = min(tps_valid, key=lambda t: hitung_jarak_euclidean(koordinat_sekarang, posisi_tps[t]))
    return tps_terpilih, hitung_jarak_euclidean(koordinat_sekarang, posisi_tps[tps_terpilih])

# Time loop
for menit_simulasi in range(661):
    waktu_real = float(T_G_MULAI + menit_simulasi)
    
    for g_id, g_data in status_gerobak.items():
        if menit_simulasi == g_data["sibuk_sampai_menit"] and g_data["tps_tujuan_booking"] is not None:
            tps_tujuan = g_data["tps_tujuan_booking"]
            tampungan_tps[tps_tujuan] += g_data["muatan_booked"]
            g_data["tps_tujuan_booking"] = None
            g_data["muatan_booked"] = 0.0

    for t_id, t_data in status_truk.items():
        if menit_simulasi == t_data["sibuk_sampai_menit"] and t_data["tps_tujuan_booking"] is not None:
            tps_tujuan = t_data["tps_tujuan_booking"]
            tampungan_tps[tps_tujuan] += t_data["muatan_booked"]
            t_data["tps_tujuan_booking"] = None
            t_data["muatan_booked"] = 0.0

    # GEROBAK
    if menit_simulasi <= 540:
        for g_id, g_data in status_gerobak.items():
            if not g_data["status_aktif"] or menit_simulasi < g_data["sibuk_sampai_menit"]: continue
            sisa_waktu_g = 540 - menit_simulasi
            tps_dekat, jarak_tps = cari_tps_tersedia_terdekat(g_data["koordinat"], g_data["muatan"])
            if tps_dekat is None: continue

            rumah_target = [f"Rumah_{i}" for i in range(JUMLAH_RUMAH) if zona_gerobak_rumah[i] == g_id and sisa_sampah_rumah[f"Rumah_{i}"] > 0]
            pilih_pulang = False; pilih_titip_truk = False
            truk_target_id = None; koordinat_temu = None; waktu_ke_temu_g = 0

            if g_data["muatan"] >= 12.0 or (len(rumah_target) == 0 and g_data["muatan"] > 0):
                truk_valid_temu = []
                for t_id, t_data in status_truk.items():
                    if t_data["status_aktif"] and menit_simulasi >= 120 and t_data["muatan"] < 200.0 and menit_simulasi >= t_data["sibuk_sampai_menit"]:
                        mid_coord = ((g_data["koordinat"][0] + t_data["koordinat"][0]) / 2, (g_data["koordinat"][1] + t_data["koordinat"][1]) / 2)
                        d_ke_mid_g = hitung_jarak_euclidean(g_data["koordinat"], mid_coord)
                        d_ke_mid_t = hitung_jarak_euclidean(t_data["koordinat"], mid_coord)
                        waktu_g_ke_mid = int(round(d_ke_mid_g / V_GEROBAK))
                        waktu_t_ke_mid = int(round(d_ke_mid_t / V_TRUK))

                        if d_ke_mid_g < jarak_tps:
                            tps_dekat_t, jarak_tps_t = cari_tps_tersedia_terdekat(mid_coord, t_data["muatan"] + 1.0)
                            if tps_dekat_t is not None:
                                waktu_t_pulang_dari_mid = int(round(jarak_tps_t / V_TRUK + 10))
                                sisa_waktu_t = 660 - menit_simulasi
                                if waktu_t_ke_mid + waktu_t_pulang_dari_mid <= sisa_waktu_t:
                                    truk_valid_temu.append((t_id, mid_coord, waktu_g_ke_mid, d_ke_mid_g, waktu_t_ke_mid))
                if truk_valid_temu:
                    truk_target_id, koordinat_temu, waktu_ke_temu_g, jarak_temu_g, waktu_t_ke_mid = min(truk_valid_temu, key=lambda x: x[3])
                    pilih_titip_truk = True
                else:
                    pilih_pulang = True
            else:
                waktu_pulang_g = int(round(jarak_tps / V_GEROBAK + g_data["muatan"] * 2))
                if sisa_waktu_g <= waktu_pulang_g and g_data["muatan"] > 0:
                    pilih_pulang = True

            if pilih_titip_truk:
                t_data = status_truk[truk_target_id]
                sampah_dititip = min(g_data["muatan"], 200.0 - t_data["muatan"])
                waktu_load = int(round(sampah_dititip * 2))
                
                g_data["total_jarak"] += jarak_temu_g
                g_data["sibuk_sampai_menit"] = menit_simulasi + waktu_ke_temu_g + waktu_load
                g_data["koordinat"] = koordinat_temu
                g_data["posisi"] = f"Rendezvous_Truk_{truk_target_id}"
                g_data["muatan"] -= sampah_dititip
                gerobak_routes[g_id].append({"pos": koordinat_temu, "t": waktu_real + waktu_ke_temu_g + waktu_load, "act": "rendezvous", "muatan": g_data["muatan"]})

                t_data["total_jarak"] += hitung_jarak_euclidean(t_data["koordinat"], koordinat_temu)
                t_data["sibuk_sampai_menit"] = menit_simulasi + waktu_t_ke_mid + waktu_load
                t_data["koordinat"] = koordinat_temu
                t_data["posisi"] = f"Rendezvous_Gerobak_{g_id}"
                t_data["muatan"] += sampah_dititip
                truk_routes[truk_target_id].append({"pos": koordinat_temu, "t": waktu_real + waktu_t_ke_mid + waktu_load, "act": "rendezvous", "muatan": t_data["muatan"]})

            elif pilih_pulang:
                waktu_jalan = int(round(jarak_tps / V_GEROBAK))
                waktu_load = int(round(g_data["muatan"] * 2))
                booking_tps[tps_dekat] += g_data["muatan"]
                g_data["total_jarak"] += jarak_tps
                g_data["sibuk_sampai_menit"] = menit_simulasi + waktu_jalan + waktu_load
                g_data["koordinat"] = posisi_tps[tps_dekat]
                g_data["posisi"] = tps_dekat
                g_data["tps_tujuan_booking"] = tps_dekat
                g_data["muatan_booked"] = g_data["muatan"]
                g_data["muatan"] = 0.0
                gerobak_routes[g_id].append({"pos": posisi_tps[tps_dekat], "t": waktu_real + waktu_jalan + waktu_load, "act": "bongkar", "muatan": 0.0})

            elif len(rumah_target) > 0 and g_data["muatan"] < KAP_GEROBAK:
                r_dekat = min(rumah_target, key=lambda r: hitung_jarak_euclidean(g_data["koordinat"], master_koordinat[r]))
                jarak = hitung_jarak_euclidean(g_data["koordinat"], master_koordinat[r_dekat])
                sampah_diangkut = min(sisa_sampah_rumah[r_dekat], KAP_GEROBAK - g_data["muatan"])
                waktu_jalan = int(round(jarak / V_GEROBAK))
                waktu_load = int(round(sampah_diangkut * 2))
                tps_baru, jarak_tps_baru = cari_tps_tersedia_terdekat(master_koordinat[r_dekat], g_data["muatan"] + sampah_diangkut)

                if tps_baru is not None:
                    if menit_simulasi + waktu_jalan + waktu_load + int(round(jarak_tps_baru / V_GEROBAK + (g_data["muatan"] + sampah_diangkut) * 2)) <= 540:
                        sisa_sampah_rumah[r_dekat] -= sampah_diangkut
                        g_data["muatan"] += sampah_diangkut
                        g_data["total_jarak"] += jarak
                        g_data["sibuk_sampai_menit"] = menit_simulasi + waktu_jalan + waktu_load
                        g_data["koordinat"] = master_koordinat[r_dekat]
                        g_data["posisi"] = r_dekat
                        gerobak_routes[g_id].append({"pos": master_koordinat[r_dekat], "t": waktu_real + waktu_jalan + waktu_load, "act": "angkut", "muatan": g_data["muatan"], "target": r_dekat, "angkut": sampah_diangkut})
                        continue

                booking_tps[tps_dekat] += g_data["muatan"]
                g_data["total_jarak"] += jarak_tps
                g_data["sibuk_sampai_menit"] = menit_simulasi + int(round(jarak_tps / V_GEROBAK + g_data["muatan"] * 2))
                g_data["koordinat"] = posisi_tps[tps_dekat]
                g_data["posisi"] = tps_dekat
                g_data["tps_tujuan_booking"] = tps_dekat
                g_data["muatan_booked"] = g_data["muatan"]
                g_data["muatan"] = 0.0
                gerobak_routes[g_id].append({"pos": posisi_tps[tps_dekat], "t": waktu_real + int(round(jarak_tps / V_GEROBAK + g_data["muatan_booked"] * 2)), "act": "bongkar", "muatan": 0.0})
            else:
                g_data["status_aktif"] = False

    # TRUK
    if menit_simulasi >= 120:
        for t_id, t_data in status_truk.items():
            if not t_data["status_aktif"] or menit_simulasi < t_data["sibuk_sampai_menit"]: continue
            sisa_waktu_t = 660 - menit_simulasi
            tps_dekat, jarak_tps = cari_tps_tersedia_terdekat(t_data["koordinat"], t_data["muatan"])
            if tps_dekat is None: continue

            rumah_target_truk = [f"Rumah_{i}" for i in range(JUMLAH_RUMAH) if zona_truk_rumah[i] == t_id and sisa_sampah_rumah[f"Rumah_{i}"] > 0]
            pilih_pulang_truk = False
            if len(rumah_target_truk) == 0 and t_data["muatan"] > 0: pilih_pulang_truk = True
            elif t_data["muatan"] >= KAP_TRUK * 0.8: pilih_pulang_truk = True
            else:
                waktu_pulang_t = int(round(jarak_tps / V_TRUK + (t_data["muatan"] / 10) * 2))
                if sisa_waktu_t <= waktu_pulang_t and t_data["muatan"] > 0: pilih_pulang_truk = True

            if pilih_pulang_truk:
                waktu_jalan = int(round(jarak_tps / V_TRUK))
                waktu_load = int(round((t_data["muatan"] / 10) * 2))
                booking_tps[tps_dekat] += t_data["muatan"]
                t_data["total_jarak"] += jarak_tps
                t_data["sibuk_sampai_menit"] = menit_simulasi + waktu_jalan + waktu_load
                t_data["koordinat"] = posisi_tps[tps_dekat]
                t_data["posisi"] = tps_dekat
                t_data["tps_tujuan_booking"] = tps_dekat
                t_data["muatan_booked"] = t_data["muatan"]
                t_data["muatan"] = 0.0
                truk_routes[t_id].append({"pos": posisi_tps[tps_dekat], "t": waktu_real + waktu_jalan + waktu_load, "act": "bongkar", "muatan": 0.0})

            elif len(rumah_target_truk) > 0 and t_data["muatan"] < KAP_TRUK:
                r_dekat = min(rumah_target_truk, key=lambda r: hitung_jarak_euclidean(t_data["koordinat"], master_koordinat[r]))
                jarak = hitung_jarak_euclidean(t_data["koordinat"], master_koordinat[r_dekat])
                sampah_diangkut = min(sisa_sampah_rumah[r_dekat], KAP_TRUK - t_data["muatan"])
                waktu_jalan = int(round(jarak / V_TRUK))
                waktu_load = int(round((sampah_diangkut / 10) * 2))
                tps_baru, jarak_tps_baru = cari_tps_tersedia_terdekat(master_koordinat[r_dekat], t_data["muatan"] + sampah_diangkut)

                if tps_baru is not None:
                    if menit_simulasi + waktu_jalan + waktu_load + int(round(jarak_tps_baru / V_TRUK + ((t_data["muatan"] + sampah_diangkut) / 10) * 2)) <= 660:
                        sisa_sampah_rumah[r_dekat] -= sampah_diangkut
                        t_data["muatan"] += sampah_diangkut
                        t_data["total_jarak"] += jarak
                        t_data["sibuk_sampai_menit"] = menit_simulasi + waktu_jalan + waktu_load
                        t_data["koordinat"] = master_koordinat[r_dekat]
                        t_data["posisi"] = r_dekat
                        truk_routes[t_id].append({"pos": master_koordinat[r_dekat], "t": waktu_real + waktu_jalan + waktu_load, "act": "angkut", "muatan": t_data["muatan"], "target": r_dekat, "angkut": sampah_diangkut})
                        continue

                booking_tps[tps_dekat] += t_data["muatan"]
                t_data["total_jarak"] += jarak_tps
                t_data["sibuk_sampai_menit"] = menit_simulasi + int(round(jarak_tps / V_TRUK + (t_data["muatan"] / 10) * 2))
                t_data["koordinat"] = posisi_tps[tps_dekat]
                t_data["posisi"] = tps_dekat
                t_data["tps_tujuan_booking"] = tps_dekat
                t_data["muatan_booked"] = t_data["muatan"]
                t_data["muatan"] = 0.0
                truk_routes[t_id].append({"pos": posisi_tps[tps_dekat], "t": waktu_real + int(round(jarak_tps / V_TRUK + (t_data["muatan_booked"] / 10) * 2)), "act": "bongkar", "muatan": 0.0})
            else:
                t_data["status_aktif"] = False

# ═══════════════════════════════════════════════════════════════════
# 5. PRE-KALKULASI FRAME UNTUK ANIMASI
# ═══════════════════════════════════════════════════════════════════
sim_times = np.arange(T_ANIM_MULAI, T_ANIM_AKHIR + SIM_DT, SIM_DT)
N_FRAMES  = len(sim_times)

collect_events = []
for routes in [gerobak_routes, truk_routes]:
    for vid, rlist in routes.items():
        for st in rlist:
            if st["act"] == "angkut":
                collect_events.append((st["t"], int(st["target"].split('_')[1]), st["angkut"]))
collect_events.sort(key=lambda x: x[0])

color_frames = []
cumul = {r: 0.0 for r in range(JUMLAH_RUMAH)}
visited = set(); done = set(); ev_i = 0

for t_f in sim_times:
    while ev_i < len(collect_events) and collect_events[ev_i][0] <= t_f:
        t_e, r_id, amt = collect_events[ev_i]
        cumul[r_id] += amt; visited.add(r_id)
        if cumul[r_id] >= sampah_rumah[r_id]: done.add(r_id)
        ev_i += 1
    clr = []
    for r in range(JUMLAH_RUMAH):
        if r in done:    clr.append(C_SELESAI)
        elif r in visited: clr.append(C_SEBAGIAN)
        else:              clr.append(C_BELUM)
    color_frames.append(clr)

def get_pos(route, t):
    if len(route) == 0: return (0,0)
    if t <= route[0]["t"]:  return route[0]["pos"]
    if t >= route[-1]["t"]: return route[-1]["pos"]
    for i in range(len(route)-1):
        t0, t1 = route[i]["t"], route[i+1]["t"]
        if t0 <= t < t1:
            f = (t-t0)/(t1-t0) if t1 > t0 else 1.0
            x0,y0 = route[i]["pos"]
            x1,y1 = route[i+1]["pos"]
            return (x0+(x1-x0)*f, y0+(y1-y0)*f)
    return route[-1]["pos"]

def get_load(route, t):
    if len(route) == 0: return 0.0
    if t <= route[0]["t"]:  return 0.0
    if t >= route[-1]["t"]: return route[-1]["muatan"]
    for i in range(len(route)-1):
        if route[i]["t"] <= t < route[i+1]["t"]:
            return route[i]["muatan"]
    return route[-1]["muatan"]

# ═══════════════════════════════════════════════════════════════════
# 6. SETUP FIGURE & STYLING
# ═══════════════════════════════════════════════════════════════════
def lerp_c(c0, c1, t):
    t = max(0.0, min(1.0, float(t)))
    r0, g0, b0 = to_rgb(c0)
    r1, g1, b1 = to_rgb(c1)
    return (r0+(r1-r0)*t, g0+(g1-g0)*t, b0+(b1-b0)*t)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.facecolor': BG_MAP,
    'figure.facecolor': BG_FIG,
})

fig = plt.figure(figsize=(20, 11), facecolor=BG_FIG)
ax  = fig.add_axes([0.01, 0.01, 0.70, 0.97], facecolor=BG_MAP)
axp = fig.add_axes([0.73, 0.01, 0.27, 0.97], facecolor=BG_PANEL)

for sp in ax.spines.values():
    sp.set_color(C_BORDER); sp.set_linewidth(0.8)
ax.tick_params(colors=C_TXT2, labelsize=7)
ax.grid(True, color=C_GRID, linewidth=0.4, alpha=0.7, linestyle=':')

mg = 8
ax.set_xlim(min(x_all)-mg, max(x_all)+mg)
ax.set_ylim(min(y_all)-mg, max(y_all)+mg)
ax.set_aspect('equal', adjustable='datalim')

# ELEMEN STATIS MAP
for (u, v) in G.edges():
    xu,yu = pos_all[u]; xv,yv = pos_all[v]
    ax.plot([xu,xv],[yu,yv], color=C_EDGE, alpha=0.28, linewidth=0.55, zorder=1)

for tid in nodes_tps:
    tx,ty = pos_all[tid]
    ax.scatter(tx, ty, s=1400, marker='D', color=C_TPS, alpha=0.10, zorder=2)
    ax.scatter(tx, ty, s=700,  marker='D', color=C_TPS, alpha=0.20, zorder=3)
    ax.scatter(tx, ty, s=320,  marker='D', color=C_TPS, edgecolors='white', linewidths=1.2, zorder=5)
    ax.text(tx, ty-6, tid, ha='center', va='top',
            color='#fca5a5', fontsize=8.5, fontweight='bold', zorder=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f0516', edgecolor=C_TPS, linewidth=0.9, alpha=0.85))

# ELEMEN DINAMIS
sc_rumah = ax.scatter(x_all, y_all, s=200, marker='o', c=[C_BELUM]*JUMLAH_RUMAH,
                       edgecolors=[C_BORDER_B]*JUMLAH_RUMAH, linewidths=0.9, zorder=4)

for i in range(JUMLAH_RUMAH):
    ax.text(x_all[i], y_all[i], str(i), ha='center', va='center',
            color='#94a3b8', fontsize=4.5, fontweight='bold', zorder=5)

sc_g = []; lb_g = []
for g_id in range(JUMLAH_GEROBAK):
    gx, gy = gerobak_routes[g_id][0]["pos"]
    gw = ax.scatter(gx, gy, s=500, marker='s', color=C_G_LIGHT, alpha=0.15, zorder=8)
    sc = ax.scatter(gx, gy, s=280, marker='s', color=C_G_LIGHT, edgecolors=G_EDGE_COLORS[g_id], linewidths=1.8, zorder=10)
    lb = ax.text(gx, gy+3.8, f"G{g_id}", ha='center', va='bottom', color=G_EDGE_COLORS[g_id], fontsize=7.5, fontweight='bold', zorder=11, bbox=dict(boxstyle='round,pad=0.22', facecolor='#070e1a', edgecolor='none', alpha=0.80))
    sc_g.append((sc, gw)); lb_g.append(lb)

sc_t = []; lb_t = []
for t_id in range(JUMLAH_TRUK):
    tx, ty = truk_routes[t_id][0]["pos"]
    tw = ax.scatter(tx, ty, s=700, marker='^', color=C_T_LIGHT, alpha=0.12, zorder=8)
    sc = ax.scatter(tx, ty, s=380, marker='^', color=C_T_LIGHT, edgecolors=T_EDGE_COLORS[t_id], linewidths=1.8, zorder=10)
    lb = ax.text(tx, ty+4.5, f"T{t_id}", ha='center', va='bottom', color=T_EDGE_COLORS[t_id], fontsize=7.5, fontweight='bold', zorder=11, bbox=dict(boxstyle='round,pad=0.22', facecolor='#07030f', edgecolor='none', alpha=0.80))
    sc_t.append((sc, tw)); lb_t.append(lb)

jam_box = ax.text(0.5, 0.993, "06:00", transform=ax.transAxes, ha='center', va='top', color=C_GOLD, fontsize=32, fontweight='bold', fontfamily='monospace', zorder=20, bbox=dict(boxstyle='round,pad=0.55', facecolor='#060a14', edgecolor=C_GOLD, linewidth=2.0, alpha=0.95))

leg_el = [
    mpatches.Patch(facecolor=C_BELUM,    edgecolor=C_BORDER_B, label='Belum dikunjungi'),
    mpatches.Patch(facecolor=C_SEBAGIAN, edgecolor=C_BORDER_S, label='Sebagian terangkut'),
    mpatches.Patch(facecolor=C_SELESAI,  edgecolor=C_BORDER_D, label='Selesai'),
    mpatches.Patch(facecolor=C_TPS,      edgecolor='white',    label='TPS'),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=C_G_LIGHT, markersize=9,  label='Gerobak'),
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor=C_T_LIGHT, markersize=10, label='Truk'),
]
leg = ax.legend(handles=leg_el, loc='lower left', fontsize=8, framealpha=0.60, facecolor='#080f1e', edgecolor=C_BORDER, labelcolor=C_TXT1)

# ═══════════════════════════════════════════════════════════════════
# 7. PANEL KANAN (Y-Koordinat Diperbaiki)
# ═══════════════════════════════════════════════════════════════════
axp.axis('off')

def bar_h(a, x, y, w, h, pct, c_lo, c_hi, bg='#0e1a2e'):
    a.add_patch(mpatches.FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.004', facecolor=bg, edgecolor=C_BORDER, linewidth=0.7, zorder=3))
    if pct > 0.005:
        a.add_patch(mpatches.FancyBboxPatch((x,y), w*pct, h, boxstyle='round,pad=0.004', facecolor=lerp_c(c_lo, c_hi, pct), edgecolor='none', zorder=4))

def render_panel(fi, t_sim):
    axp.cla()
    axp.set_xlim(0, 1); axp.set_ylim(0, 1); axp.axis('off')
    axp.set_facecolor(BG_PANEL)

    # ── Header ──
    axp.text(0.50, 0.985, "MANAJEMEN SAMPAH", ha='center', va='top', color=C_TXT1, fontsize=12, fontweight='bold')
    axp.text(0.50, 0.965, "Sistem Monitoring Real-Time", ha='center', va='top', color=C_TXT2, fontsize=9)
    axp.axhline(0.950, xmin=0.03, xmax=0.97, color=C_BORDER, linewidth=0.8)

    # ── Display Jam ──
    jam  = int(t_sim)//60; mnt = int(t_sim)%60
    if   t_sim < T_G_AKHIR: c_jam = C_GOLD;  fase = "Gerobak & Truk aktif"
    elif t_sim < T_T_AKHIR: c_jam = C_WARN;  fase = "Hanya Truk aktif"
    else:                    c_jam = C_ERROR; fase = "Semua selesai"

    # Box Clock (Tinggi=0.055, y_center=0.910, top=0.9375, bottom=0.8825)
    axp.add_patch(mpatches.FancyBboxPatch((0.05, 0.8825), 0.90, 0.055, boxstyle='round,pad=0.01', facecolor='#0a0e1a', edgecolor=c_jam, linewidth=1.8, zorder=2))
    axp.text(0.50, 0.910, f"{jam:02d}:{mnt:02d}", ha='center', va='center', color=c_jam, fontsize=28, fontweight='bold', fontfamily='monospace')
    axp.text(0.50, 0.865, fase, ha='center', va='center', color=C_TXT2, fontsize=8.5)

    axp.axhline(0.850, xmin=0.03, xmax=0.97, color=C_BORDER, linewidth=0.5)
    axp.text(0.05, 0.835, "Gerobak: 06:00 - 15:00", ha='left', va='center', color=C_OK if t_sim <= T_G_AKHIR else C_ERROR, fontsize=8)
    axp.text(0.95, 0.835, "Truk: 08:00 - 17:00", ha='right', va='center', color=(C_TXT2 if t_sim < T_T_MULAI else (C_OK if t_sim <= T_T_AKHIR else C_ERROR)), fontsize=8)

    axp.axhline(0.820, xmin=0.03, xmax=0.97, color=C_BORDER, linewidth=0.7)
    
    # ── GEROBAK ──
    g_aktif = t_sim <= T_G_AKHIR
    axp.text(0.05, 0.805, "GEROBAK", ha='left', va='center', color='#fca5a5', fontsize=9.5, fontweight='bold')
    axp.text(0.95, 0.805, "AKTIF" if g_aktif else "SELESAI OPERASI", ha='right', va='center', color=C_OK if g_aktif else C_ERROR, fontsize=8)

    y0_g = 0.765
    delta_g = 0.065
    for g_id in range(JUMLAH_GEROBAK):
        y = y0_g - g_id * delta_g
        t_eff = min(t_sim, T_G_AKHIR)
        load  = get_load(gerobak_routes[g_id], t_eff)
        pct   = load / KAP_GEROBAK if KAP_GEROBAK > 0 else 0
        c_g   = lerp_c(C_G_LIGHT, C_G_DARK, pct)

        axp.add_patch(mpatches.FancyBboxPatch((0.04, y), 0.10, 0.040, boxstyle='round,pad=0.005', facecolor=c_g, edgecolor=G_EDGE_COLORS[g_id], linewidth=0.9, zorder=4))
        axp.text(0.09, y+0.020, f"G{g_id}", ha='center', va='center', color='#0a0e1a', fontsize=8, fontweight='bold', zorder=5)

        bar_h(axp, 0.18, y+0.008, 0.58, 0.024, pct, C_G_LIGHT, C_G_DARK)
        axp.text(0.78, y+0.020, f"{load:.1f}/{KAP_GEROBAK:.0f}kg", ha='left', va='center', color=C_TXT1, fontsize=8)
        c_dot = (C_OK if pct < 0.95 else C_WARN) if g_aktif else C_TXT2
        axp.scatter([0.96], [y+0.020], s=80, marker='o', color=c_dot, zorder=5)

    # ── TRUK ──
    y_truk_top = y0_g - (JUMLAH_GEROBAK-1) * delta_g - 0.035 # ~0.470
    axp.axhline(y_truk_top, xmin=0.03, xmax=0.97, color=C_BORDER, linewidth=0.7)
    
    t_aktif = T_T_MULAI <= t_sim <= T_T_AKHIR
    axp.text(0.05, y_truk_top - 0.015, "TRUK", ha='left', va='center', color='#c4b5fd', fontsize=9.5, fontweight='bold')
    axp.text(0.95, y_truk_top - 0.015, "BELUM MULAI" if t_sim < T_T_MULAI else ("AKTIF" if t_aktif else "SELESAI"), ha='right', va='center', color=(C_TXT2 if t_sim < T_T_MULAI else (C_OK if t_aktif else C_ERROR)), fontsize=8)

    y0_t = y_truk_top - 0.055 # ~0.415
    delta_t = 0.065
    for t_id in range(JUMLAH_TRUK):
        y = y0_t - t_id * delta_t
        t_eff = max(min(t_sim, T_T_AKHIR), T_T_MULAI)
        load  = get_load(truk_routes[t_id], t_eff) if t_sim >= T_T_MULAI else 0.0
        pct   = load / KAP_TRUK if KAP_TRUK > 0 else 0
        c_t   = lerp_c(C_T_LIGHT, C_T_DARK, pct)

        axp.scatter([0.09], [y+0.015], s=250, marker='^', color=c_t, edgecolors=T_EDGE_COLORS[t_id], linewidths=0.9, zorder=4)
        axp.text(0.09, y+0.040, f"T{t_id}", ha='center', va='bottom', color=T_EDGE_COLORS[t_id], fontsize=8, fontweight='bold', zorder=5)

        bar_h(axp, 0.18, y+0.008, 0.58, 0.024, pct, C_T_LIGHT, C_T_DARK)
        axp.text(0.78, y+0.020, f"{load:.1f}/{KAP_TRUK:.0f}kg", ha='left', va='center', color=C_TXT1, fontsize=8)
        c_dot = (C_OK if t_aktif else C_TXT2)
        axp.scatter([0.96], [y+0.020], s=80, marker='o', color=c_dot, zorder=5)

    # ── STATUS RUMAH ──
    y_stat = y0_t - (JUMLAH_TRUK-1) * delta_t - 0.035 # ~0.315
    axp.axhline(y_stat, xmin=0.03, xmax=0.97, color=C_BORDER, linewidth=0.7)
    axp.text(0.05, y_stat - 0.015, "STATUS RUMAH", ha='left', va='center', color=C_TXT2, fontsize=9.5, fontweight='bold')

    clr_now  = color_frames[fi]
    n_sls    = clr_now.count(C_SELESAI)
    n_seb    = clr_now.count(C_SEBAGIAN)
    n_blm    = clr_now.count(C_BELUM)
    pct_tot  = n_sls / JUMLAH_RUMAH

    items = [
        (C_BORDER_B, C_BELUM,    f"Belum dikunjungi   {n_blm:3d}"),
        (C_BORDER_S, C_SEBAGIAN, f"Sebagian terangkut {n_seb:3d}"),
        (C_BORDER_D, C_SELESAI,  f"Selesai            {n_sls:3d}"),
    ]
    y_item = y_stat - 0.045
    delta_s = 0.035
    for (ec, fc, txt) in items:
        axp.scatter([0.07], [y_item], s=180, marker='o', color=fc, edgecolors=ec, linewidths=1.2, zorder=5)
        axp.text(0.13, y_item, txt, ha='left', va='center', color=C_TXT1, fontsize=8.5)
        y_item -= delta_s

    # ── PROGRESS BAR ──
    y_pb = y_item - 0.010
    axp.text(0.05, y_pb, f"Progress: {pct_tot*100:.0f}%  ({n_sls}/{JUMLAH_RUMAH})", ha='left', va='center', color=C_BORDER_D if pct_tot > 0.5 else C_TXT2, fontsize=9, fontweight='bold')
    bar_h(axp, 0.05, y_pb-0.030, 0.90, 0.020, pct_tot, C_BELUM, C_SELESAI)

    kumpul_smp = sum(amt for (t_e, _, amt) in collect_events if t_e <= t_sim)
    y_info = y_pb - 0.050
    axp.axhline(y_info, xmin=0.03, xmax=0.97, color=C_BORDER, linewidth=0.5)
    axp.text(0.05, y_info - 0.015, f"Terkumpul: {kumpul_smp:.0f} / {total_sampah} KG", ha='left', va='center', color=C_TXT2, fontsize=8.5)

# ═══════════════════════════════════════════════════════════════════
# 8. UPDATE ANIMASI & KONTROL
# ═══════════════════════════════════════════════════════════════════
is_paused = False

def update(fi):
    global is_paused
    t = sim_times[fi]

    clr = color_frames[fi]
    sc_rumah.set_facecolor(clr)
    ec = [C_BORDER_D if c == C_SELESAI else (C_BORDER_S if c == C_SEBAGIAN else C_BORDER_B) for c in clr]
    sc_rumah.set_edgecolor(ec)

    g_aktif = t <= T_G_AKHIR
    for g_id in range(JUMLAH_GEROBAK):
        sc, gw = sc_g[g_id]; lb = lb_g[g_id]
        t_eff = min(t, T_G_AKHIR)
        gx, gy = get_pos(gerobak_routes[g_id], t_eff)
        sc.set_offsets([[gx, gy]]); gw.set_offsets([[gx, gy]])
        load = get_load(gerobak_routes[g_id], t_eff)
        pct  = load / KAP_GEROBAK if KAP_GEROBAK > 0 else 0
        sc.set_facecolor([lerp_c(C_G_LIGHT, C_G_DARK, pct)])
        alpha = 1.0 if g_aktif else 0.25
        sc.set_alpha(alpha); gw.set_alpha(0.0 if not g_aktif else 0.15)
        lb.set_position((gx, gy+3.8)); lb.set_alpha(alpha)

    for t_id in range(JUMLAH_TRUK):
        sc, tw = sc_t[t_id]; lb = lb_t[t_id]
        if t < T_T_MULAI:
            tx, ty = truk_routes[t_id][0]["pos"]; alpha_t = 0.35
        elif t > T_T_AKHIR:
            tx, ty = get_pos(truk_routes[t_id], T_T_AKHIR); alpha_t = 0.25
        else:
            tx, ty = get_pos(truk_routes[t_id], t); alpha_t = 1.0
        sc.set_offsets([[tx, ty]]); tw.set_offsets([[tx, ty]])
        load = get_load(truk_routes[t_id], max(min(t, T_T_AKHIR), T_T_MULAI)) if t >= T_T_MULAI else 0.0
        pct  = load / KAP_TRUK if KAP_TRUK > 0 else 0
        sc.set_facecolor([lerp_c(C_T_LIGHT, C_T_DARK, pct)])
        sc.set_alpha(alpha_t); tw.set_alpha(alpha_t * 0.15)
        lb.set_position((tx, ty+4.5)); lb.set_alpha(alpha_t)

    h = int(t)//60; m = int(t)%60
    jam_box.set_text(f"{h:02d}:{m:02d}")
    c_j = C_GOLD if t <= T_G_AKHIR else (C_WARN if t <= T_T_AKHIR else C_ERROR)
    jam_box.set_color(c_j)
    jam_box.get_bbox_patch().set_edgecolor(c_j)

    render_panel(fi, t)
    
    # Auto-pause at the end of the simulation
    if fi == N_FRAMES - 1:
        ani.pause()
        is_paused = True
        btn_pause.label.set_text("▶ Run")
        fig.canvas.draw_idle()

    return ([sc_rumah, jam_box] + [s for s,_ in sc_g] + [s for s,_ in sc_t] + lb_g + lb_t)

fig.text(0.355, 0.998, "Simulasi Pengumpulan Sampah  |  100 Rumah  |  5 Gerobak (06-15)  |  2 Truk (08-17)", ha='center', va='top', color=C_TXT2, fontsize=10, fontweight='bold')

ax_pause = fig.add_axes([0.83, 0.015, 0.15, 0.04])
btn_pause = Button(ax_pause, '⏸ Pause', color='#1e293b', hovercolor='#334155')
btn_pause.label.set_color(C_TXT1)
btn_pause.label.set_fontweight('bold')

def toggle_pause(event):
    global is_paused
    if is_paused:
        ani.resume()
        btn_pause.label.set_text("⏸ Pause")
        is_paused = False
    else:
        ani.pause()
        btn_pause.label.set_text("▶ Run")
        is_paused = True
    fig.canvas.draw_idle()

btn_pause.on_clicked(toggle_pause)

print(f"\n[Animasi] {N_FRAMES} frame  |  ~{N_FRAMES/FPS:.0f} detik  |  {FPS} fps")
ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=int(1000 / FPS), blit=False, repeat=True)
plt.show()
