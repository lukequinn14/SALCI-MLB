<!DOCTYPE html>
<html>
<head>
    <title>SALCI v4.0 - Full Updated Code</title>
</head>
<body>
<pre><code>#!/usr/bin/env python3
"""
SALCI v4.0 - MLB Decision Engine (Pitcher Strikeouts + Hitter Matchups)
Enhanced with SALCI v2 Decision Layer, Workload Profile, Yesterday Reflection,
Pitch Zone Heatmaps, and Explainable Insights

Run with:
streamlit run mlb_salci_full.py

This is the complete, production-ready refactored codebase.
All original SALCI logic is untouched.
New modular layers added in parallel exactly as specified.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="SALCI v4.0 - MLB Decision Engine",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS (unchanged + minor v2 enhancements)
# ----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3a5f, #2e5a8f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .lineup-confirmed {
        background: linear-gradient(135deg, #10b981, #34d399);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .lineup-pending {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .hot-streak {
        background: linear-gradient(135deg, #ff6b6b, #ffa500);
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .cold-streak {
        background: linear-gradient(135deg, #4a90d9, #67b8de);
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .elite {
        color: #10b981;
        font-weight: bold;
    }
    .strong {
        color: #22c55e;
        font-weight: bold;
    }
    .average {
        color: #eab308;
        font-weight: bold;
    }
    .below {
        color: #f97316;
        font-weight: bold;
    }
    .poor {
        color: #ef4444;
        font-weight: bold;
    }
    .batting-order {
        background: #f0f9ff;
        border-left: 3px solid #3b82f6;
        padding: 0.2rem 0.5rem;
        font-weight: bold;
        border-radius: 3px;
    }
    .stat-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .matchup-good {
        background-color: #d4edda;
    }
    .matchup-neutral {
        background-color: #fff3cd;
    }
    .matchup-bad {
        background-color: #f8d7da;
    }
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .salci-watermark {
        font-size: 0.7rem;
        color: #999;
        text-align: right;
        margin-top: 0.5rem;
    }
    .v2-insight {
        background: linear-gradient(135deg, #6366f1, #a5b4fc);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    div[data-testid="stHorizontalBlock"] div {
        padding: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Configuration (unchanged)
# ----------------------------
WEIGHT_PRESETS = {
    "balanced": {
        "name": "⚖️ Balanced",
        "desc": "Equal weight to pitcher and matchup",
        "weights": {
            "K9": 0.18, "K_percent": 0.18, "K/BB": 0.14, "P/IP": 0.10,
            "OppK%": 0.22, "OppContact%": 0.18
        }
    },
    "pitcher": {
        "name": "💪 Pitcher Heavy",
        "desc": "Focus on pitcher's K ability",
        "weights": {
            "K9": 0.28, "K_percent": 0.25, "K/BB": 0.20, "P/IP": 0.12,
            "OppK%": 0.08, "OppContact%": 0.07
        }
    },
    "matchup": {
        "name": "🎯 Matchup Heavy",
        "desc": "Focus on opponent K tendencies",
        "weights": {
            "K9": 0.12, "K_percent": 0.10, "K/BB": 0.08, "P/IP": 0.08,
            "OppK%": 0.32, "OppContact%": 0.30
        }
    }
}

BOUNDS = {
    "K9": (6.0, 13.0, True),
    "K_percent": (0.15, 0.38, True),
    "K/BB": (1.5, 7.0, True),
    "P/IP": (13, 18, False),
    "OppK%": (0.18, 0.28, True),
    "OppContact%": (0.70, 0.85, False)
}

HITTER_BOUNDS = {
    "avg": (0.200, 0.350),
    "ops": (0.600, 1.000),
    "slg": (0.350, 0.600),
    "k_rate": (0.30, 0.15),
    "hr": (0, 3),
}

COLORS = {
    "elite": "#10b981",
    "strong": "#3b82f6",
    "average": "#eab308",
    "below": "#f97316",
    "poor": "#ef4444",
    "hot": "#D85A30",
    "cold": "#4a90d9",
    "primary": "#1e3a5f",
    "secondary": "#7F77DD",
    "accent": "#1D9E75"
}

# ----------------------------
# Helper Functions (unchanged)
# ----------------------------
def normalize(val: float, min_val: float, max_val: float, higher_is_better: bool = True) -> float:
    norm = np.clip((val - min_val) / (max_val - min_val), 0, 1)
    return norm if higher_is_better else (1 - norm)

def get_blend_weights(games_played: int) -> Tuple[float, float]:
    if games_played < 3:
        return 0.2, 0.8
    elif games_played < 7:
        return 0.4, 0.6
    elif games_played < 15:
        return 0.6, 0.4
    return 0.8, 0.2

def get_rating(salci: float) -> Tuple[str, str, str]:
    if salci >= 75:
        return "Elite", "🔥", "elite"
    elif salci >= 60:
        return "Strong", "✅", "strong"
    elif salci >= 45:
        return "Average", "➖", "average"
    elif salci >= 30:
        return "Below Avg", "⚠️", "below"
    return "Poor", "❌", "poor"

def get_hitter_rating(score: float) -> Tuple[str, str]:
    if score >= 80:
        return "🔥 On Fire", "hot-streak"
    elif score >= 60:
        return "✅ Hot", "strong"
    elif score >= 40:
        return "➖ Normal", "average"
    elif score >= 20:
        return "❄️ Cold", "cold-streak"
    return "🥶 Ice Cold", "poor"

def get_salci_color(salci: float) -> str:
    if salci >= 75:
        return COLORS["elite"]
    elif salci >= 60:
        return COLORS["strong"]
    elif salci >= 45:
        return COLORS["average"]
    elif salci >= 30:
        return COLORS["below"]
    return COLORS["poor"]

# ----------------------------
# NEW: SALCI v2 Decision Layer (added exactly as specified + tiny compatibility fixes)
# ----------------------------
def calculate_stuff_score(pitcher_stats: Dict) -> float:
    """Raw pitch quality - untouched original metrics"""
    return np.mean([
        pitcher_stats.get("K9", 0),
        pitcher_stats.get("K_percent", 0),
        pitcher_stats.get("K/BB", 0)
    ])

def calculate_location_score(stats_dict: Dict) -> float:
    """Command + efficiency proxy (merged stats for OppContact%)"""
    return np.mean([
        1 / stats_dict.get("P/IP", 1),  # lower is better
        stats_dict.get("OppContact%", 0)
    ])

def calculate_matchup_score(pitcher_stats: Dict, opponent_stats: Dict) -> float:
    """Opponent-based interaction"""
    return np.mean([
        opponent_stats.get("OppK%", 0),
        1 - opponent_stats.get("OppContact%", 0)
    ])

def generate_matchup_explanation(stuff: float, location: float, matchup: float) -> str:
    if stuff > location and stuff > matchup:
        return "Pitcher wins on raw stuff, but execution may vary."
    elif location > stuff and location > matchup:
        return "Command-driven edge — relies on precision and control."
    elif matchup > stuff:
        return "Matchup-driven advantage based on opponent tendencies."
    else:
        return "Balanced profile with no dominant edge."

def calculate_workload_profile(game_logs: pd.DataFrame) -> Dict:
    """Workload / Innings Layer - exactly as specified"""
    if game_logs.empty:
        return {"season_avg_ip": 0.0, "last5_avg_ip": 0.0, "trend": 0.0}
    last5 = game_logs.tail(5)
    return {
        "season_avg_ip": game_logs["IP"].mean(),
        "last5_avg_ip": last5["IP"].mean(),
        "trend": last5["IP"].mean() - game_logs["IP"].mean()
    }

def evaluate_model_performance(results_df: pd.DataFrame) -> Dict:
    """Yesterday Reflection Engine - exactly as specified"""
    if results_df.empty or len(results_df) == 0:
        return {
            "avg_ip": 0.0,
            "avg_k_diff": 0.0,
            "overperformers": 0,
            "underperformers": 0
        }
    return {
        "avg_ip": results_df["IP"].mean(),
        "avg_k_diff": (results_df["Ks"] - results_df["proj_K"]).mean(),
        "overperformers": results_df[results_df["Ks"] > results_df["proj_K"]].shape[0],
        "underperformers": results_df[results_df["Ks"] < results_df["proj_K"]].shape[0]
    }

def estimate_hit_probability(pitcher_profile: Dict, hitter_profile: Dict) -> float:
    """Hit Probability Model - exactly as specified (simple but powerful)"""
    zone_overlap = abs(pitcher_profile.get("zone", 0.5) - hitter_profile.get("zone", 0.5))
    contact_factor = hitter_profile.get("contact", 0.25) - pitcher_profile.get("whiff", 0.25)
    return np.clip(0.5 + contact_factor - zone_overlap, 0, 1)

# ----------------------------
# NEW: Pitch Zone Heatmap (clean Plotly integration)
# ----------------------------
def create_zone_heatmap(data: pd.DataFrame, title: str) -> go.Figure:
    """Heat Map System - exactly as specified"""
    if data.empty:
        # Demo data when real Statcast not yet integrated
        data = pd.DataFrame({
            "plate_x": np.random.normal(0, 0.6, 800),
            "plate_z": np.random.normal(2.5, 1.1, 800),
        })
    fig = px.density_heatmap(
        data,
        x="plate_x",
        y="plate_z",
        nbinsx=30,
        nbinsy=30,
        title=title,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=400)
    return fig

# ----------------------------
# API Functions - Teams & Schedule (unchanged)
# ----------------------------
@st.cache_data(ttl=300)
def get_team_id_lookup() -> Dict[str, int]:
    url = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
    try:
        res = requests.get(url, timeout=10)
        return {team["name"]: team["id"] for team in res.json().get("teams", [])}
    except:
        return {}

@st.cache_data(ttl=60)
def get_games_by_date(date_str: str) -> List[Dict]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,lineups,team"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if not data.get("dates"):
            return []

        games = []
        for g in data["dates"][0]["games"]:
            game_info = {
                "game_pk": g.get("gamePk"),
                "game_time": g.get("gameDate"),
                "status": g.get("status", {}).get("abstractGameState", ""),
                "detailed_status": g.get("status", {}).get("detailedState", ""),
                "home_team": g["teams"]["home"]["team"]["name"],
                "away_team": g["teams"]["away"]["team"]["name"],
                "home_team_id": g["teams"]["home"]["team"]["id"],
                "away_team_id": g["teams"]["away"]["team"]["id"],
                "lineups_available": False
            }

            for side in ["home", "away"]:
                pp = g["teams"][side].get("probablePitcher")
                if pp:
                    game_info[f"{side}_pitcher"] = pp.get("fullName", "TBD")
                    game_info[f"{side}_pid"] = pp.get("id")
                    game_info[f"{side}_pitcher_hand"] = pp.get("pitchHand", {}).get("code", "R")
                else:
                    game_info[f"{side}_pitcher"] = "TBD"
                    game_info[f"{side}_pid"] = None
                    game_info[f"{side}_pitcher_hand"] = "R"

            games.append(game_info)
        return games
    except Exception as e:
        st.error(f"Error fetching games: {e}")
        return []

@st.cache_data(ttl=60)
def get_game_boxscore(game_pk: int) -> Optional[Dict]:
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        res = requests.get(url, timeout=15)
        return res.json()
    except:
        return None

def get_confirmed_lineup(game_pk: int, team_side: str) -> Tuple[List[Dict], bool]:
    data = get_game_boxscore(game_pk)
    if not data:
        return [], False

    try:
        game_data = data.get("gameData", {})
        live_data = data.get("liveData", {})
        boxscore = live_data.get("boxscore", {})

        teams = boxscore.get("teams", {})
        team_data = teams.get(team_side, {})

        batting_order = team_data.get("battingOrder", [])

        if not batting_order:
            return [], False

        players = team_data.get("players", {})
        lineup = []

        for i, player_id in enumerate(batting_order):
            player_key = f"ID{player_id}"
            player_info = players.get(player_key, {})
            person = player_info.get("person", {})
            position = player_info.get("position", {})

            all_players = game_data.get("players", {})
            full_player = all_players.get(player_key, {})
            bat_side = full_player.get("batSide", {}).get("code", "R")

            lineup.append({
                "id": player_id,
                "name": person.get("fullName", "Unknown"),
                "position": position.get("abbreviation", ""),
                "batting_order": i + 1,
                "bat_side": bat_side
            })

        return lineup, len(lineup) >= 9
    except Exception as e:
        return [], False

# ----------------------------
# API Functions - Pitchers (unchanged + NEW game logs for workload)
# ----------------------------
@st.cache_data(ttl=300)
def get_player_season_stats(player_id: int, season: int) -> Optional[Dict]:
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&season={season}&group=pitching"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if data.get("stats") and data["stats"][0].get("splits"):
            return data["stats"][0]["splits"][0]["stat"]
    except:
        pass
    return None

@st.cache_data(ttl=300)
def get_recent_pitcher_stats(player_id: int, num_games: int = 7) -> Optional[Dict]:
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&group=pitching"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if not data.get("stats") or not data["stats"][0].get("splits"):
            return None

        games = sorted(data["stats"][0]["splits"],
                       key=lambda x: x.get("date", ""), reverse=True)[:num_games]
        if not games:
            return None

        totals = {"ip": 0, "so": 0, "bb": 0, "tbf": 0, "np": 0, "games": len(games)}

        for g in games:
            s = g.get("stat", {})
            ip_raw = str(s.get("inningsPitched", "0.0"))
            if "." in ip_raw:
                parts = ip_raw.split(".")
                ip = int(parts[0]) + int(parts[1]) / 3
            else:
                ip = float(ip_raw)

            totals["ip"] += ip
            totals["so"] += int(s.get("strikeOuts", 0))
            totals["bb"] += int(s.get("baseOnBalls", 0))
            totals["tbf"] += int(s.get("battersFaced", 0))
            totals["np"] += int(s.get("numberOfPitches", 0))

        if totals["ip"] == 0 or totals["tbf"] == 0:
            return None

        return {
            "K9": totals["so"] / totals["ip"] * 9,
            "K_percent": totals["so"] / totals["tbf"],
            "K/BB": totals["so"] / totals["bb"] if totals["bb"] > 0 else totals["so"] * 2,
            "P/IP": totals["np"] / totals["ip"],
            "games_sampled": totals["games"],
            "total_so": totals["so"],
            "total_ip": totals["ip"]
        }
    except:
        pass
    return None

# NEW: Game logs for workload layer
@st.cache_data(ttl=300)
def get_pitcher_game_logs(player_id: int, num_games: int = 15) -> pd.DataFrame:
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&group=pitching"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if not data.get("stats") or not data["stats"][0].get("splits"):
            return pd.DataFrame()

        splits = data["stats"][0]["splits"][-num_games:]
        logs = []
        for g in splits:
            s = g.get("stat", {})
            ip_raw = str(s.get("inningsPitched", "0.0"))
            if "." in ip_raw:
                parts = ip_raw.split(".")
                ip = int(parts[0]) + int(parts[1]) / 3
            else:
                ip = float(ip_raw)
            logs.append({
                "date": g.get("date", ""),
                "IP": ip,
                "Ks": int(s.get("strikeOuts", 0))
            })
        return pd.DataFrame(logs)
    except:
        return pd.DataFrame()

def parse_season_stats(stats: Dict) -> Dict:
    if not stats:
        return {}

    ip_raw = str(stats.get("inningsPitched", "0.0"))
    if "." in ip_raw:
        parts = ip_raw.split(".")
        ip = int(parts[0]) + int(parts[1]) / 3
    else:
        ip = float(ip_raw)

    if ip == 0:
        return {}

    so = int(stats.get("strikeOuts", 0))
    bb = int(stats.get("baseOnBalls", 0))
    tbf = int(stats.get("battersFaced", 1))
    np_total = int(stats.get("numberOfPitches", 0))

    return {
        "K9": so / ip * 9,
        "K_percent": so / tbf,
        "K/BB": so / bb if bb > 0 else so * 2,
        "P/IP": np_total / ip if np_total > 0 else 15.0,
        "ERA": float(stats.get("era", 0)),
        "WHIP": float(stats.get("whip", 0))
    }

# Hitters API functions unchanged (kept exactly as before)
@st.cache_data(ttl=300)
def get_hitter_recent_stats(player_id: int, num_games: int = 7) -> Optional[Dict]:
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&group=hitting"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if not data.get("stats") or not data["stats"][0].get("splits"):
            return None

        games = sorted(data["stats"][0]["splits"],
                       key=lambda x: x.get("date", ""), reverse=True)[:num_games]
        if not games:
            return None

        totals = {
            "ab": 0, "hits": 0, "doubles": 0, "triples": 0, "hr": 0,
            "rbi": 0, "bb": 0, "so": 0, "sb": 0, "games": len(games)
        }

        game_results = []

        for g in games:
            s = g.get("stat", {})
            ab = int(s.get("atBats", 0))
            hits = int(s.get("hits", 0))

            totals["ab"] += ab
            totals["hits"] += hits
            totals["doubles"] += int(s.get("doubles", 0))
            totals["triples"] += int(s.get("triples", 0))
            totals["hr"] += int(s.get("homeRuns", 0))
            totals["rbi"] += int(s.get("rbi", 0))
            totals["bb"] += int(s.get("baseOnBalls", 0))
            totals["so"] += int(s.get("strikeOuts", 0))
            totals["sb"] += int(s.get("stolenBases", 0))

            if ab > 0:
                game_results.append({"date": g.get("date"), "hits": hits, "ab": ab})

        if totals["ab"] == 0:
            return None

        avg = totals["hits"] / totals["ab"]
        slg = (totals["hits"] + totals["doubles"] + 2 * totals["triples"] + 3 * totals["hr"]) / totals["ab"]
        obp = (totals["hits"] + totals["bb"]) / (totals["ab"] + totals["bb"]) if (totals["ab"] + totals["bb"]) > 0 else 0
        ops = obp + slg
        k_rate = totals["so"] / totals["ab"]

        hit_streak = 0
        for gr in game_results:
            if gr["hits"] > 0:
                hit_streak += 1
            else:
                break

        hitless_streak = 0
        for gr in game_results:
            if gr["hits"] == 0:
                hitless_streak += 1
            else:
                break

        return {
            "avg": avg,
            "obp": obp,
            "slg": slg,
            "ops": ops,
            "k_rate": k_rate,
            "hr": totals["hr"],
            "rbi": totals["rbi"],
            "sb": totals["sb"],
            "hits": totals["hits"],
            "ab": totals["ab"],
            "so": totals["so"],
            "games": totals["games"],
            "hit_streak": hit_streak,
            "hitless_streak": hitless_streak
        }

    except:
        pass
    return None

@st.cache_data(ttl=3600)
def get_hitter_season_stats(player_id: int, season: int = 2025) -> Optional[Dict]:
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&season={season}&group=hitting"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if data.get("stats") and data["stats"][0].get("splits"):
            s = data["stats"][0]["splits"][0]["stat"]
            ab = int(s.get("atBats", 1))
            return {
                "avg": float(s.get("avg", 0)),
                "obp": float(s.get("obp", 0)),
                "slg": float(s.get("slg", 0)),
                "ops": float(s.get("ops", 0)),
                "hr": int(s.get("homeRuns", 0)),
                "rbi": int(s.get("rbi", 0)),
                "k_rate": int(s.get("strikeOuts", 0)) / ab if ab > 0 else 0,
                "ab": ab
            }
    except:
        pass
    return None

@st.cache_data(ttl=300)
def get_team_batting_stats(team_id: int, days: int = 14) -> Optional[Dict]:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}"
           f"&startDate={start_date.strftime('%Y-%m-%d')}&endDate={end_date.strftime('%Y-%m-%d')}")

    try:
        res = requests.get(url, timeout=10)
        dates = res.json().get("dates", [])
        games = [g["gamePk"] for d in dates for g in d.get("games", [])]

        totals = {"pa": 0, "so": 0, "hits": 0, "ab": 0}
        games_counted = 0

        for gid in games[:7]:
            b_url = f"https://statsapi.mlb.com/api/v1/game/{gid}/boxscore"
            b_res = requests.get(b_url, timeout=10)
            box = b_res.json().get("teams", {})

            for side in ["home", "away"]:
                if box.get(side, {}).get("team", {}).get("id") == team_id:
                    stats = box[side].get("teamStats", {}).get("batting", {})
                    totals["so"] += int(stats.get("strikeOuts", 0))
                    totals["pa"] += int(stats.get("plateAppearances", 0))
                    totals["hits"] += int(stats.get("hits", 0))
                    totals["ab"] += int(stats.get("atBats", 0))
                    games_counted += 1
                    break

        if totals["pa"] == 0:
            return None

        return {
            "OppK%": totals["so"] / totals["pa"],
            "OppContact%": totals["hits"] / totals["ab"] if totals["ab"] > 0 else 0.25,
            "games_sampled": games_counted
        }

    except:
        pass
    return None

@st.cache_data(ttl=3600)
def get_team_season_batting(team_id: int, season: int = 2025) -> Optional[Dict]:
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&season={season}&group=hitting"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if data.get("stats") and data["stats"][0].get("splits"):
            stats = data["stats"][0]["splits"][0]["stat"]
            pa = int(stats.get("plateAppearances", 1))
            so = int(stats.get("strikeOuts", 0))
            ab = int(stats.get("atBats", 1))
            hits = int(stats.get("hits", 0))
            return {"OppK%": so / pa, "OppContact%": hits / ab}
    except:
        pass
    return None

# ----------------------------
# SALCI Computation (COMPLETELY UNCHANGED - as per architectural rule)
# ----------------------------
def compute_salci(
    pitcher_recent: Optional[Dict],
    pitcher_baseline: Optional[Dict],
    opp_recent: Optional[Dict],
    opp_baseline: Optional[Dict],
    weights: Dict,
    games_played: int = 5
) -> Tuple[Optional[float], Dict, List[str]]:
    recent_w, baseline_w = get_blend_weights(games_played)

    pitcher_stats = {}
    for metric in ["K9", "K_percent", "K/BB", "P/IP"]:
        recent_val = pitcher_recent.get(metric) if pitcher_recent else None
        baseline_val = pitcher_baseline.get(metric) if pitcher_baseline else None

        if recent_val is not None and baseline_val is not None:
            pitcher_stats[metric] = recent_w * recent_val + baseline_w * baseline_val
        elif recent_val is not None:
            pitcher_stats[metric] = recent_val
        elif baseline_val is not None:
            pitcher_stats[metric] = baseline_val

    opp_stats = {}
    for metric in ["OppK%", "OppContact%"]:
        recent_val = opp_recent.get(metric) if opp_recent else None
        baseline_val = opp_baseline.get(metric) if opp_baseline else None

        if recent_val is not None and baseline_val is not None:
            opp_stats[metric] = recent_w * recent_val + baseline_w * baseline_val
        elif recent_val is not None:
            opp_stats[metric] = recent_val
        elif baseline_val is not None:
            opp_stats[metric] = baseline_val

    score = 0.0
    total_weight = 0.0
    breakdown = {}
    missing = []
    all_stats = {**pitcher_stats, **opp_stats}

    for metric, weight in weights.items():
        if metric in all_stats:
            val = all_stats[metric]
            bounds = BOUNDS.get(metric)
            if bounds:
                min_v, max_v, higher_better = bounds
                norm_val = normalize(val, min_v, max_v, higher_better)
                score += weight * norm_val
                total_weight += weight
                breakdown[metric] = {"raw": val, "norm": norm_val, "weight": weight}
        elif weight > 0.05:
            missing.append(metric)

    if total_weight == 0:
        return None, {}, missing

    return round((score / total_weight) * 100, 1), breakdown, missing

def compute_hitter_score(recent: Dict, baseline: Dict = None) -> float:
    if not recent:
        return 50

    score = 0
    weights_total = 0

    if recent.get("avg"):
        avg_score = normalize(recent["avg"], 0.180, 0.380, True) * 100
        score += avg_score * 0.25
        weights_total += 0.25

    if recent.get("ops"):
        ops_score = normalize(recent["ops"], 0.550, 1.100, True) * 100
        score += ops_score * 0.25
        weights_total += 0.25

    if recent.get("k_rate"):
        k_score = normalize(recent["k_rate"], 0.35, 0.10, False) * 100
        score += k_score * 0.15
        weights_total += 0.15

    if recent.get("hit_streak", 0) >= 3:
        streak_bonus = min(recent["hit_streak"] * 3, 15)
        score += streak_bonus
        weights_total += 0.15

    if recent.get("hitless_streak", 0) >= 2:
        hitless_penalty = min(recent["hitless_streak"] * 5, 20)
        score -= hitless_penalty

    if recent.get("hr", 0) >= 1:
        score += min(recent["hr"] * 5, 15)
        weights_total += 0.10

    if weights_total == 0:
        return 50

    return max(0, min(100, score / weights_total * 1.1))

def project_lines(salci: float, base_k9: float = 9.0) -> Dict:
    expected = (base_k9 * 5.5 / 9) * (0.7 + (salci / 100) * 0.6)

    lines = {}
    for k in range(3, 9):
        diff = k - expected
        if diff <= -2:
            prob = 92
        elif diff <= -1:
            prob = 80
        elif diff <= 0:
            prob = 65
        elif diff <= 1:
            prob = 45
        elif diff <= 2:
            prob = 28
        else:
            prob = 15

        prob = max(5, min(95, prob + (salci - 50) / 10))
        lines[k] = round(prob)

    return {"expected": round(expected, 1), "lines": lines}

def get_matchup_grade(hitter_k_rate: float, pitcher_k_pct: float,
                      hitter_hand: str, pitcher_hand: str) -> Tuple[str, str]:
    platoon_adv = 0
    if hitter_hand == "L" and pitcher_hand == "R":
        platoon_adv = 10
    elif hitter_hand == "R" and pitcher_hand == "L":
        platoon_adv = 10
    elif hitter_hand == pitcher_hand:
        platoon_adv = -5

    k_matchup = 0
    if hitter_k_rate < 0.18 and pitcher_k_pct > 0.28:
        k_matchup = 15
    elif hitter_k_rate > 0.28 and pitcher_k_pct > 0.28:
        k_matchup = -15
    elif hitter_k_rate < 0.20:
        k_matchup = 10
    elif hitter_k_rate > 0.30:
        k_matchup = -10

    total = 50 + platoon_adv + k_matchup

    if total >= 65:
        return "🟢 Favorable", "matchup-good"
    elif total >= 45:
        return "🟡 Neutral", "matchup-neutral"
    else:
        return "🔴 Tough", "matchup-bad"

# ----------------------------
# UI Render Helpers (original + NEW v2 insight renderer)
# ----------------------------
def render_pitcher_card(result: Dict):
    rating_label, emoji, css_class = get_rating(result["salci"])
    pitcher = result["pitcher"]
    opponent = result["opponent"]
    salci = result["salci"]
    expected = result.get("expected", 0)
    pitcher_hand = result.get("pitcher_hand", "R")
    lineup_badge = "✅ Confirmed" if result.get("lineup_confirmed") else "⏳ Pending"

    st.markdown(f"""
    <div class="stat-card">
        <h4>{emoji} {pitcher} ({pitcher_hand}) vs {opponent}</h4>
        <p><span class="{css_class}">{rating_label}</span> | SALCI: <strong>{salci}</strong> | {lineup_badge}</p>
        <p>Expected Ks: <strong>{expected}</strong></p>
    </div>
    """, unsafe_allow_html=True)

def render_v2_insights(result: Dict):
    """NEW: Clean integration of SALCI v2 + Workload"""
    st.markdown(f"""
    <div class="v2-insight">
        <h4>🧠 SALCI v2 Matchup Insight</h4>
        <p><strong>Stuff:</strong> {result.get('stuff', 0):.2f} 
           <strong>Location:</strong> {result.get('location', 0):.2f} 
           <strong>Matchup:</strong> {result.get('matchup', 0):.2f}</p>
        <p><em>👉 {result.get('explanation', 'Balanced profile')}</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    wl = result.get("workload", {})
    st.markdown(f"""
    ### ⏱ Workload Profile
    - Season Avg IP: **{wl.get('season_avg_ip', 0):.1f}**
    - Last 5 Avg IP: **{wl.get('last5_avg_ip', 0):.1f}**
    - Trend: **{wl.get('trend', 0):+.2f}**
    """)

def render_hitter_card(result: Dict, show_batting_order: bool = False):
    name = result["name"]
    score = result["score"]
    recent = result["recent"]
    rating_label, css_class = get_hitter_rating(score)
    order = f"#{result.get('batting_order', '-')}" if show_batting_order else ""
    bats = result.get("bat_side", "R")
    team = result.get("team", "")

    st.markdown(f"""
    <div class="stat-card">
        <h4>{name} {order} ({bats})</h4>
        <p><span class="{css_class}">{rating_label}</span> | Score: <strong>{score:.1f}</strong></p>
        <p>{team} | AVG: {recent.get("avg", 0):.3f} | OPS: {recent.get("ops", 0):.3f}</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Chart Functions (original + NEW heatmap)
# ----------------------------
def create_pitcher_comparison_chart(pitcher_results: List[Dict]) -> go.Figure:
    if not pitcher_results:
        return None

    top_pitchers = sorted(pitcher_results, key=lambda x: x["salci"], reverse=True)[:10][::-1]
    names = [f"{p['pitcher'].split()[-1]} ({p.get('pitcher_hand', 'R')})" for p in top_pitchers]
    scores = [p["salci"] for p in top_pitchers]
    colors = [get_salci_color(s) for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=scores,
        orientation="h",
        marker_color=colors,
        text=[f"{s}" for s in scores],
        textposition="outside",
        textfont=dict(size=12, color="#333")
    ))

    fig.add_vline(x=75, line_dash="dash", line_color="#10b981", line_width=2,
                  annotation_text="Elite (75+)", annotation_position="top")
    fig.add_vline(x=60, line_dash="dot", line_color="#3b82f6", line_width=1,
                  annotation_text="Strong (60+)", annotation_position="bottom")

    fig.update_layout(
        title=dict(text="Today's Top SALCI Pitchers", font=dict(size=18)),
        xaxis_title="SALCI Score",
        yaxis_title="",
        xaxis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100]),
        height=400,
        margin=dict(l=100, r=50, t=80, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text="#SALCI",
            xref="paper", yref="paper",
            x=0, y=-0.15,
            showarrow=False,
            font=dict(size=11, color="#999")
        )]
    )
    return fig

def create_hitter_hotness_chart(hitter_results: List[Dict]) -> go.Figure:
    if not hitter_results:
        return None

    top_hitters = sorted(hitter_results, key=lambda x: x["score"], reverse=True)[:8]
    names = [f"{h['name'].split()[-1]} ({h.get('bat_side', 'R')})" for h in top_hitters]
    avgs = [h["recent"].get("avg", 0) for h in top_hitters]
    ops_vals = [h["recent"].get("ops", 0) for h in top_hitters]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="AVG (L7)",
        x=names,
        y=avgs,
        marker_color=COLORS["hot"],
        text=[f".{int(a*1000):03d}" for a in avgs],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="OPS (L7)",
        x=names,
        y=ops_vals,
        marker_color=COLORS["secondary"],
        text=[f"{o:.3f}" for o in ops_vals],
        textposition="outside"
    ))

    fig.update_layout(
        title=dict(text="Hottest Hitters (Last 7 Games)", font=dict(size=18)),
        yaxis_title="",
        barmode="group",
        height=350,
        margin=dict(l=50, r=50, t=80, b=80),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[dict(
            text="#SALCI",
            xref="paper", yref="paper",
            x=0, y=-0.22,
            showarrow=False,
            font=dict(size=11, color="#999")
        )]
    )
    return fig

def create_salci_breakdown_chart() -> go.Figure:
    labels = ["K/9", "K%", "Opp K%", "Opp Contact%", "K/BB", "P/IP"]
    values = [18, 18, 22, 18, 14, 10]
    colors_list = ["#3266ad", "#7F77DD", "#1D9E75", "#D85A30", "#eab308", "#888780"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors_list,
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=11)
    )])

    fig.update_layout(
        title=dict(text="SALCI Score Breakdown (Balanced Weights)", font=dict(size=16)),
        height=350,
        margin=dict(l=20, r=20, t=60, b=60),
        showlegend=False,
        annotations=[
            dict(text="SALCI\nScore", x=0.5, y=0.5, font=dict(size=14, color="#333"), showarrow=False),
            dict(text="#SALCI", xref="paper", yref="paper", x=0, y=-0.1, showarrow=False, font=dict(size=11, color="#999"))
        ]
    )
    return fig

def create_matchup_scatter(hitter_results: List[Dict]) -> go.Figure:
    if not hitter_results or len(hitter_results) < 3:
        return None

    k_rates = [h["recent"].get("k_rate", 0.22) * 100 for h in hitter_results]
    avgs = [h["recent"].get("avg", 0.250) for h in hitter_results]
    short_names = [f"{h['name'].split()[-1]} ({h.get('bat_side', 'R')})" for h in hitter_results]
    scores = [h["score"] for h in hitter_results]
    colors = [get_salci_color(s) if s >= 50 else COLORS["cold"] for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_rates,
        y=avgs,
        mode="markers+text",
        marker=dict(size=12, color=colors, line=dict(width=1, color="white")),
        text=short_names,
        textposition="top center",
        textfont=dict(size=8),
        hovertemplate="%{text}<br>K%%: %{x:.1f}%<br>AVG: %{y:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Hitter Contact vs Strikeout Profile", font=dict(size=18)),
        xaxis_title="K%",
        yaxis_title="AVG",
        height=420,
        margin=dict(l=50, r=50, t=80, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ----------------------------
# Main App
# ----------------------------
st.markdown('<h1 class="main-header">⚾ SALCI v4.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Pitcher Strikeout Predictions + Hitter Matchups + SALCI v2 Decision Engine + Workload + Yesterday Reflection</p>', unsafe_allow_html=True)

# Sidebar (unchanged)
with st.sidebar:
    st.header("⚙️ Settings")
    selected_date = st.date_input(
        "📅 Select Date",
        value=datetime.today(),
        min_value=datetime.today() - timedelta(days=7),
        max_value=datetime.today() + timedelta(days=7)
    )

    st.markdown("---")
    preset_key = st.selectbox(
        "Pitcher Model Weights",
        options=list(WEIGHT_PRESETS.keys()),
        format_func=lambda x: WEIGHT_PRESETS[x]["name"]
    )
    st.caption(WEIGHT_PRESETS[preset_key]["desc"])

    st.markdown("---")
    st.subheader("Filters")
    min_salci = st.slider("Min Pitcher SALCI", 0, 80, 0, 5)
    show_hitters = st.checkbox("Show Hitter Analysis", value=True)
    confirmed_only = st.checkbox("Confirmed Lineups Only", value=True)
    hot_hitters_only = st.checkbox("Hot Hitters Only (Score ≥ 60)", value=False)

    st.markdown("---")
    if st.button("🔄 Refresh Lineups", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("💡 Lineups are usually posted 1-2 hours before game time. Click refresh to get latest!")

    st.markdown("---")
    with st.expander("📊 About SALCI v4.0"):
        st.markdown("""
        **SALCI** = Strikeout Adjusted Lineup Confidence Index

        **New in v4.0:**
        - SALCI v2 Decision Layer (Stuff / Location / Matchup)
        - Workload & Innings Trend
        - Yesterday Reflection Engine (self-learning loop)
        - Pitch Zone Heatmaps (demo)
        - Explainable insights for every pitcher

        **Original SALCI unchanged** — now layered with decision intelligence.
        """)

date_str = selected_date.strftime("%Y-%m-%d")
weights = WEIGHT_PRESETS[preset_key]["weights"]

tab1, tab2, tab3, tab4 = st.tabs(["⚾ Pitcher Analysis", "🏏 Hitter Matchups", "🎯 Best Bets", "📊 Charts & Insights"])

with st.spinner("🔍 Fetching games and lineups..."):
    games = get_games_by_date(date_str)

if not games:
    st.warning(f"No games found for {date_str}")
    st.stop()

st.success(f"Found **{len(games)} games** for {selected_date.strftime('%A, %B %d, %Y')}")

# Lineups
lineup_status = {}
for game in games:
    game_pk = game["game_pk"]
    home_lineup, home_confirmed = get_confirmed_lineup(game_pk, "home")
    away_lineup, away_confirmed = get_confirmed_lineup(game_pk, "away")
    lineup_status[game_pk] = {
        "home": {"lineup": home_lineup, "confirmed": home_confirmed},
        "away": {"lineup": away_lineup, "confirmed": away_confirmed}
    }

confirmed_count = sum(
    1 for g in games
    if lineup_status[g["game_pk"]]["home"]["confirmed"] or lineup_status[g["game_pk"]]["away"]["confirmed"]
)

if confirmed_count == 0:
    st.warning("⏳ **No lineups confirmed yet.** Lineups are typically released 1-2 hours before game time.")
else:
    st.info(f"✅ **{confirmed_count} games** have confirmed lineups")

# Data processing
all_pitcher_results = []
all_hitter_results = []
progress = st.progress(0)

for i, game in enumerate(games):
    progress.progress((i + 1) / len(games))
    game_pk = game["game_pk"]
    game_lineups = lineup_status[game_pk]

    for side in ["home", "away"]:
        pitcher = game.get(f"{side}_pitcher", "TBD")
        pid = game.get(f"{side}_pid")
        pitcher_hand = game.get(f"{side}_pitcher_hand", "R")
        team = game.get(f"{side}_team")
        opp = game.get("away_team" if side == "home" else "home_team")
        opp_id = game.get("away_team_id" if side == "home" else "home_team_id")
        opp_side = "away" if side == "home" else "home"

        if not pid or pitcher == "TBD":
            continue

        p_recent = get_recent_pitcher_stats(pid, 7)
        p_baseline = parse_season_stats(get_player_season_stats(pid, 2025))
        opp_recent = get_team_batting_stats(opp_id, 14)
        opp_baseline = get_team_season_batting(opp_id, 2025)
        games_played = p_recent.get("games_sampled", 0) if p_recent else 0

        salci, breakdown, missing = compute_salci(
            p_recent, p_baseline, opp_recent, opp_baseline, weights, games_played
        )

        # SALCI v2 + Workload - parallel layer (original SALCI untouched)
        if p_recent or p_baseline:
            pitcher_stats = {}
            for metric in ["K9", "K_percent", "K/BB", "P/IP"]:
                recent_val = p_recent.get(metric) if p_recent else None
                baseline_val = p_baseline.get(metric) if p_baseline else None
                if recent_val is not None and baseline_val is not None:
                    pitcher_stats[metric] = 0.5 * recent_val + 0.5 * baseline_val
                elif recent_val is not None:
                    pitcher_stats[metric] = recent_val
                elif baseline_val is not None:
                    pitcher_stats[metric] = baseline_val

            opp_stats = {}
            for metric in ["OppK%", "OppContact%"]:
                recent_val = opp_recent.get(metric) if opp_recent else None
                baseline_val = opp_baseline.get(metric) if opp_baseline else None
                if recent_val is not None and baseline_val is not None:
                    opp_stats[metric] = 0.5 * recent_val + 0.5 * baseline_val
                elif recent_val is not None:
                    opp_stats[metric] = recent_val
                elif baseline_val is not None:
                    opp_stats[metric] = baseline_val

            full_stats = {**pitcher_stats, **opp_stats}
            stuff = calculate_stuff_score(pitcher_stats)
            location = calculate_location_score(full_stats)
            matchup = calculate_matchup_score(pitcher_stats, opp_stats)
            explanation = generate_matchup_explanation(stuff, location, matchup)
        else:
            stuff = location = matchup = 0.0
            explanation = "No data for v2 analysis"

        # Workload
        game_logs = get_pitcher_game_logs(pid, 15)
        workload = calculate_workload_profile(game_logs)

        if salci is not None:
            base_k9 = (p_baseline or p_recent or {}).get("K9", 9.0)
            proj = project_lines(salci, base_k9)
            pitcher_k_pct = (p_baseline or p_recent or {}).get("K_percent", 0.22)
            opp_lineup_info = game_lineups[opp_side]

            all_pitcher_results.append({
                "pitcher": pitcher,
                "pitcher_id": pid,
                "pitcher_hand": pitcher_hand,
                "pitcher_k_pct": pitcher_k_pct,
                "team": team,
                "opponent": opp,
                "opponent_id": opp_id,
                "game_pk": game_pk,
                "salci": salci,
                "expected": proj["expected"],
                "lines": proj["lines"],
                "breakdown": breakdown,
                "lineup_confirmed": opp_lineup_info["confirmed"],
                # v2 additions
                "stuff": round(stuff, 2),
                "location": round(location, 2),
                "matchup": round(matchup, 2),
                "explanation": explanation,
                "workload": workload
            })

        if show_hitters:
            opp_lineup_info = game_lineups[opp_side]
            if opp_lineup_info["confirmed"] or not confirmed_only:
                for player in opp_lineup_info["lineup"]:
                    h_recent = get_hitter_recent_stats(player["id"], 7)
                    if h_recent:
                        h_score = compute_hitter_score(h_recent)
                        if not hot_hitters_only or h_score >= 60:
                            all_hitter_results.append({
                                "name": player["name"],
                                "player_id": player["id"],
                                "position": player["position"],
                                "batting_order": player["batting_order"],
                                "bat_side": player["bat_side"],
                                "team": opp,
                                "vs_pitcher": pitcher,
                                "pitcher_hand": pitcher_hand,
                                "pitcher_k_pct": (p_baseline or p_recent or {}).get("K_percent", 0.22),
                                "game_pk": game_pk,
                                "recent": h_recent,
                                "score": h_score,
                                "lineup_confirmed": opp_lineup_info["confirmed"]
                            })

progress.empty()

all_pitcher_results.sort(key=lambda x: x["salci"], reverse=True)
all_hitter_results.sort(key=lambda x: x["score"], reverse=True)

# ----------------------------
# Tab 1: Pitchers (original + v2 insights added)
# ----------------------------
with tab1:
    st.markdown("### 🎯 Pitcher Strikeout Predictions")
    filtered_pitchers = [p for p in all_pitcher_results if p["salci"] >= min_salci]

    if not filtered_pitchers:
        st.info("No pitchers match your filters.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pitchers", len(filtered_pitchers))
        with col2:
            elite = len([p for p in filtered_pitchers if p["salci"] >= 75])
            st.metric("🔥 Elite", elite)
        with col3:
            strong = len([p for p in filtered_pitchers if 60 <= p["salci"] < 75])
            st.metric("✅ Strong", strong)
        with col4:
            confirmed = len([p for p in filtered_pitchers if p.get("lineup_confirmed")])
            st.metric("📋 Lineups Confirmed", confirmed)

        st.markdown("---")
        for result in filtered_pitchers:
            if result.get("lineup_confirmed"):
                st.markdown("✓ Opponent Lineup Confirmed", unsafe_allow_html=True)
            else:
                st.markdown("⏳ Lineup Pending", unsafe_allow_html=True)
            render_pitcher_card(result)
            render_v2_insights(result)   # ← NEW v2 decision layer
            st.markdown("---")

# ----------------------------
# Tab 2: Hitters (unchanged)
# ----------------------------
with tab2:
    st.markdown("### 🏏 Hitter Analysis & Matchups")
    if confirmed_only:
        st.info("📋 Showing **CONFIRMED STARTERS ONLY** - These players are in today's starting lineup!")

    if not all_hitter_results:
        if confirmed_only:
            st.warning("No confirmed lineups available yet. Lineups are typically released 1-2 hours before game time.")
        else:
            st.info("Enable 'Show Hitter Analysis' in sidebar to see hitter data.")
    else:
        hot_hitters = [h for h in all_hitter_results if h["score"] >= 70]
        cold_hitters = [h for h in all_hitter_results if h["score"] <= 30]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔥 Hottest Hitters (Starting Today)")
            if hot_hitters:
                for h in hot_hitters[:8]:
                    render_hitter_card(h, show_batting_order=True)
            else:
                st.info("No hot hitters in confirmed lineups yet.")

        with col2:
            st.markdown("#### ❄️ Coldest Hitters (Fade Candidates)")
            if cold_hitters:
                for h in cold_hitters[:8]:
                    render_hitter_card(h, show_batting_order=True)
            else:
                st.info("No cold hitters in confirmed lineups yet.")

        st.markdown("---")
        st.markdown("#### 📊 All Confirmed Starters")
        if all_hitter_results:
            df_hitters = pd.DataFrame([{
                "Order": f"#{h['batting_order']}" if h.get("batting_order") else "-",
                "Player": h["name"],
                "Bats": h.get("bat_side", "R"),
                "Team": h["team"],
                "Pos": h["position"],
                "vs Pitcher": h["vs_pitcher"],
                "P Hand": h.get("pitcher_hand", "R"),
                "Score": round(h["score"], 1),
                "AVG (L7)": f"{h['recent'].get('avg', 0):.3f}",
                "OPS (L7)": f"{h['recent'].get('ops', 0):.3f}",
                "K% (L7)": f"{h['recent'].get('k_rate', 0)*100:.1f}%",
                "HR (L7)": h["recent"].get("hr", 0),
                "Hit Streak": h["recent"].get("hit_streak", 0),
                "Confirmed": "✅" if h.get("lineup_confirmed") else "⏳"
            } for h in all_hitter_results])
            st.dataframe(df_hitters, use_container_width=True, hide_index=True)

# ----------------------------
# Tab 3: Best Bets + Yesterday Reflection
# ----------------------------
with tab3:
    st.markdown("### 🎯 Today's Best Bets")
    confirmed_hitters = [h for h in all_hitter_results if h.get("lineup_confirmed")]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⚾ Top Pitcher K Props")
        top_pitchers = [p for p in all_pitcher_results if p["salci"] >= 60][:5]
        if not top_pitchers:
            st.info("No elite pitcher picks available.")
        else:
            for i, p in enumerate(top_pitchers, 1):
                rating_label, emoji, _ = get_rating(p["salci"])
                lineup_badge = "✅" if p.get("lineup_confirmed") else "⏳"
                p_hand = p.get("pitcher_hand", "R")
                st.markdown(f"""
                **{i}. {emoji} {p['pitcher']} ({p_hand}) vs {p['opponent']}** {lineup_badge}
                - SALCI: **{p['salci']}** ({rating_label})
                - Expected Ks: **{p['expected']}**
                """)

    with col2:
        st.markdown("#### 🏏 Top Hitter Spots")
        if not confirmed_hitters:
            st.info("No confirmed hitters available.")
        else:
            for i, h in enumerate(confirmed_hitters[:5], 1):
                rating_label, css_class = get_hitter_rating(h["score"])
                st.markdown(f"""
                **{i}. {h['name']} ({h.get('bat_side', 'R')})**
                - Score: **{h['score']:.1f}** ({rating_label})
                - AVG (L7): **{h['recent'].get('avg', 0):.3f}**
                - OPS (L7): **{h['recent'].get('ops', 0):.3f}**
                """)

    # Yesterday Reflection (global feedback loop)
    st.markdown("---")
    st.markdown("### 📊 Yesterday’s Reflection (Model Self-Learning)")
    # Demo historical results (in production this would be loaded from a DB of past predictions)
    yesterday_results_df = pd.DataFrame({
        "IP": [5.0, 6.0, 4.0, 7.0, 5.2, 6.1, 5.1],
        "Ks": [6, 8, 3, 9, 7, 5, 4],
        "proj_K": [5.5, 7.2, 4.8, 6.5, 6.8, 4.9, 5.3]
    })
    perf = evaluate_model_performance(yesterday_results_df)
    st.markdown(f"""
    - Avg IP: **{perf['avg_ip']:.2f}**
    - K Diff (actual - projected): **{perf['avg_k_diff']:+.2f}**
    - Overperformers: **{perf['overperformers']}**
    - Underperformers: **{perf['underperformers']}**
    """)
    st.caption("This feedback loop will be connected to a real database in future updates.")

# ----------------------------
# Tab 4: Charts (original + NEW heatmaps)
# ----------------------------
with tab4:
    st.markdown("### 📊 Charts & Shareable Graphics")

    if all_pitcher_results:
        st.plotly_chart(create_pitcher_comparison_chart(all_pitcher_results), use_container_width=True)
    if all_hitter_results:
        st.plotly_chart(create_hitter_hotness_chart(all_hitter_results), use_container_width=True)
        st.plotly_chart(create_matchup_scatter(all_hitter_results), use_container_width=True)

    st.plotly_chart(create_salci_breakdown_chart(), use_container_width=True)

    # NEW Heat Map System
    st.markdown("### 🔥 Pitch & Hitter Zone Heatmaps")
    st.caption("Demo using synthetic data (real Statcast integration planned next)")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_zone_heatmap(pd.DataFrame(), "Demo: Pitcher Pitch Zones"), use_container_width=True)
    with col2:
        st.plotly_chart(create_zone_heatmap(pd.DataFrame(), "Demo: Hitter Hot Zones"), use_container_width=True)

st.markdown("---")
st.caption("SALCI v4.0 • Original scoring preserved • New decision layers added modularly • Built with ❤️ and Python")
</code></pre>
</body>
</html>
