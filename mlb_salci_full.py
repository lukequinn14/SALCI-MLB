#!/usr/bin/env python3
"""
SALCI v3.3 - Full Pitcher + Hitter Analysis with Charts
MLB Prediction Model with CONFIRMED LINEUPS + SHAREABLE GRAPHICS

Run with:
    streamlit run mlb_salci_full.py

NOTE: Lineups are typically released 1-2 hours before game time.
Run this closer to game time for best results!
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
    page_title="SALCI v3.3 - MLB Predictions",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
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
    .elite { color: #10b981; font-weight: bold; }
    .strong { color: #22c55e; font-weight: bold; }
    .average { color: #eab308; font-weight: bold; }
    .below { color: #f97316; font-weight: bold; }
    .poor { color: #ef4444; font-weight: bold; }
    
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
    .matchup-good { background-color: #d4edda; }
    .matchup-neutral { background-color: #fff3cd; }
    .matchup-bad { background-color: #f8d7da; }
    
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
    
    div[data-testid="stHorizontalBlock"] > div {
        padding: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Configuration
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

# Chart color scheme
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
# Helper Functions
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
# API Functions - Teams & Schedule
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
# API Functions - Pitchers
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


# ----------------------------
# API Functions - Hitters
# ----------------------------
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
        slg = (totals["hits"] + totals["doubles"] + 2*totals["triples"] + 3*totals["hr"]) / totals["ab"]
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
# SALCI Computation
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
# Chart Functions
# ----------------------------
def create_pitcher_comparison_chart(pitcher_results: List[Dict]) -> go.Figure:
    """Create horizontal bar chart of top pitchers by SALCI score."""
    if not pitcher_results:
        return None
    
    # Take top 10 pitchers
    top_pitchers = sorted(pitcher_results, key=lambda x: x["salci"], reverse=True)[:10]
    top_pitchers = top_pitchers[::-1]  # Reverse for horizontal bar
    
    # Include handedness in name
    names = [f"{p['pitcher'].split()[-1]} ({p.get('pitcher_hand', 'R')})" for p in top_pitchers]
    scores = [p["salci"] for p in top_pitchers]
    colors = [get_salci_color(s) for s in scores]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=names,
        x=scores,
        orientation='h',
        marker_color=colors,
        text=[f"{s}" for s in scores],
        textposition='outside',
        textfont=dict(size=12, color='#333')
    ))
    
    # Add elite threshold line
    fig.add_vline(x=75, line_dash="dash", line_color="#10b981", line_width=2,
                  annotation_text="Elite (75+)", annotation_position="top")
    
    # Add strong threshold line
    fig.add_vline(x=60, line_dash="dot", line_color="#3b82f6", line_width=1,
                  annotation_text="Strong (60+)", annotation_position="bottom")
    
    fig.update_layout(
        title=dict(text="Today's Top SALCI Pitchers", font=dict(size=18)),
        xaxis_title="SALCI Score",
        yaxis_title="",
        xaxis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100]),
        height=400,
        margin=dict(l=100, r=50, t=80, b=60),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text="#SALCI",
                xref="paper", yref="paper",
                x=0, y=-0.15,
                showarrow=False,
                font=dict(size=11, color="#999")
            )
        ]
    )
    
    return fig


def create_hitter_hotness_chart(hitter_results: List[Dict]) -> go.Figure:
    """Create grouped bar chart of hot hitters with AVG and OPS."""
    if not hitter_results:
        return None
    
    # Take top 8 hitters by score
    top_hitters = sorted(hitter_results, key=lambda x: x["score"], reverse=True)[:8]
    
    # Include handedness in name
    names = [f"{h['name'].split()[-1]} ({h.get('bat_side', 'R')})" for h in top_hitters]
    avgs = [h["recent"].get("avg", 0) for h in top_hitters]
    ops_vals = [h["recent"].get("ops", 0) for h in top_hitters]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='AVG (L7)',
        x=names,
        y=avgs,
        marker_color=COLORS["hot"],
        text=[f".{int(a*1000):03d}" for a in avgs],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='OPS (L7)',
        x=names,
        y=ops_vals,
        marker_color=COLORS["secondary"],
        text=[f"{o:.3f}" for o in ops_vals],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text="Hottest Hitters (Last 7 Games)", font=dict(size=18)),
        yaxis_title="",
        barmode='group',
        height=350,
        margin=dict(l=50, r=50, t=80, b=80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[
            dict(
                text="#SALCI",
                xref="paper", yref="paper",
                x=0, y=-0.22,
                showarrow=False,
                font=dict(size=11, color="#999")
            )
        ]
    )
    
    return fig


def create_salci_breakdown_chart() -> go.Figure:
    """Create donut chart showing SALCI metric weights."""
    labels = ['K/9', 'K%', 'Opp K%', 'Opp Contact%', 'K/BB', 'P/IP']
    values = [18, 18, 22, 18, 14, 10]
    colors_list = ['#3266ad', '#7F77DD', '#1D9E75', '#D85A30', '#eab308', '#888780']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors_list,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=11)
    )])
    
    fig.update_layout(
        title=dict(text="SALCI Score Breakdown (Balanced Weights)", font=dict(size=16)),
        height=350,
        margin=dict(l=20, r=20, t=60, b=60),
        showlegend=False,
        annotations=[
            dict(
                text="SALCI<br>Score",
                x=0.5, y=0.5,
                font=dict(size=14, color="#333"),
                showarrow=False
            ),
            dict(
                text="#SALCI",
                xref="paper", yref="paper",
                x=0, y=-0.1,
                showarrow=False,
                font=dict(size=11, color="#999")
            )
        ]
    )
    
    return fig


def create_matchup_scatter(hitter_results: List[Dict]) -> go.Figure:
    """Create scatter plot of hitters: K% vs AVG with matchup coloring."""
    if not hitter_results or len(hitter_results) < 3:
        return None
    
    k_rates = [h["recent"].get("k_rate", 0.22) * 100 for h in hitter_results]
    avgs = [h["recent"].get("avg", 0.250) for h in hitter_results]
    # Include handedness in hover/display
    names = [f"{h['name']} ({h.get('bat_side', 'R')})" for h in hitter_results]
    short_names = [f"{h['name'].split()[-1]} ({h.get('bat_side', 'R')})" for h in hitter_results]
    scores = [h["score"] for h in hitter_results]
    
    # Color by score
    colors = [get_salci_color(s) if s >= 50 else COLORS["cold"] for s in scores]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_rates,
        y=avgs,
        mode='markers+text',
        marker=dict(size=12, color=colors, line=dict(width=1, color='white')),
        text=short_names,
        textposition='top center',
        textfont=dict(size=8),
        hovertemplate='<b>%{text}</b><br>K%: %{x:.1f}%<br>AVG: %{y:.3f}<extra></extra>'
    ))
    
    # Add quadrant lines
    fig.add_hline(y=0.270, line_dash="dash", line_color="#ccc", line_width=1)
    fig.add_vline(x=22, line_dash="dash", line_color="#ccc", line_width=1)
    
    # Add quadrant labels
    fig.add_annotation(x=15, y=0.35, text="🔥 Low K%, High AVG", showarrow=False, font=dict(size=10, color="#10b981"))
    fig.add_annotation(x=30, y=0.20, text="❄️ High K%, Low AVG", showarrow=False, font=dict(size=10, color="#ef4444"))
    
    fig.update_layout(
        title=dict(text="Hitter Profile: K% vs AVG (L7)", font=dict(size=16)),
        xaxis_title="K% (Lower is better)",
        yaxis_title="AVG (Higher is better)",
        height=400,
        margin=dict(l=60, r=40, t=80, b=80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text="#SALCI",
                xref="paper", yref="paper",
                x=0, y=-0.18,
                showarrow=False,
                font=dict(size=11, color="#999")
            )
        ]
    )
    
    return fig


def create_k_projection_chart(pitcher_results: List[Dict]) -> go.Figure:
    """Create bar chart showing K line projections for top pitchers."""
    if not pitcher_results:
        return None
    
    # Take top 5 pitchers
    top_pitchers = sorted(pitcher_results, key=lambda x: x["salci"], reverse=True)[:5]
    
    # Include handedness in name
    names = [f"{p['pitcher'].split()[-1]} ({p.get('pitcher_hand', 'R')})" for p in top_pitchers]
    expected_ks = [p["expected"] for p in top_pitchers]
    k5_probs = [p["lines"].get(5, 50) for p in top_pitchers]
    k6_probs = [p["lines"].get(6, 40) for p in top_pitchers]
    k7_probs = [p["lines"].get(7, 30) for p in top_pitchers]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='5+ Ks',
        x=names,
        y=k5_probs,
        marker_color='#10b981',
        text=[f"{p}%" for p in k5_probs],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='6+ Ks',
        x=names,
        y=k6_probs,
        marker_color='#3b82f6',
        text=[f"{p}%" for p in k6_probs],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='7+ Ks',
        x=names,
        y=k7_probs,
        marker_color='#7F77DD',
        text=[f"{p}%" for p in k7_probs],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text="K Line Probabilities (Top Pitchers)", font=dict(size=16)),
        yaxis_title="Probability %",
        yaxis=dict(range=[0, 100]),
        barmode='group',
        height=350,
        margin=dict(l=50, r=50, t=80, b=80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[
            dict(
                text="#SALCI",
                xref="paper", yref="paper",
                x=0, y=-0.22,
                showarrow=False,
                font=dict(size=11, color="#999")
            )
        ]
    )
    
    return fig


def create_game_day_card(
    selected_date: datetime,
    num_games: int,
    top_pitcher: Dict,
    top_hitter: Dict,
    elite_count: int,
    strong_count: int,
    hot_hitters_count: int,
    lineups_confirmed: bool
) -> None:
    """Create the shareable Game Day Summary Card."""
    
    date_str = selected_date.strftime("%A, %b %d")
    lineup_status = "✓" if lineups_confirmed else "⏳"
    lineup_text = "Lineups In" if lineups_confirmed else "Pending"
    
    # Top pitcher stats
    if top_pitcher:
        p_name = top_pitcher.get("pitcher", "TBD")
        p_hand = top_pitcher.get("pitcher_hand", "R")
        p_team = top_pitcher.get("team", "").split()[-1] if top_pitcher.get("team") else ""
        p_opp = top_pitcher.get("opponent", "").split()[-1] if top_pitcher.get("opponent") else ""
        p_salci = top_pitcher.get("salci", 0)
        p_5k = top_pitcher.get("lines", {}).get(5, 0)
        p_6k = top_pitcher.get("lines", {}).get(6, 0)
        p_7k = top_pitcher.get("lines", {}).get(7, 0)
    else:
        p_name, p_hand, p_team, p_opp, p_salci, p_5k, p_6k, p_7k = "TBD", "R", "", "", 0, 0, 0, 0
    
    # Top hitter stats
    if top_hitter:
        h_name = top_hitter.get("name", "TBD")
        h_hand = top_hitter.get("bat_side", "R")
        h_team = top_hitter.get("team", "").split()[-1] if top_hitter.get("team") else ""
        h_pos = top_hitter.get("position", "")
        h_recent = top_hitter.get("recent", {})
        h_avg = h_recent.get("avg", 0)
        h_ops = h_recent.get("ops", 0)
        h_krate = h_recent.get("k_rate", 0) * 100
        h_streak = h_recent.get("hit_streak", 0)
        h_vs_pitcher = top_hitter.get("vs_pitcher", "")
        h_vs_hand = top_hitter.get("pitcher_hand", "R")
    else:
        h_name, h_hand, h_team, h_pos, h_avg, h_ops, h_krate, h_streak = "TBD", "R", "", "", 0, 0, 0, 0
        h_vs_pitcher, h_vs_hand = "", "R"
    
    st.markdown(f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; max-width: 520px; margin: 0 auto;">
      <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2e5a8f 100%); border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
        
        <!-- Header -->
        <div style="padding: 20px 24px; border-bottom: 1px solid rgba(255,255,255,0.1);">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <div style="font-size: 12px; color: rgba(255,255,255,0.7); letter-spacing: 1px; margin-bottom: 4px;">SALCI GAME DAY</div>
              <div style="font-size: 24px; font-weight: 700; color: white;">{date_str}</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 32px;">⚾</div>
              <div style="font-size: 11px; color: rgba(255,255,255,0.7);">{num_games} Games</div>
            </div>
          </div>
        </div>
        
        <!-- Top Pitcher Section -->
        <div style="padding: 20px 24px; background: rgba(255,255,255,0.05);">
          <div style="font-size: 11px; color: #10b981; letter-spacing: 1px; margin-bottom: 12px; font-weight: 600;">🎯 TOP K PLAY</div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <div style="font-size: 22px; font-weight: 700; color: white;">{p_name} <span style="font-size: 14px; color: rgba(255,255,255,0.6); font-weight: 500;">({p_hand}HP)</span></div>
              <div style="font-size: 14px; color: rgba(255,255,255,0.7);">{p_team} vs {p_opp}</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 36px; font-weight: 800; color: #10b981;">{p_salci}</div>
              <div style="font-size: 11px; color: rgba(255,255,255,0.6);">SALCI</div>
            </div>
          </div>
          <div style="display: flex; gap: 8px; margin-top: 16px;">
            <div style="flex: 1; background: rgba(16,185,129,0.2); border-radius: 8px; padding: 10px; text-align: center;">
              <div style="font-size: 20px; font-weight: 700; color: #10b981;">{p_5k}%</div>
              <div style="font-size: 10px; color: rgba(255,255,255,0.6);">5+ Ks</div>
            </div>
            <div style="flex: 1; background: rgba(16,185,129,0.2); border-radius: 8px; padding: 10px; text-align: center;">
              <div style="font-size: 20px; font-weight: 700; color: #10b981;">{p_6k}%</div>
              <div style="font-size: 10px; color: rgba(255,255,255,0.6);">6+ Ks</div>
            </div>
            <div style="flex: 1; background: rgba(16,185,129,0.2); border-radius: 8px; padding: 10px; text-align: center;">
              <div style="font-size: 20px; font-weight: 700; color: #10b981;">{p_7k}%</div>
              <div style="font-size: 10px; color: rgba(255,255,255,0.6);">7+ Ks</div>
            </div>
          </div>
        </div>
        
        <!-- Hot Hitter Section -->
        <div style="padding: 20px 24px; border-top: 1px solid rgba(255,255,255,0.1);">
          <div style="font-size: 11px; color: #f59e0b; letter-spacing: 1px; margin-bottom: 12px; font-weight: 600;">🔥 HOTTEST BAT</div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <div style="font-size: 22px; font-weight: 700; color: white;">{h_name} <span style="font-size: 14px; color: rgba(255,255,255,0.6); font-weight: 500;">({h_hand}HB)</span></div>
              <div style="font-size: 14px; color: rgba(255,255,255,0.7);">{h_team} • {h_pos} • vs {h_vs_hand}HP</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 11px; color: rgba(255,255,255,0.5);">L7 AVG</div>
              <div style="font-size: 28px; font-weight: 800; color: #f59e0b;">{h_avg:.3f}</div>
            </div>
          </div>
          <div style="display: flex; gap: 8px; margin-top: 16px;">
            <div style="flex: 1; background: rgba(245,158,11,0.15); border-radius: 8px; padding: 10px; text-align: center;">
              <div style="font-size: 16px; font-weight: 700; color: #f59e0b;">{h_ops:.3f}</div>
              <div style="font-size: 10px; color: rgba(255,255,255,0.6);">OPS</div>
            </div>
            <div style="flex: 1; background: rgba(245,158,11,0.15); border-radius: 8px; padding: 10px; text-align: center;">
              <div style="font-size: 16px; font-weight: 700; color: {'#10b981' if h_krate < 20 else '#f59e0b'};">{h_krate:.1f}%</div>
              <div style="font-size: 10px; color: rgba(255,255,255,0.6);">K%</div>
            </div>
            <div style="flex: 1; background: rgba(245,158,11,0.15); border-radius: 8px; padding: 10px; text-align: center;">
              <div style="font-size: 16px; font-weight: 700; color: white;">{h_streak}</div>
              <div style="font-size: 10px; color: rgba(255,255,255,0.6);">Hit Streak</div>
            </div>
          </div>
        </div>
        
        <!-- Quick Stats Row -->
        <div style="padding: 16px 24px; background: rgba(0,0,0,0.2); display: flex; justify-content: space-around;">
          <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: 700; color: #10b981;">{elite_count}</div>
            <div style="font-size: 10px; color: rgba(255,255,255,0.6);">Elite</div>
          </div>
          <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: 700; color: #3b82f6;">{strong_count}</div>
            <div style="font-size: 10px; color: rgba(255,255,255,0.6);">Strong</div>
          </div>
          <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: 700; color: #f59e0b;">{hot_hitters_count}</div>
            <div style="font-size: 10px; color: rgba(255,255,255,0.6);">Hot Bats</div>
          </div>
          <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: 700; color: white;">{lineup_status}</div>
            <div style="font-size: 10px; color: rgba(255,255,255,0.6);">{lineup_text}</div>
          </div>
        </div>
        
        <!-- Footer -->
        <div style="padding: 12px 24px; background: rgba(0,0,0,0.3); display: flex; justify-content: space-between; align-items: center;">
          <div style="font-size: 11px; color: rgba(255,255,255,0.5);">Not financial advice • For entertainment only</div>
          <div style="font-size: 13px; font-weight: 700; color: rgba(255,255,255,0.8);">#SALCI</div>
        </div>
        
      </div>
    </div>
    """, unsafe_allow_html=True)


def create_daily_picks_card(top_pitcher: Dict, top_hitter: Dict) -> None:
    """Create the daily picks summary card (legacy version)."""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3a5f, #2e5a8f); 
                border-radius: 12px; padding: 1.5rem; color: white; margin-bottom: 1rem;'>
        <div style='font-size: 0.8rem; opacity: 0.8; margin-bottom: 0.5rem;'>TODAY'S TOP SALCI PLAYS</div>
        <div style='font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem;'>⚾ Daily Picks Card</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if top_pitcher:
            rating_label, emoji, _ = get_rating(top_pitcher["salci"])
            st.markdown(f"""
            <div style='background: white; border-radius: 10px; padding: 1rem; 
                        border-left: 4px solid #10b981; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <div style='font-size: 0.7rem; color: #666; margin-bottom: 0.3rem;'>🎯 TOP PITCHER PLAY</div>
                <div style='font-size: 1.3rem; font-weight: bold;'>{top_pitcher['pitcher']}</div>
                <div style='font-size: 0.9rem; color: #666;'>{top_pitcher['team']} vs {top_pitcher['opponent']}</div>
                <div style='margin-top: 0.8rem; display: flex; gap: 1rem;'>
                    <div>
                        <div style='font-size: 2rem; font-weight: bold; color: #10b981;'>{top_pitcher['salci']}</div>
                        <div style='font-size: 0.7rem; color: #666;'>SALCI</div>
                    </div>
                    <div>
                        <div style='font-size: 1.5rem; font-weight: bold;'>{top_pitcher['expected']}</div>
                        <div style='font-size: 0.7rem; color: #666;'>Expected Ks</div>
                    </div>
                </div>
                <div style='margin-top: 0.8rem; font-size: 0.85rem;'>
                    <span style='background: #d4edda; padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.5rem;'>
                        6+ @ {top_pitcher['lines'][6]}%
                    </span>
                    <span style='background: #e8f4fd; padding: 0.2rem 0.5rem; border-radius: 4px;'>
                        7+ @ {top_pitcher['lines'][7]}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No pitcher data available")
    
    with col2:
        if top_hitter:
            r = top_hitter["recent"]
            matchup, _ = get_matchup_grade(r.get("k_rate", 0.22), top_hitter.get("pitcher_k_pct", 0.22),
                                           top_hitter.get("bat_side", "R"), top_hitter.get("pitcher_hand", "R"))
            streak_text = f"🔥 {r.get('hit_streak', 0)}-game hit streak" if r.get('hit_streak', 0) >= 3 else ""
            
            st.markdown(f"""
            <div style='background: white; border-radius: 10px; padding: 1rem; 
                        border-left: 4px solid #D85A30; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <div style='font-size: 0.7rem; color: #666; margin-bottom: 0.3rem;'>🔥 HOT HITTER PLAY</div>
                <div style='font-size: 1.3rem; font-weight: bold;'>{top_hitter['name']}</div>
                <div style='font-size: 0.9rem; color: #666;'>{top_hitter['team']} • Batting #{top_hitter.get('batting_order', '?')}</div>
                <div style='margin-top: 0.8rem; display: flex; gap: 1rem;'>
                    <div>
                        <div style='font-size: 1.5rem; font-weight: bold; color: #10b981;'>.{int(r.get('avg', 0)*1000):03d}</div>
                        <div style='font-size: 0.7rem; color: #666;'>AVG (L7)</div>
                    </div>
                    <div>
                        <div style='font-size: 1.5rem; font-weight: bold;'>{r.get('ops', 0):.3f}</div>
                        <div style='font-size: 0.7rem; color: #666;'>OPS (L7)</div>
                    </div>
                </div>
                <div style='margin-top: 0.8rem; font-size: 0.85rem;'>
                    <span style='background: #d4edda; padding: 0.2rem 0.5rem; border-radius: 4px;'>{matchup}</span>
                    {f"<span style='margin-left: 0.5rem;'>{streak_text}</span>" if streak_text else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No confirmed hitter data available")
    
    st.markdown("""
    <div style='text-align: center; margin-top: 1rem; padding: 0.8rem; 
                background: #f8f9fa; border-radius: 8px; font-size: 0.8rem; color: #666;'>
        ⚠️ <strong>Disclaimer:</strong> SALCI is for entertainment purposes only. Baseball is unpredictable. Bet responsibly.
    </div>
    <div style='text-align: right; margin-top: 0.5rem; font-size: 0.7rem; color: #999;'>#SALCI</div>
    """, unsafe_allow_html=True)


# ----------------------------
# UI Components
# ----------------------------
def render_pitcher_card(result: Dict):
    rating_label, emoji, css_class = get_rating(result["salci"])
    
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"### {result['pitcher']}")
            st.markdown(f"**{result['team']}** vs {result['opponent']}")
        
        with col2:
            st.markdown(f"<div style='text-align: center;'>"
                       f"<span style='font-size: 2.5rem; font-weight: bold;'>{result['salci']}</span><br>"
                       f"<span class='{css_class}'>{emoji} {rating_label}</span></div>",
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"**Expected Ks:** {result['expected']}")
            lines = result['lines']
            cols = st.columns(4)
            for i, (k, prob) in enumerate(list(lines.items())[1:5]):
                with cols[i]:
                    color = "#22c55e" if prob >= 65 else "#eab308" if prob >= 45 else "#ef4444"
                    st.markdown(f"<div style='text-align:center;'><small>{k}+</small><br>"
                               f"<span style='color:{color}; font-weight:bold;'>{prob}%</span></div>",
                               unsafe_allow_html=True)
        
        st.progress(min(result["salci"] / 100, 1.0))
        st.markdown("---")


def render_hitter_card(hitter: Dict, show_batting_order: bool = True):
    score = hitter.get("score", 50)
    rating, css = get_hitter_rating(score)
    recent = hitter.get("recent", {})
    
    matchup_grade, matchup_css = get_matchup_grade(
        recent.get("k_rate", 0.22),
        hitter.get("pitcher_k_pct", 0.22),
        hitter.get("bat_side", "R"),
        hitter.get("pitcher_hand", "R")
    )
    
    col1, col2, col3, col4 = st.columns([2.5, 1, 1, 1])
    
    with col1:
        order_badge = ""
        if show_batting_order and hitter.get("batting_order"):
            order_badge = f"<span class='batting-order'>#{hitter['batting_order']}</span> "
        
        st.markdown(f"{order_badge}**{hitter['name']}** ({hitter.get('position', '')})", 
                   unsafe_allow_html=True)
        
        if recent.get("hit_streak", 0) >= 3:
            st.markdown(f"<span class='hot-streak'>🔥 {recent['hit_streak']}-game hit streak</span>", 
                       unsafe_allow_html=True)
        elif recent.get("hitless_streak", 0) >= 3:
            st.markdown(f"<span class='cold-streak'>❄️ {recent['hitless_streak']}-game hitless</span>",
                       unsafe_allow_html=True)
    
    with col2:
        if recent:
            st.metric("AVG (L7)", f"{recent.get('avg', 0):.3f}")
    
    with col3:
        if recent:
            st.metric("OPS (L7)", f"{recent.get('ops', 0):.3f}")
    
    with col4:
        st.markdown(f"<div class='{matchup_css}' style='padding: 0.5rem; border-radius: 5px; text-align: center;'>"
                   f"{matchup_grade}</div>", unsafe_allow_html=True)


# ----------------------------
# Main App
# ----------------------------
def main():
    # Header
    st.markdown("<h1 class='main-header'>⚾ SALCI v3.3</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Pitcher Strikeouts + Hitter Analysis + Shareable Charts</p>", 
               unsafe_allow_html=True)
    
    # Sidebar
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
        
        # Refresh button
        if st.button("🔄 Refresh Lineups", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.caption("💡 Lineups are usually posted 1-2 hours before game time. Click refresh to get latest!")
        
        st.markdown("---")
        
        with st.expander("📊 About SALCI"):
            st.markdown("""
            **SALCI** = Strikeout Adjusted Lineup Confidence Index
            
            **Pitcher Metrics:**
            - K/9: Strikeouts per 9 innings
            - K%: Strikeout rate
            - K/BB: Strikeout to walk ratio
            - P/IP: Pitches per inning
            
            **Matchup Factors:**
            - Opp K%: Opponent team strikeout rate
            - Opp Contact%: Opponent contact rate
            
            **Hitter Grades:**
            - 🟢 Favorable: Platoon advantage + low K%
            - 🟡 Neutral: Mixed factors
            - 🔴 Tough: Same-hand + high K%
            """)
    
    # Main content
    date_str = selected_date.strftime("%Y-%m-%d")
    weights = WEIGHT_PRESETS[preset_key]["weights"]
    
    # Tabs - Added Charts tab!
    tab1, tab2, tab3, tab4 = st.tabs(["⚾ Pitcher Analysis", "🏏 Hitter Matchups", "🎯 Best Bets", "📊 Charts & Share"])
    
    with st.spinner("🔍 Fetching games and lineups..."):
        games = get_games_by_date(date_str)
    
    if not games:
        st.warning(f"No games found for {date_str}")
        return
    
    st.success(f"Found **{len(games)} games** for {selected_date.strftime('%A, %B %d, %Y')}")
    
    # Check lineup status for all games
    lineup_status = {}
    for game in games:
        game_pk = game["game_pk"]
        home_lineup, home_confirmed = get_confirmed_lineup(game_pk, "home")
        away_lineup, away_confirmed = get_confirmed_lineup(game_pk, "away")
        lineup_status[game_pk] = {
            "home": {"lineup": home_lineup, "confirmed": home_confirmed},
            "away": {"lineup": away_lineup, "confirmed": away_confirmed}
        }
    
    # Count confirmed lineups
    confirmed_count = sum(1 for g in games 
                         if lineup_status[g["game_pk"]]["home"]["confirmed"] 
                         or lineup_status[g["game_pk"]]["away"]["confirmed"])
    
    if confirmed_count == 0:
        st.warning("⏳ **No lineups confirmed yet.** Lineups are typically released 1-2 hours before game time. Click 'Refresh Lineups' to check for updates.")
    else:
        st.info(f"✅ **{confirmed_count} games** have confirmed lineups")
    
    # Process all data
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
            
            # Pitcher stats
            p_recent = get_recent_pitcher_stats(pid, 7)
            p_baseline = parse_season_stats(get_player_season_stats(pid, 2025))
            opp_recent = get_team_batting_stats(opp_id, 14)
            opp_baseline = get_team_season_batting(opp_id, 2025)
            
            games_played = p_recent.get("games_sampled", 0) if p_recent else 0
            
            salci, breakdown, missing = compute_salci(
                p_recent, p_baseline, opp_recent, opp_baseline, weights, games_played
            )
            
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
                    "lineup_confirmed": opp_lineup_info["confirmed"]
                })
            
            # Hitter stats - ONLY from confirmed lineups
            if show_hitters:
                opp_lineup_info = game_lineups[opp_side]
                
                if opp_lineup_info["confirmed"] or not confirmed_only:
                    lineup = opp_lineup_info["lineup"]
                    
                    for player in lineup:
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
    
    # Sort results
    all_pitcher_results.sort(key=lambda x: x["salci"], reverse=True)
    all_hitter_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Tab 1: Pitcher Analysis
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
                    st.markdown(f"<span class='lineup-confirmed'>✓ Opponent Lineup Confirmed</span>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='lineup-pending'>⏳ Lineup Pending</span>", 
                               unsafe_allow_html=True)
                render_pitcher_card(result)
    
    # Tab 2: Hitter Matchups
    with tab2:
        st.markdown("### 🏏 Hitter Analysis & Matchups")
        
        if confirmed_only:
            st.info("📋 Showing **CONFIRMED STARTERS ONLY** - These players are in today's starting lineup!")
        
        if not all_hitter_results:
            if confirmed_only:
                st.warning("No confirmed lineups available yet. Lineups are typically released 1-2 hours before game time. Uncheck 'Confirmed Lineups Only' to see projected starters, or click 'Refresh Lineups'.")
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
                        st.markdown("")
                else:
                    st.info("No hot hitters in confirmed lineups yet.")
            
            with col2:
                st.markdown("#### ❄️ Coldest Hitters (Fade Candidates)")
                if cold_hitters:
                    for h in cold_hitters[:8]:
                        render_hitter_card(h, show_batting_order=True)
                        st.markdown("")
                else:
                    st.info("No cold hitters in confirmed lineups yet.")
            
            st.markdown("---")
            
            st.markdown("#### 📊 All Confirmed Starters")
            
            if all_hitter_results:
                df_hitters = pd.DataFrame([{
                    "Order": f"#{h['batting_order']}" if h.get('batting_order') else "-",
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
    
    # Tab 3: Best Bets
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
                    <div style='background: #f0f9ff; padding: 1rem; border-radius: 10px; 
                                margin-bottom: 0.5rem; border-left: 4px solid #3b82f6;'>
                        <strong>#{i} {p['pitcher']} ({p_hand}HP)</strong> ({p['team']} vs {p['opponent']}) {lineup_badge}<br>
                        <span style='font-size: 1.2rem;'>{emoji} SALCI: {p['salci']}</span><br>
                        Expected: <strong>{p['expected']} Ks</strong><br>
                        5+ @ {p['lines'][5]}% | 6+ @ {p['lines'][6]}% | 7+ @ {p['lines'][7]}%
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🏏 Hot Hitter Props (Confirmed Starters)")
            top_hitters = [h for h in confirmed_hitters if h["score"] >= 65][:5]
            
            if not top_hitters:
                st.info("⏳ Waiting for lineup confirmations. Check back closer to game time!")
            else:
                for i, h in enumerate(top_hitters, 1):
                    r = h["recent"]
                    h_hand = h.get("bat_side", "R")
                    p_hand = h.get("pitcher_hand", "R")
                    matchup, _ = get_matchup_grade(r.get("k_rate", 0.22), h["pitcher_k_pct"],
                                                   h_hand, p_hand)
                    st.markdown(f"""
                    <div style='background: #fef3c7; padding: 1rem; border-radius: 10px;
                                margin-bottom: 0.5rem; border-left: 4px solid #f59e0b;'>
                        <strong>#{i} {h['name']} ({h_hand}HB)</strong> ({h['team']}) - Batting #{h.get('batting_order', '?')}<br>
                        vs {h['vs_pitcher']} ({p_hand}HP) | {matchup}<br>
                        L7: <strong>{r.get('avg', 0):.3f} AVG</strong> / {r.get('ops', 0):.3f} OPS<br>
                        {f"🔥 {r.get('hit_streak', 0)}-game hit streak" if r.get('hit_streak', 0) >= 3 else ""}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("#### 🎲 Combined Play Ideas")
        
        if top_pitchers and top_hitters:
            st.markdown(f"""
            1. **Pitcher K Parlay:** {top_pitchers[0]['pitcher']} 5+ Ks + {top_pitchers[1]['pitcher'] if len(top_pitchers) > 1 else 'N/A'} 5+ Ks
            2. **Hot Bat + K:** {top_hitters[0]['name']} Hit + {top_pitchers[0]['pitcher']} 6+ Ks
            3. **Fade Cold:** Look for high-K hitters in cold streaks vs elite SALCI pitchers
            """)
        elif top_pitchers:
            st.markdown(f"""
            1. **Pitcher K Parlay:** {top_pitchers[0]['pitcher']} 5+ Ks + {top_pitchers[1]['pitcher'] if len(top_pitchers) > 1 else 'N/A'} 5+ Ks
            
            ⏳ *Hitter plays will be available once lineups are confirmed*
            """)
        else:
            st.info("Check back closer to game time for play recommendations!")
    
    # Tab 4: Charts & Share (NEW!)
    with tab4:
        st.markdown("### 📊 Shareable Charts & Insights")
        st.markdown("*Screenshot these charts for Twitter/X posts! All include #SALCI branding.*")
        
        st.markdown("---")
        
        # ===== GAME DAY CARD - Featured at top =====
        st.markdown("#### 📱 Game Day Card")
        st.markdown("*Perfect for your morning tweet! Screenshot and share.*")
        
        # Calculate stats for the card
        top_pitcher = all_pitcher_results[0] if all_pitcher_results else None
        confirmed_hitters_list = [h for h in all_hitter_results if h.get("lineup_confirmed")]
        top_hitter = confirmed_hitters_list[0] if confirmed_hitters_list else (all_hitter_results[0] if all_hitter_results else None)
        elite_count = len([p for p in all_pitcher_results if p["salci"] >= 75])
        strong_count = len([p for p in all_pitcher_results if 60 <= p["salci"] < 75])
        hot_hitters_count = len([h for h in all_hitter_results if h.get("score", 0) >= 70])
        lineups_confirmed = confirmed_count > 0
        
        # Render the Game Day Card
        create_game_day_card(
            selected_date=selected_date,
            num_games=len(games),
            top_pitcher=top_pitcher,
            top_hitter=top_hitter,
            elite_count=elite_count,
            strong_count=strong_count,
            hot_hitters_count=hot_hitters_count,
            lineups_confirmed=lineups_confirmed
        )
        
        # Sample tweet for copy/paste
        if top_pitcher:
            st.markdown("---")
            st.markdown("##### 📝 Sample Tweet (copy & paste!)")
            p_hand = top_pitcher.get('pitcher_hand', 'R')
            h_hand = top_hitter.get('bat_side', 'R') if top_hitter else 'R'
            h_vs_hand = top_hitter.get('pitcher_hand', 'R') if top_hitter else 'R'
            
            tweet_text = f"""🚨 SALCI Game Day - {selected_date.strftime('%b %d')} 🚨

🎯 Top K Play: {top_pitcher['pitcher']} ({p_hand}HP)
• SALCI: {top_pitcher['salci']}
• 6+ Ks: {top_pitcher['lines'].get(6, 0)}%
• 7+ Ks: {top_pitcher['lines'].get(7, 0)}%

{"🔥 Hot Bat: " + top_hitter['name'] + " (" + h_hand + "HB) vs " + h_vs_hand + "HP" + chr(10) + "• .{:03d} AVG L7".format(int(top_hitter['recent'].get('avg', 0)*1000)) if top_hitter else "⏳ Lineups pending..."}

{elite_count} Elite | {strong_count} Strong | {hot_hitters_count} Hot Bats

#SALCI #MLB"""
            
            st.code(tweet_text, language=None)
        
        st.markdown("---")
        
        # Charts in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Pitcher SALCI Rankings")
            fig_pitchers = create_pitcher_comparison_chart(all_pitcher_results)
            if fig_pitchers:
                st.plotly_chart(fig_pitchers, use_container_width=True)
            else:
                st.info("No pitcher data available")
        
        with col2:
            st.markdown("#### 🔥 Hot Hitters (L7)")
            fig_hitters = create_hitter_hotness_chart(all_hitter_results)
            if fig_hitters:
                st.plotly_chart(fig_hitters, use_container_width=True)
            else:
                st.info("No hitter data available")
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### 🎯 K Line Projections")
            fig_k_lines = create_k_projection_chart(all_pitcher_results)
            if fig_k_lines:
                st.plotly_chart(fig_k_lines, use_container_width=True)
            else:
                st.info("No pitcher data available")
        
        with col4:
            st.markdown("#### 🧮 SALCI Formula Breakdown")
            fig_breakdown = create_salci_breakdown_chart()
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
        st.markdown("---")
        
        # Scatter plot full width
        st.markdown("#### 📊 Hitter Profile: K% vs AVG")
        fig_scatter = create_matchup_scatter(all_hitter_results)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Need at least 3 hitters for scatter plot")
        
        st.markdown("---")
        
        # Twitter tips
        with st.expander("📱 Tips for Sharing on Twitter/X"):
            st.markdown("""
            **How to share these charts:**
            
            1. **Screenshot** the chart you want to share
            2. **Crop** to focus on the visual
            3. **Post** with hashtag #SALCI
            
            **Sample tweet templates:**
            
            > 🚨 Today's top SALCI pitcher: [Name] at 82 🔥
            > 
            > Expected: 7.2 Ks | 6+ @ 78%
            > 
            > The matchup + metrics check out. Let's ride.
            > 
            > #SALCI #MLB
            
            ---
            
            > 🔥 Hottest bats in baseball right now (L7):
            > 
            > [attach Hot Hitters chart]
            > 
            > SALCI tracks AVG, OPS, K%, and streaks. These guys are locked in.
            > 
            > #SALCI
            """)


if __name__ == "__main__":
    main()
