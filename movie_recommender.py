#!/usr/bin/env python3
"""
Movie Recommendation System (CLI)
Python 3.12

Implements:
  1) Movie popularity (global)
  2) Movie popularity within a genre
  3) Genre popularity (average of movie averages)
  4) User preference for genre
  5) Recommend movies (top 3 unseen from user's top genre)
  6) Reload data
  7) Quit

Data & Parsing Rules:
- Movies file rows:   genre|movie_id|movie_name
  * Abort for: malformed rows, empty file, negative movie_id.
  * Abort for title/year format errors: movie_name must end with "(YYYY)" where YYYY is 4 digits.
  * Abort if the same conceptual movie (case-insensitive title, normalized whitespace, same year)
    appears with a different movie_id.
  * Abort if a movie_id is reused for a different conceptual movie.
  * Case-only duplicates with the same ID are allowed (keeps the first record).

- Ratings file rows:  movie_name|rating|user_id
  * Abort for: malformed rows, empty file, non-numeric or non-finite rating (NaN/inf), non-integer user_id,
    negative user_id anywhere in the file.
  * Skip (do NOT abort) rows with rating out of range [0, 5].
  * Skip (do NOT abort) rows whose movie_name is not found in the loaded movies.
  * Duplicate ratings (same user_id + movie_name): keep the first, ignore later duplicates.

- Genres are case-insensitive internally, but display with original casing from the movies file.
- Movie names are equal iff same letters ignoring case AND same length (year is part of the name) — used
  for canonicalizing display keys. Separately, "conceptual duplicates" are detected by normalized title + (YYYY).

CLI:
- Always accept q/Q to quit and b/B to go back.
- Numeric menu inputs may include a trailing period like "1." (treated as 1).
- After initial successful load, show main menu with all features and a "reload data" option
  that clears globals and re-runs the loader (with a small one-line loading animation).
- Display averages to two decimals.
"""

from __future__ import annotations

import os
import sys
import time
import re
import math
from typing import Dict, List, Tuple, Optional, Any

# =========================
# Global session datastore
# =========================

MOVIES_BY_ID: Dict[int, Dict[str, Any]] = {}
MOVIES_BY_NAME: Dict[str, Dict[str, Any]] = {}  # canonical movie name -> movie dict
GENRES: Dict[str, set[str]] = {}                # normalized_genre -> set of movie names (canonical)
RATINGS_BY_MOVIE: Dict[str, List[Tuple[int, float]]] = {}  # movie_name -> list of (user_id, rating)
RATINGS_BY_USER: Dict[int, Dict[str, float]] = {}          # user_id -> {movie_name: rating}
MOVIE_STATS: Dict[str, Dict[str, Any]] = {}     # movie_name -> {"avg": float, "count": int}
GENRE_STATS: Dict[str, Dict[str, Any]] = {}     # normalized_genre -> {"avg_of_movie_avgs": float, "total_ratings": int}
USER_IDS: List[int] = []
USER_TOP_GENRE: Dict[int, Tuple[str, float, int]] = {}     # user_id -> (norm_genre, avg_for_user_in_genre, count)

# For displaying original case of genres
GENRE_ORIGINAL_CASE: Dict[str, str] = {}        # normalized_genre -> original_case_seen_first


# =========================
# Utility / Helpers
# =========================

def _spinner_one_line(action_text: str, cycles: int = 10, delay: float = 0.06) -> None:
    """
    Show a short one-line spinner animation to indicate loading.
    """
    seq = "|/-\\"
    print(action_text, end="", flush=True)
    for i in range(cycles):
        print(f"\r{action_text} {seq[i % len(seq)]}", end="", flush=True)
        time.sleep(delay)
    print("\r" + " " * (len(action_text) + 2), end="\r", flush=True)


def _strip_int_like(s: str) -> Optional[int]:
    """
    Accept numeric inputs like "1" or "1." and return int(1). Returns None if not valid.
    """
    s = s.strip()
    if s.endswith("."):
        s = s[:-1]
    if s.isdigit() or (s and s[0] in "+-" and s[1:].isdigit()):
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _fmt_avg(x: float) -> str:
    """Format averages to two decimals."""
    return f"{x:.2f}"


def _norm_genre(g: str) -> str:
    """Normalize genre for internal keys (case-insensitive)."""
    return g.strip().lower()


def _case_insensitive_equal_same_length(a: str, b: str) -> bool:
    """
    Return True if strings are the same except for case; requires same length.
    (Year is part of the movie name and must match under this equality.)
    """
    return len(a) == len(b) and a.lower() == b.lower()


def _get_canonical_movie_name(new_name: str) -> str:
    """
    If a movie already exists in MOVIES_BY_NAME that matches new_name by case-insensitive-same-length rule,
    return the existing canonical key; otherwise, return new_name to become canonical.
    """
    for existing in MOVIES_BY_NAME.keys():
        if _case_insensitive_equal_same_length(existing, new_name):
            return existing
    return new_name


def _clear_globals() -> None:
    """Clear all global data structures."""
    MOVIES_BY_ID.clear()
    MOVIES_BY_NAME.clear()
    GENRES.clear()
    RATINGS_BY_MOVIE.clear()
    RATINGS_BY_USER.clear()
    MOVIE_STATS.clear()
    GENRE_STATS.clear()
    USER_IDS.clear()
    USER_TOP_GENRE.clear()
    GENRE_ORIGINAL_CASE.clear()


# =========================
# Parsing & Loading
# =========================

class LoadError(Exception):
    """Raised when a load operation must be aborted due to validation errors."""
    pass


def _parse_movies_line(line: str, line_no: int) -> Tuple[str, int, str]:
    """
    Parse a movies line: genre|movie_id|movie_name
    Returns (genre_original_case, movie_id, movie_name_display)
    Raises LoadError for malformed rows.
    """
    parts = line.rstrip("\n").split("|")
    if len(parts) != 3:
        raise LoadError(f"Movies file malformed at line {line_no}: expected 3 fields (genre|movie_id|movie_name).")
    genre, movie_id_s, movie_name = parts
    genre = genre.strip()
    movie_name = movie_name.strip()
    if not genre or not movie_id_s.strip() or not movie_name:
        raise LoadError(f"Movies file malformed at line {line_no}: empty field(s).")
    try:
        movie_id = int(movie_id_s)
    except ValueError:
        raise LoadError(f"Movies file malformed at line {line_no}: movie_id is not an integer.")
    return genre, movie_id, movie_name


# --- Concept-level (title + year) validation helpers for MOVIES ---
_YEAR_RE = re.compile(r'^(?P<title>.+?)\s*\(\s*(?P<year>\d{4})\s*\)\s*$')

def _extract_title_year(movie_name: str, line_no: int) -> tuple[str, int]:
    """
    Enforce 'Title (YYYY)' with a 4-digit year. Title is normalized by collapsing inner whitespace.
    """
    m = _YEAR_RE.match(movie_name)
    if not m:
        raise LoadError(f"Movies file malformed at line {line_no}: movie_name must end with (YYYY).")
    title = " ".join(m.group("title").split())  # collapse multiple spaces into single
    year = int(m.group("year"))
    return title, year

def _concept_key(movie_name: str, line_no: int) -> str:
    """
    Build a conceptual key 'title (YYYY)' with title lowercased and whitespace-normalized.
    """
    title, year = _extract_title_year(movie_name, line_no)
    return f"{title.lower()} ({year})"


def _parse_ratings_line(line: str, line_no: int) -> Tuple[str, float, int]:
    """
    Parse a ratings line: movie_name|rating|user_id
    Returns (movie_name, rating, user_id)
    Raises LoadError for malformed rows or non-numeric/non-finite rating or non-integer user_id.
    (Out-of-range rating and unknown movie handling are done in the loader and do not abort.)
    """
    parts = line.rstrip("\n").split("|")
    if len(parts) != 3:
        raise LoadError(f"Ratings file malformed at line {line_no}: expected 3 fields (movie_name|rating|user_id).")
    movie_name, rating_s, user_id_s = (p.strip() for p in parts)
    if not movie_name or not rating_s or not user_id_s:
        raise LoadError(f"Ratings file malformed at line {line_no}: empty field(s).")
    try:
        rating = float(rating_s)
    except ValueError:
        raise LoadError(f"Ratings file malformed at line {line_no}: rating is not numeric.")
    if not math.isfinite(rating):
        raise LoadError(f"Ratings file malformed at line {line_no}: rating must be finite (no NaN/inf).")
    try:
        user_id = int(user_id_s)
    except ValueError:
        raise LoadError(f"Ratings file malformed at line {line_no}: user_id is not an integer.")
    return movie_name, rating, user_id


def load_movies_file(path: str) -> None:
    """
    Load and validate the movies file. Populates:
      - MOVIES_BY_ID
      - MOVIES_BY_NAME
      - GENRES
      - GENRE_ORIGINAL_CASE
    Raises LoadError on any validation failure (abort load).
    """
    if not os.path.isfile(path):
        raise LoadError("Movies file does not exist.")

    # BOM tolerant
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln for ln in (l.strip("\n") for l in f) if ln.strip()]

    if not lines:
        raise LoadError("Movies file is empty.")

    # Track conceptual duplicates and id conflicts
    concept_to_id: Dict[str, int] = {}

    for i, raw in enumerate(lines, start=1):
        genre_original, movie_id, movie_name_raw = _parse_movies_line(raw, i)

        # negative id → abort
        if movie_id < 0:
            raise LoadError(f"Movies file malformed at line {i}: movie_id cannot be negative.")

        # Must look like Title (YYYY)
        ck = _concept_key(movie_name_raw, i)

        # same concept with different id → abort
        prev_id = concept_to_id.get(ck)
        if prev_id is not None and prev_id != movie_id:
            raise LoadError(
                f"Movies file malformed at line {i}: same title+year concept appears with different IDs."
            )
        concept_to_id.setdefault(ck, movie_id)

        # duplicate ID used for a different concept → abort
        if movie_id in MOVIES_BY_ID:
            existing_ck = _concept_key(MOVIES_BY_ID[movie_id]["name"], i)
            if existing_ck != ck:
                raise LoadError(
                    f"Movies file malformed at line {i}: movie_id {movie_id} reused for a different movie."
                )

        # Merge / canonicalize display name by case-insensitive same-length equivalence
        canonical_name = _get_canonical_movie_name(movie_name_raw)

        if canonical_name not in MOVIES_BY_NAME:
            # Insert movie record
            MOVIES_BY_NAME[canonical_name] = {
                "movie_id": movie_id,
                "name": canonical_name,                 # canonical name for display
                "genre_original": genre_original,       # original case for display
                "genre_norm": _norm_genre(genre_original),
            }
            MOVIES_BY_ID[movie_id] = MOVIES_BY_NAME[canonical_name]
        else:
            # If same display-name canonical but different ID → abort
            existing = MOVIES_BY_NAME[canonical_name]
            if existing["movie_id"] != movie_id:
                raise LoadError(
                    f"Movies file malformed at line {i}: same movie name (case-insensitive) with different IDs."
                )
            # else: duplicate line with same info → ignore

        # Track genre mapping
        norm_g = _norm_genre(genre_original)
        GENRES.setdefault(norm_g, set()).add(canonical_name)
        GENRE_ORIGINAL_CASE.setdefault(norm_g, genre_original)


def load_ratings_file(path: str) -> None:
    """
    Load and validate the ratings file. Populates:
      - RATINGS_BY_MOVIE
      - RATINGS_BY_USER
      - USER_IDS
      - (observes duplicate rating policy: keep first)
    Aborts on file-level validation failures; skips per-row issues as per rules.
    """
    if not os.path.isfile(path):
        raise LoadError("Ratings file does not exist.")

    # BOM tolerant
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln for ln in (l.strip("\n") for l in f) if ln.strip()]

    if not lines:
        raise LoadError("Ratings file is empty.")

    seen_user_movie: set[Tuple[int, str]] = set()

    for i, raw in enumerate(lines, start=1):
        movie_name_raw, rating, user_id = _parse_ratings_line(raw, i)

        # negative user id → abort whole file
        if user_id < 0:
            raise LoadError(f"Ratings file malformed at line {i}: user_id cannot be negative.")

        # out-of-range rating → skip row (do NOT abort)
        if not (0.0 <= rating <= 5.0):
            continue

        # map to canonical movie (case-insensitive, same-length rule)
        canonical_name = None
        for existing in MOVIES_BY_NAME.keys():
            if _case_insensitive_equal_same_length(existing, movie_name_raw):
                canonical_name = existing
                break

        # unknown movie → skip row (do NOT abort)
        if canonical_name is None:
            continue

        key = (user_id, canonical_name)
        if key in seen_user_movie:
            # duplicate rating for same user+movie -> ignore (keep first)
            continue
        seen_user_movie.add(key)

        RATINGS_BY_MOVIE.setdefault(canonical_name, []).append((user_id, rating))
        RATINGS_BY_USER.setdefault(user_id, {})[canonical_name] = rating

    # USER_IDS sorted
    USER_IDS[:] = sorted(RATINGS_BY_USER.keys())


def compute_movie_stats() -> None:
    """
    Compute MOVIE_STATS from RATINGS_BY_MOVIE.
    """
    MOVIE_STATS.clear()
    for movie_name, rating_list in RATINGS_BY_MOVIE.items():
        if not rating_list:
            continue
        total = sum(r for _, r in rating_list)
        cnt = len(rating_list)
        MOVIE_STATS[movie_name] = {"avg": (total / cnt), "count": cnt}


def compute_genre_stats() -> None:
    """
    Compute GENRE_STATS as average of movie averages per genre (exclude genres with zero ratings).
    Also compute total ratings per genre (sum of movie counts).
    """
    GENRE_STATS.clear()
    for norm_g, movie_names in GENRES.items():
        avgs: List[float] = []
        total_count = 0
        for m in movie_names:
            st = MOVIE_STATS.get(m)
            if st and st["count"] > 0:
                avgs.append(st["avg"])
                total_count += st["count"]
        if avgs:
            GENRE_STATS[norm_g] = {
                "avg_of_movie_avgs": sum(avgs) / len(avgs),
                "total_ratings": total_count,
            }


def compute_user_top_genre_cache() -> None:
    """
    Compute USER_TOP_GENRE: for each user, pick their preferred genre based on the
    user's average ratings within the genre (min count = 1). Ties broken by:
      - higher count within that genre by the user
      - genre name A-Z
    """
    USER_TOP_GENRE.clear()
    for uid, rated_map in RATINGS_BY_USER.items():
        # genre -> (sum, count)
        agg: Dict[str, Tuple[float, int]] = {}
        for movie_name, r in rated_map.items():
            movie = MOVIES_BY_NAME.get(movie_name)
            if not movie:
                continue
            gn = movie["genre_norm"]
            s, c = agg.get(gn, (0.0, 0))
            agg[gn] = (s + r, c + 1)

        if not agg:
            continue

        # pick best by avg desc, then count desc, then genre name A-Z
        best = sorted(
            ((gn, s / c, c) for gn, (s, c) in agg.items() if c >= 1),
            key=lambda tup: (-tup[1], -tup[2], GENRE_ORIGINAL_CASE.get(tup[0], tup[0]).lower()),
        )
        if best:
            top = best[0]  # (gn, avg, count)
            USER_TOP_GENRE[uid] = (top[0], top[1], top[2])


def load_all_with_prompt() -> None:
    """
    Interactive loader: prompts for movies file and ratings file, applies rules,
    shows a short loading animation, and computes all caches/stats.
    Re-prompts until successful or user quits.
    """
    while True:
        movies_path = input("Enter path to MOVIES file (or 'q' to quit): ").strip()
        if movies_path.lower() == "q":
            sys.exit(0)

        _spinner_one_line("Loading movies...")

        try:
            _clear_globals()
            load_movies_file(movies_path)
        except LoadError as e:
            print(f"[Error] {e}")
            continue
        except Exception as e:
            print(f"[Error] Unexpected error while loading movies: {e}")
            continue

        # Ratings file
        while True:
            ratings_path = input("Enter path to RATINGS file (or 'q' to quit): ").strip()
            if ratings_path.lower() == "q":
                sys.exit(0)

            _spinner_one_line("Loading ratings...")

            try:
                load_ratings_file(ratings_path)
                compute_movie_stats()
                compute_genre_stats()
                compute_user_top_genre_cache()
                print("✅ Data loaded successfully.\n")
                return
            except LoadError as e:
                print(f"[Error] {e}")
                # Reset ratings-dependent structures before re-prompting ratings file
                RATINGS_BY_MOVIE.clear()
                RATINGS_BY_USER.clear()
                MOVIE_STATS.clear()
                GENRE_STATS.clear()
                USER_IDS.clear()
                USER_TOP_GENRE.clear()
                continue
            except Exception as e:
                print(f"[Error] Unexpected error while loading ratings: {e}")
                RATINGS_BY_MOVIE.clear()
                RATINGS_BY_USER.clear()
                MOVIE_STATS.clear()
                GENRE_STATS.clear()
                USER_IDS.clear()
                USER_TOP_GENRE.clear()
                continue


# =========================
# Sorting Keys
# =========================

def _movie_sort_key(movie_name: str) -> Tuple[float, int, str, int]:
    """
    Sort key for movies:
      1) avg rating desc
      2) rating count desc
      3) movie name A-Z
      4) movie_id asc
    """
    st = MOVIE_STATS.get(movie_name, {"avg": 0.0, "count": 0})
    movie_id = MOVIES_BY_NAME.get(movie_name, {}).get("movie_id", 10**12)
    # negative for descending on avg and count
    return (-st["avg"], -st["count"], movie_name.lower(), movie_id)


def _genre_sort_key(norm_genre: str) -> Tuple[float, int, str]:
    """
    Sort key for genres (with ratings):
      1) avg_of_movie_avgs desc
      2) total_ratings desc
      3) genre name A-Z (display/original case)
    """
    st = GENRE_STATS.get(norm_genre, {"avg_of_movie_avgs": 0.0, "total_ratings": 0})
    display = GENRE_ORIGINAL_CASE.get(norm_genre, norm_genre)
    return (-st["avg_of_movie_avgs"], -st["total_ratings"], display.lower())


# =========================
# CLI Feature Implementations
# =========================

def feature_movie_popularity() -> None:
    """
    List all movies by popularity (avg desc, then count desc, then A-Z, then ID asc).
    """
    print("\n=== Movie Popularity (All) ===")
    all_movies = sorted(MOVIES_BY_NAME.keys(), key=_movie_sort_key)
    if not all_movies:
        print("No movies found.")
    else:
        for idx, name in enumerate(all_movies, start=1):
            m = MOVIES_BY_NAME[name]
            st = MOVIE_STATS.get(name, {"avg": 0.0, "count": 0})
            print(f"{idx}) {m['name']} — Avg: {_fmt_avg(st['avg'])} (Count: {st['count']}) — Genre: {m['genre_original']}")
    print()
    _back_or_quit()


def feature_movie_popularity_in_genre() -> None:
    """
    Prompt for a genre and list movies within it by popularity.
    """
    print("\n=== Movie Popularity by Genre ===")
    if not GENRES:
        print("No genres available.\n")
        _back_or_quit()
        return

    # Build display list of genres in original case, sorted A-Z
    genres_display = sorted({GENRE_ORIGINAL_CASE.get(g, g) for g in GENRES.keys()}, key=lambda s: s.lower())

    while True:
        print("Available genres:")
        for i, g in enumerate(genres_display, start=1):
            print(f"{i}) {g}")
        choice = input("\nSelect a genre by number, or 'b' to go back, or 'q' to quit: ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        if choice.lower() == "b":
            return
        idx = _strip_int_like(choice)
        if idx is None or not (1 <= idx <= len(genres_display)):
            print("Invalid selection. Please choose a listed number, or 'b'/'q'.\n")
            continue

        chosen_display = genres_display[idx - 1]
        norm = _norm_genre(chosen_display)
        movies_in_g = sorted(GENRES.get(norm, set()), key=_movie_sort_key)
        print(f"\n--- {chosen_display} ---")
        if not movies_in_g:
            print("No movies in this genre.")
        else:
            for j, name in enumerate(movies_in_g, start=1):
                m = MOVIES_BY_NAME[name]
                st = MOVIE_STATS.get(name, {"avg": 0.0, "count": 0})
                print(f"{j}) {m['name']} — Avg: {_fmt_avg(st['avg'])} (Count: {st['count']})")
        print()
        _back_or_quit()
        return


def feature_genre_popularity() -> None:
    """
    List genres ranked by average of movie averages (exclude zero-rated genres).
    """
    print("\n=== Genre Popularity ===")
    if not GENRE_STATS:
        print("No rated genres found.\n")
        _back_or_quit()
        return

    ranked = sorted(GENRE_STATS.keys(), key=_genre_sort_key)
    for i, g in enumerate(ranked, start=1):
        st = GENRE_STATS[g]
        disp = GENRE_ORIGINAL_CASE.get(g, g)
        print(f"{i}) {disp} — Avg of avgs: {_fmt_avg(st['avg_of_movie_avgs'])} (Total ratings: {st['total_ratings']})")
    print()
    _back_or_quit()


def feature_user_preference_for_genre() -> None:
    """
    Prompt for a user and display the user's preferred genre (avg of their ratings within a genre).
    """
    print("\n=== User Preference for Genre ===")
    if not USER_IDS:
        print("No user ratings available.\n")
        _back_or_quit()
        return

    _print_user_list()

    while True:
        s = input("\nEnter a user ID (value, not index), or 'b' to go back, or 'q' to quit: ").strip()
        if s.lower() == "q":
            sys.exit(0)
        if s.lower() == "b":
            return
        uid = _strip_int_like(s)
        if uid is None or uid not in RATINGS_BY_USER:
            print("Invalid user ID. Please choose an ID from the list, or 'b'/'q'.")
            continue

        # Compute or use cache
        top = USER_TOP_GENRE.get(uid)
        if not top:
            # compute on the fly (should already be cached, but safe)
            _compute_user_top_genre_for(uid)

        top = USER_TOP_GENRE.get(uid)
        if not top:
            print("No data for this user.\n")
        else:
            norm_g, avg, cnt = top
            disp = GENRE_ORIGINAL_CASE.get(norm_g, norm_g)
            print(f"\nTop Genre for User {uid}: {disp} — Your Avg: {_fmt_avg(avg)} (Count: {cnt})\n")
        _back_or_quit()
        return


def feature_recommend_movies() -> None:
    """
    Recommend up to 3 unseen movies from user's preferred genre, ranked by global popularity.
    """
    print("\n=== Recommend Movies ===")
    if not USER_IDS:
        print("No user ratings available.\n")
        _back_or_quit()
        return

    _print_user_list()

    while True:
        s = input("\nEnter a user ID (value, not index), or 'b' to go back, or 'q' to quit: ").strip()
        if s.lower() == "q":
            sys.exit(0)
        if s.lower() == "b":
            return
        uid = _strip_int_like(s)
        if uid is None or uid not in RATINGS_BY_USER:
            print("Invalid user ID. Please choose an ID from the list, or 'b'/'q'.")
            continue

        # Preferred genre
        top = USER_TOP_GENRE.get(uid)
        if not top:
            _compute_user_top_genre_for(uid)
            top = USER_TOP_GENRE.get(uid)

        if not top:
            print("\nNo data: this user has no ratings.\n")
            _back_or_quit()
            return

        norm_g, avg, cnt = top
        disp = GENRE_ORIGINAL_CASE.get(norm_g, norm_g)
        print(f"\nPreferred Genre for User {uid}: {disp} — Avg: {_fmt_avg(avg)} (Count: {cnt})")

        rated = set(RATINGS_BY_USER.get(uid, {}).keys())
        candidates = sorted(GENRES.get(norm_g, set()), key=_movie_sort_key)

        unseen: List[str] = [m for m in candidates if m not in rated]
        if not unseen:
            print("No unseen movies to recommend in this genre.\n")
            _back_or_quit()
            return

        recs = unseen[:3]
        print("\nTop Recommendations:")
        for i, name in enumerate(recs, start=1):
            st = MOVIE_STATS.get(name, {"avg": 0.0, "count": 0})
            print(f"{i}) {name} — Avg: {_fmt_avg(st['avg'])} (Count: {st['count']})")
        print()
        _back_or_quit()
        return


def feature_reload_data() -> None:
    """
    Confirm and reload data, clearing globals and re-running the loader.
    """
    print("\n=== Reload Data ===")
    while True:
        ans = input("Are you sure you want to reload? This will clear current data. (Y/N): ").strip().lower()
        if ans == "q":
            sys.exit(0)
        if ans == "b":
            return
        if ans in ("y", "yes"):
            _spinner_one_line("Clearing data...")
            _clear_globals()
            load_all_with_prompt()
            return
        if ans in ("n", "no"):
            print("Reload canceled.\n")
            return
        print("Please enter Y/N (or 'b' to go back, 'q' to quit).")


# =========================
# Internal computations
# =========================

def _compute_user_top_genre_for(uid: int) -> None:
    """
    Compute and place the top genre for a single user into USER_TOP_GENRE.
    """
    rated_map = RATINGS_BY_USER.get(uid, {})
    agg: Dict[str, Tuple[float, int]] = {}
    for movie_name, r in rated_map.items():
        movie = MOVIES_BY_NAME.get(movie_name)
        if not movie:
            continue
        gn = movie["genre_norm"]
        s, c = agg.get(gn, (0.0, 0))
        agg[gn] = (s + r, c + 1)
    if not agg:
        return
    best = sorted(
        ((gn, s / c, c) for gn, (s, c) in agg.items() if c >= 1),
        key=lambda tup: (-tup[1], -tup[2], GENRE_ORIGINAL_CASE.get(tup[0], tup[0]).lower()),
    )
    if best:
        top = best[0]
        USER_TOP_GENRE[uid] = (top[0], top[1], top[2])


def _print_user_list() -> None:
    """
    Print available user IDs in a compact sorted list.
    """
    if not USER_IDS:
        print("No users available.")
        return
    print("Available user IDs:")
    line = []
    for i, uid in enumerate(USER_IDS, start=1):
        line.append(str(uid))
        if len(line) == 12:
            print("  " + ", ".join(line))
            line = []
    if line:
        print("  " + ", ".join(line))


def _back_or_quit() -> None:
    """
    Wait for user to press 'b' to go back or 'q' to quit.
    """
    while True:
        s = input("Press 'b' to go back or 'q' to quit: ").strip().lower()
        if s == "q":
            sys.exit(0)
        if s == "b":
            return
        print("Invalid input. Please press 'b' or 'q'.")


# =========================
# Main Menu
# =========================

def main_menu() -> None:
    """
    Display the main menu and route to chosen features.
    """
    while True:
        print("=== Main Menu ===")
        print("1) Movie popularity (Top N overall)")
        print("2) Movie popularity in a genre")
        print("3) Genre popularity (Top N)")
        print("4) User preference for genre")
        print("5) Recommend movies")
        print("6) Reload data")
        print("7) Quit")

        choice = input("Choose an option (1-7), or 'q' to quit: ").strip()
        if choice.lower() == "q":
            sys.exit(0)

        num = _strip_int_like(choice)
        if num is None or not (1 <= num <= 7):
            print("Invalid choice. Please select 1-7 (or 'q' to quit).\n")
            continue

        if num == 1:
            feature_movie_popularity()
        elif num == 2:
            feature_movie_popularity_in_genre()
        elif num == 3:
            feature_genre_popularity()
        elif num == 4:
            feature_user_preference_for_genre()
        elif num == 5:
            feature_recommend_movies()
        elif num == 6:
            feature_reload_data()
        elif num == 7:
            sys.exit(0)


# =========================
# Entry Point
# =========================

def main() -> None:
    """
    Program entry point. Forces data load first, then shows main menu.
    """
    print("Movie Recommendation System (CLI)")
    print("Python 3.12\n")
    load_all_with_prompt()
    main_menu()


if __name__ == "__main__":
    main()
