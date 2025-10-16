#!/usr/bin/env python3
"""
test_movie_recommender.py
Jest-like runner for movie_recommender.py

Rules:
1) Movie-file tests: if a movie file is meant to ABORT, do NOT test ratings.
2) Rating-file tests: always pair with one known-good movies file.
3) Integration smoke tests: run every menu option (incl. reload) on valid data.

Run:
  python test_movie_recommender.py
"""

from __future__ import annotations

import builtins
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Pretty printing (ANSI)
# ----------------------------
def _c(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"

GREEN = lambda s: _c(s, "32")
RED   = lambda s: _c(s, "31")
YEL   = lambda s: _c(s, "33")
CYAN  = lambda s: _c(s, "36")
DIM   = lambda s: _c(s, "2")
BOLD  = lambda s: _c(s, "1")

CHECK = GREEN("✓")
CROSS = RED("✗")
SKIP  = YEL("↷")

# ----------------------------
# Import SUT
# ----------------------------
try:
    import movie_recommender as mr
except Exception as e:
    print(RED("FATAL: Could not import movie_recommender.py. Make sure it's alongside this file."))
    print(e)
    sys.exit(1)

# ----------------------------
# Expected outcomes (hard-coded comparator)
# ----------------------------
EXPECTED_MOVIES: Dict[str, str] = {
    "movies_01_valid_small.txt": "OK",
    "movies_02_valid_medium.txt": "OK",
    "movies_03_valid_large_150.txt": "OK",
    "movies_04_blank_lines_and_spaces.txt": "OK",
    "movies_05_wrong_delimiter_commas.txt": "ABORT",
    "movies_06_missing_field_too_few_columns.txt": "ABORT",
    "movies_07_extra_field_too_many_columns.txt": "ABORT",
    "movies_08_non_integer_id.txt": "ABORT",
    "movies_09_negative_id.txt": "ABORT",
    "movies_10_duplicate_id_conflict.txt": "ABORT",
    "movies_11_duplicate_name_same_year_diff_id.txt": "ABORT",
    "movies_12_case_only_duplicate_merge_ok.txt": "OK",
    "movies_13_missing_year_in_name.txt": "ABORT",
    "movies_14_malformed_year_text.txt": "ABORT",
    "movies_15_missing_closing_paren.txt": "ABORT",
    "movies_16_trailing_delimiter.txt": "ABORT",
    "movies_17_unicode_whitespace_nbsp.txt": "OK",
    "movies_18_bom_utf8.txt": "OK",
    "movies_19_mixed_newlines.txt": "OK",
    "movies_20_genre_case_variants.txt": "OK",
    "movies_21_very_long_title.txt": "OK",
    "movies_22_exact_duplicate_lines.txt": "OK",
    "movies_23_whitespace_variation_in_year_format.txt": "ABORT",
    "movies_24_tab_delimited.txt": "ABORT",
    "movies_25_trailing_leading_spaces_only.txt": "OK",
}

EXPECTED_RATINGS: Dict[str, str] = {
    # OK files
    "ratings_01_valid_small.txt": "OK",
    "ratings_02_valid_medium.txt": "OK",
    "ratings_03_valid_large_200.txt": "OK",
    "ratings_04_blank_lines_and_spaces.txt": "OK",
    "ratings_09_out_of_range_ratings.txt": "OK",  # out-of-range rows should be skipped
    "ratings_12_duplicate_user_movie_keep_first.txt": "OK",
    "ratings_13_unknown_movies_rows_skipped.txt": "OK",  # unknown titles should be skipped
    "ratings_14_case_only_duplicate_same_user.txt": "OK",
    "ratings_15_year_whitespace_variation.txt": "OK",
    "ratings_16_bom_utf8.txt": "OK",
    "ratings_17_mixed_newlines.txt": "OK",
    "ratings_18_unicode_whitespace_nbsp.txt": "OK",
    "ratings_20_trailing_leading_spaces_only.txt": "OK",
    "ratings_21_exact_duplicate_lines.txt": "OK",
    "ratings_22_very_long_movie_name.txt": "OK",
    "ratings_24_large_user_ids.txt": "OK",
    "ratings_25_mixed_valid_outofrange_unknown.txt": "OK",

    # ABORT files
    "ratings_05_wrong_delimiter_commas.txt": "ABORT",
    "ratings_06_missing_field_too_few_columns.txt": "ABORT",
    "ratings_07_extra_field_too_many_columns.txt": "ABORT",
    "ratings_08_non_numeric_rating_text.txt": "ABORT",
    "ratings_08b_non_numeric_rating_NaN.txt": "ABORT",
    "ratings_08c_non_numeric_rating_inf.txt": "ABORT",
    "ratings_10_non_integer_user_id.txt": "ABORT",
    "ratings_11_negative_user_id.txt": "ABORT",
    "ratings_19_tab_delimited.txt": "ABORT",
    "ratings_23_float_like_user_id.txt": "ABORT",
}

# ----------------------------
# File discovery
# ----------------------------
ROOT = Path(__file__).resolve().parent
MOVIE_DIR  = ROOT / "testData" / "movie"
RATING_DIR = ROOT / "testData" / "rating"

def discover(base: Path, expected: Dict[str, str]) -> List[Path]:
    return [base / n for n in expected if (base / n).exists()]

# ----------------------------
# Input/Output patchers
# ----------------------------
class InputPatch:
    """Temporarily replace builtins.input with queued responses."""
    def __init__(self, seq: List[str], echo_prompts: bool = False):
        self.seq = list(seq)
        self._orig = None
        self.echo_prompts = echo_prompts
    def __enter__(self):
        self._orig = builtins.input
        def fake_input(prompt: str = "") -> str:
            if self.echo_prompts and prompt:
                # optionally echo prompt to stdout so tests can see it
                sys.stdout.write(prompt)
                sys.stdout.flush()
            return self.seq.pop(0) if self.seq else "q"
        builtins.input = fake_input
        return self
    def __exit__(self, exc_type, exc, tb):
        builtins.input = self._orig

class StdoutCapture:
    """Capture sys.stdout into a buffer."""
    def __enter__(self):
        self._orig = sys.stdout
        self.buf = io.StringIO()
        sys.stdout = self.buf
        return self
    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._orig
    def text(self) -> str:
        return self.buf.getvalue()

# ----------------------------
# Helpers to drive the CLI loader (authoritative for "ABORT")
# ----------------------------

def cli_try_movies_only(movies_path: str) -> str:
    """
    Drive mr.load_all_with_prompt once for movies:
      - Feed [movies_path, 'q'] so that if movies succeed we quit at the ratings prompt.
      - Determine success by presence of movies in globals after the call.
    """
    mr._clear_globals()
    try:
        with InputPatch([movies_path, "q"]), StdoutCapture() as cap:
            try:
                mr.load_all_with_prompt()
            except SystemExit:
                pass
        # movies OK iff movies were populated
        return "OK" if mr.MOVIES_BY_NAME else "ABORT"
    finally:
        # do not leave state hanging across tests
        mr._clear_globals()

def cli_try_movies_and_ratings(movies_path: str, ratings_path: str, expect: str) -> str:
    """
    Drive mr.load_all_with_prompt for ratings:
      - For expected OK: feed [movies_path, ratings_path], expect the loader to return normally.
      - For expected ABORT: feed [movies_path, ratings_path, 'q'], expect SystemExit with ratings cleared.
    """
    mr._clear_globals()
    if expect == "OK":
        try:
            with InputPatch([movies_path, ratings_path]), StdoutCapture() as cap:
                mr.load_all_with_prompt()
            # on success, ratings structures must be populated
            has_ratings = bool(mr.RATINGS_BY_MOVIE) and bool(mr.USER_IDS)
            return "OK" if has_ratings else "ABORT"
        except SystemExit:
            # if we exited, we did not finish successfully
            return "ABORT"
        finally:
            mr._clear_globals()
    else:  # expect ABORT
        try:
            with InputPatch([movies_path, ratings_path, "q"]), StdoutCapture() as cap:
                try:
                    mr.load_all_with_prompt()
                except SystemExit:
                    pass
            # If ratings aborted, RATINGS_* should be empty; movies should still be present (since movies OK).
            movies_ok = bool(mr.MOVIES_BY_NAME)
            ratings_empty = not mr.RATINGS_BY_MOVIE and not mr.USER_IDS
            return "ABORT" if movies_ok and ratings_empty else "OK"
        finally:
            mr._clear_globals()

# ----------------------------
# Feature runners (smoke)
# ----------------------------
def run_features_print_outputs() -> bool:
    """Returns True if all feature calls completed without exceptions."""
    ok = True
    saved_back = getattr(mr, "_back_or_quit", None)
    mr._back_or_quit = lambda: None  # non-blocking
    try:
        print("  " + DIM("running features…"))

        print("------ Feature: Movie Popularity (All) ------")
        mr.feature_movie_popularity()

        print("------ Feature: Movie Popularity in Genre (first genre) ------")
        with InputPatch(["1", "b"]):
            mr.feature_movie_popularity_in_genre()

        print("------ Feature: Genre Popularity ------")
        mr.feature_genre_popularity()

        uid = str(mr.USER_IDS[0]) if mr.USER_IDS else "b"
        print(f"------ Feature: User Preference for Genre (user {uid}) ------")
        with InputPatch([uid, "b"]):
            mr.feature_user_preference_for_genre()

        print(f"------ Feature: Recommend Movies (user {uid}) ------")
        with InputPatch([uid, "b"]):
            mr.feature_recommend_movies()

        # Reload data (provide same paths again)
        print("------ Feature: Reload Data ------")
        mr.feature_reload_data()

    except SystemExit:
        ok = False  # unexpected quit during features
    except Exception:
        ok = False
    finally:
        if saved_back is not None:
            mr._back_or_quit = saved_back
    return ok

# ----------------------------
# Choose a known-good movies file for ratings tests
# ----------------------------
PREFERRED_VALID_MOVIES_ORDER = [
    "movies_02_valid_medium.txt",
    "movies_01_valid_small.txt",
    "movies_03_valid_large_150.txt",
    "movies_04_blank_lines_and_spaces.txt",
    "movies_12_case_only_duplicate_merge_ok.txt",
    "movies_17_unicode_whitespace_nbsp.txt",
    "movies_18_bom_utf8.txt",
    "movies_19_mixed_newlines.txt",
    "movies_20_genre_case_variants.txt",
    "movies_21_very_long_title.txt",
    "movies_22_exact_duplicate_lines.txt",
    "movies_25_trailing_leading_spaces_only.txt",
]

def pick_baseline_valid_movies(movie_files: List[Path]) -> Optional[Path]:
    present = {p.name: p for p in movie_files}
    for name in PREFERRED_VALID_MOVIES_ORDER:
        if name in present:
            return present[name]
    # fallback: any file expected OK
    for p in movie_files:
        if EXPECTED_MOVIES.get(p.name) == "OK":
            return p
    return None

# ----------------------------
# Test runners
# ----------------------------
def run_movie_file_tests(movie_files: List[Path],
                         counters: Dict[str, int]) -> None:
    """Each movie file is its own suite. If expected ABORT, do not attempt ratings here."""
    for i, mpath in enumerate(movie_files, start=1):
        mname = mpath.name
        m_exp = EXPECTED_MOVIES[mname]
        print(BOLD(f"Movie Suite {i}") + " " + DIM(f"({mname})"))

        m_obs = cli_try_movies_only(str(mpath))
        if m_obs == m_exp:
            print(f"  {CHECK} movies load — expected {m_exp}, observed {m_obs}\n")
            counters["tests_passed"] += 1
            counters["suites_passed"] += 1
        else:
            print(f"  {CROSS} movies load — expected {m_exp}, observed {m_obs}\n")
            counters["tests_failed"] += 1
            counters["suites_failed"] += 1
        counters["suites_total"] += 1

def run_rating_file_tests(rating_files: List[Path],
                          baseline_movies: Path,
                          counters: Dict[str, int]) -> None:
    """Each ratings file is its own suite, paired with one known-good movies file, evaluated via the CLI loader."""
    for i, rpath in enumerate(rating_files, start=1):
        rname = rpath.name
        r_exp  = EXPECTED_RATINGS[rname]
        print(BOLD(f"Ratings Suite {i}") + " " + DIM(f"({rname} × {baseline_movies.name})"))

        r_obs = cli_try_movies_and_ratings(str(baseline_movies), str(rpath), r_exp)
        if r_obs == r_exp:
            print(f"  {CHECK} ratings load — expected {r_exp}, observed {r_obs}\n")
            counters["tests_passed"] += 1
            counters["suites_passed"] += 1
        else:
            print(f"  {CROSS} ratings load — expected {r_exp}, observed {r_obs}\n")
            counters["tests_failed"] += 1
            counters["suites_failed"] += 1

        counters["suites_total"] += 1

def run_integration_smoke_tests(baseline_movies: Path,
                                ok_ratings: List[Path],
                                counters: Dict[str, int]) -> None:
    """
    For a handful of valid ratings files, run all features (incl. reload) and count each feature call as a test.
    Uses direct loaders for speed, since these are valid files.
    """
    # Fallback direct loaders for valid-only integration
    def clear_and_load_movies(path: str) -> str:
        try:
            mr._clear_globals()
            mr.load_movies_file(path)
            return "OK"
        except mr.LoadError:
            return "ABORT"
        except Exception:
            return "ABORT"

    def load_ratings_and_compute(path: str) -> str:
        try:
            mr.load_ratings_file(path)
            mr.compute_movie_stats()
            mr.compute_genre_stats()
            mr.compute_user_top_genre_cache()
            return "OK"
        except mr.LoadError:
            return "ABORT"
        except Exception:
            return "ABORT"

    # Pick up to 3 OK ratings for variety
    picks: List[Path] = []
    preferred = [
        "ratings_02_valid_medium.txt",
        "ratings_21_exact_duplicate_lines.txt",
        "ratings_24_large_user_ids.txt",
        "ratings_18_unicode_whitespace_nbsp.txt",
        "ratings_16_bom_utf8.txt",
    ]
    byname = {p.name: p for p in ok_ratings}
    for name in preferred:
        if name in byname:
            picks.append(byname[name])
        if len(picks) == 3:
            break
    if not picks and ok_ratings:
        picks = ok_ratings[:3]

    if not picks:
        print(DIM("No OK ratings files available for integration smoke tests.\n"))
        return

    for i, rpath in enumerate(picks, start=1):
        print(BOLD(f"Integration Suite {i}") + " " + DIM(f"({baseline_movies.name} × {rpath.name})"))
        # load both OK
        m_obs = clear_and_load_movies(str(baseline_movies))
        r_obs = load_ratings_and_compute(str(rpath)) if m_obs == "OK" else "ABORT"

        if m_obs != "OK" or r_obs != "OK":
            print(f"  {CROSS} setup failed — movies:{m_obs} ratings:{r_obs}\n")
            counters["tests_failed"] += 1
            counters["suites_failed"] += 1
            counters["suites_total"] += 1
            continue

        # Run features (count 6 tests: 5 features + reload)
        try:
            # patch input for reload (last feature)
            with InputPatch(["y", str(baseline_movies), str(rpath)]):
                ok = run_features_print_outputs()
        except Exception:
            ok = False

        # Count tests: 6 features
        counters["tests_passed"] += 6 if ok else 0
        counters["tests_failed"] += 0 if ok else 6
        counters["suites_passed"] += 1 if ok else 0
        counters["suites_failed"] += 0 if ok else 1
        counters["suites_total"] += 1
        print()

# ----------------------------
# Main
# ----------------------------
def main() -> int:
    print(BOLD("=== test_movie_recommender.py ===\n"))

    movie_files  = discover(MOVIE_DIR, EXPECTED_MOVIES)
    rating_files = discover(RATING_DIR, EXPECTED_RATINGS)

    # Missing files diagnostics
    miss_movies  = [n for n in EXPECTED_MOVIES  if not (MOVIE_DIR / n).exists()]
    miss_ratings = [n for n in EXPECTED_RATINGS if not (RATING_DIR / n).exists()]
    if miss_movies:
        print(DIM("Missing movie test files (skipping those):"))
        for n in miss_movies: print(DIM(f"  - {n}"))
        print()
    if miss_ratings:
        print(DIM("Missing rating test files (skipping those):"))
        for n in miss_ratings: print(DIM(f"  - {n}"))
        print()

    if not movie_files:
        print(RED("No discoverable MOVIE test files.\n"))
        return 1

    counters = {
        "suites_total": 0,
        "suites_passed": 0,
        "suites_failed": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,  # left for future, currently not used in new scheme
    }

    start = time.perf_counter()

    # 1) Movie-only tests (no ratings attempted if expected ABORT)
    run_movie_file_tests(movie_files, counters)

    # 2) Ratings-only tests against a single valid movie file
    baseline = pick_baseline_valid_movies(movie_files)
    if baseline is None:
        print(YEL("No valid movies file found; ratings tests and integration smoke tests skipped.\n"))
    else:
        run_rating_file_tests(rating_files, baseline, counters)

        # 3) Integration smoke tests on valid data (exercise all menu options)
        ok_ratings = [p for p in rating_files if EXPECTED_RATINGS[p.name] == "OK"]
        run_integration_smoke_tests(baseline, ok_ratings, counters)

    elapsed = time.perf_counter() - start

    # Summary (Jest-like)
    print(BOLD("Summary"))
    print(f"  Test Suites: "
          f"{GREEN(str(counters['suites_passed']) + ' passed') if counters['suites_passed'] else '0 passed'}, "
          f"{RED(str(counters['suites_failed']) + ' failed') if counters['suites_failed'] else '0 failed'}, "
          f"0 skipped, "
          f"{counters['suites_total']} total")

    tests_total = counters["tests_passed"] + counters["tests_failed"] + counters["tests_skipped"]
    print(f"  Tests:       "
          f"{GREEN(str(counters['tests_passed']) + ' passed') if counters['tests_passed'] else '0 passed'}, "
          f"{RED(str(counters['tests_failed']) + ' failed') if counters['tests_failed'] else '0 failed'}, "
          f"{YEL(str(counters['tests_skipped']) + ' skipped') if counters['tests_skipped'] else '0 skipped'}, "
          f"{tests_total} total")

    print(f"  Time:        {elapsed:.2f}s\n")
    return 0 if counters["tests_failed"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
