"""Weekly US flu hospital-admission forecast.

Angelo Vescio

Pull CDC FluSight target data, keep the US series, and compare three
one-week-ahead forecasts:

1. A random forest that predicts next week's level.
2. A random forest that predicts next week's change, added back onto this week.
3. A persistence baseline: next week equals this week.

Scores come from walk-forward validation. Figures go to figures/, tables to
results/. The script downloads the target file, so it needs a network connection.

    python flu_forecast.py
"""

import os

import matplotlib

# Write figures to files. Agg does not open a window, so nothing sits waiting
# on plt.show().
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Knobs for the forecast. Change these here rather than hunting through the script.
DATA_URL = (
    "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/"
    "main/target-data/target-hospital-admissions.csv"
)
LOCATION = "US"
HORIZON = 1  # weeks ahead
HIGH_ACTIVITY_THRESHOLD = 20000  # admissions in the target week
N_SPLITS = 4
TEST_SIZE = 24  # weeks in each test window
N_ESTIMATORS = 100
RANDOM_STATE = 42
ROLLING_WEEKS = 8
N_LAGS = 5
# How many weeks on either side of a season peak to print.
PEAK_WINDOW_WEEKS = 3

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")

LAG_FEATURES = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
FEATURES = [
    "value",
    "rolling_avg",
    *LAG_FEATURES,
    "weekly_change",
    "growth_rate",
    "week_of_year",
]


def flu_season(date):
    """Label October–May as one flu season.

    October–December belong to the season that starts that calendar year.
    January–May belong to the season that started the year before.
    June–September are off-season and get no label.

    This is the same window as filtering date >= Oct 1 and date < Jun 1,
    without typing a separate start and end for every year.
    """
    if date.month >= 10:
        start_year = date.year
    elif date.month <= 5:
        start_year = date.year - 1
    else:
        return pd.NA
    return f"{start_year}-{start_year + 1}"


def load_us_admissions(url):
    """Download the FluSight target file and keep the national series."""
    flu_data = pd.read_csv(url)
    us_flu = flu_data.loc[flu_data["location_name"] == LOCATION].copy()
    us_flu["date"] = pd.to_datetime(us_flu["date"])
    us_flu = us_flu.sort_values("date").reset_index(drop=True)
    return us_flu


def add_features(us_flu):
    """Build the features and the one-week-ahead target.

    Each row is a forecast origin. `value` and the lags are known on that
    date. `target` is admissions HORIZON weeks later, so it is not known yet.
    """
    us_flu = us_flu.copy()
    us_flu["rolling_avg"] = us_flu["value"].rolling(window=ROLLING_WEEKS).mean()

    for lag in range(1, N_LAGS + 1):
        us_flu[f"lag_{lag}"] = us_flu["value"].shift(lag)

    us_flu["target"] = us_flu["value"].shift(-HORIZON)
    us_flu["target_change"] = us_flu["target"] - us_flu["value"]
    us_flu["weekly_change"] = us_flu["value"].diff()
    us_flu["growth_rate"] = us_flu["value"].pct_change()
    us_flu["week_of_year"] = us_flu["date"].dt.isocalendar().week.astype(int)

    # A 0 in the previous week would make the percent change infinite.
    us_flu = us_flu.replace([np.inf, -np.inf], np.nan)

    model_data = us_flu.dropna(subset=FEATURES + ["target", "target_change"]).copy()
    model_data = model_data.reset_index(drop=True)
    return us_flu, model_data


def season_summary(us_flu):
    """Peak and season total for every October–May season in the file.

    A season is complete when the file has both an October week and a May
    week for it. The first and last seasons are usually partial.
    """
    labeled = us_flu.copy()
    labeled["season"] = labeled["date"].map(flu_season)
    in_season = labeled.dropna(subset=["season"])

    rows = []
    for season, group in in_season.groupby("season", sort=True):
        peak_idx = group["value"].idxmax()
        has_october = bool((group["date"].dt.month == 10).any())
        has_may = bool((group["date"].dt.month == 5).any())
        rows.append(
            {
                "season": season,
                "complete": has_october and has_may,
                "weeks": int(len(group)),
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "peak_admissions": group.loc[peak_idx, "value"],
                "peak_date": group.loc[peak_idx, "date"],
                "total_admissions": group["value"].sum(),
            }
        )
    return pd.DataFrame(rows)


def weeks_around_peaks(us_flu, seasons):
    """A few weeks on either side of each season's computed peak date.

    The window is built from the peak date. It is not a date range typed in
    by hand.
    """
    frames = []
    for row in seasons.itertuples(index=False):
        start = row.peak_date - pd.Timedelta(weeks=PEAK_WINDOW_WEEKS)
        end = row.peak_date + pd.Timedelta(weeks=PEAK_WINDOW_WEEKS)
        window = us_flu.loc[
            (us_flu["date"] >= start) & (us_flu["date"] <= end),
            ["date", "value"],
        ].copy()
        window.insert(0, "season", row.season)
        window.insert(1, "peak_date", row.peak_date)
        window["weeks_from_peak"] = (
            (window["date"] - row.peak_date).dt.days // 7
        ).astype(int)
        frames.append(window)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_model():
    """Same forest each time: 100 trees, fixed seed, otherwise the defaults."""
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )


def high_activity_mae(y_true, y_pred):
    """MAE on target weeks at or above the high-activity cutoff.

    Returns (mae, n_weeks). mae is NaN when the test window has no such week.
    The cutoff is on the actual target, so every model is scored on the same weeks.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true >= HIGH_ACTIVITY_THRESHOLD
    n_weeks = int(mask.sum())
    if n_weeks == 0:
        return np.nan, 0
    return float(mean_absolute_error(y_true[mask], y_pred[mask])), n_weeks


def peak_error(target_dates, y_true, y_pred):
    """How far off the predicted peak is, in admissions and in weeks.

    Timing error is predicted peak week minus actual peak week. Positive
    means the model put the peak too late. Dates in this file are 7 days
    apart, so an index step is one week.

    Magnitude error compares the height of those two weeks. They are not
    always the same week. Persistence copies this week into next week's
    forecast, so the real peak often shows up one week late with the same
    height. That is a delay, not an early call of the peak.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    actual_i = int(np.argmax(y_true))
    predicted_i = int(np.argmax(y_pred))
    return {
        "actual_peak": y_true[actual_i],
        "actual_peak_date": pd.Timestamp(target_dates.iloc[actual_i]),
        "predicted_peak": y_pred[predicted_i],
        "predicted_peak_date": pd.Timestamp(target_dates.iloc[predicted_i]),
        "magnitude_error": abs(y_true[actual_i] - y_pred[predicted_i]),
        "timing_error_weeks": predicted_i - actual_i,
    }


def forecast_title(target_dates):
    """Title from the target weeks in the plot, not a hardcoded season name."""
    start = pd.Timestamp(np.min(target_dates))
    end = pd.Timestamp(np.max(target_dates))
    return (
        "Influenza forecast comparison "
        f"({start:%b %d, %Y} to {end:%b %d, %Y})"
    )


def save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def style_admissions_axis(values):
    """Comma-separated admissions, starting at zero when every value is positive."""
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    if np.nanmin(np.asarray(values, dtype=float)) >= 0:
        plt.ylim(bottom=0)


def plot_series(us_flu):
    """Weekly admissions and the 8-week rolling average."""
    plt.figure(figsize=(12, 5))
    plt.plot(
        us_flu["date"],
        us_flu["value"],
        label="Weekly hospital admissions",
        color="#1f77b4",
    )
    plt.plot(
        us_flu["date"],
        us_flu["rolling_avg"],
        label=f"{ROLLING_WEEKS}-week rolling average",
        color="#ff7f0e",
    )
    plt.title("US influenza hospital admissions")
    plt.xlabel("Week ending")
    plt.ylabel("Weekly hospital admissions")
    plt.legend()
    style_admissions_axis(us_flu["value"].to_numpy(dtype=float))
    save_current_figure(os.path.join(FIGURES_DIR, "us_admissions_rolling_average.png"))


def plot_forecast(dates, actual, level_pred, change_pred, baseline_pred, title, path):
    """Draw forecasts on the target week.

    `dates` must already be the week the prediction is for (origin + HORIZON),
    not the week the features came from. Plotting on the origin date puts
    every forecast one week too early.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual weekly hospital admissions", color="#1f77b4", linewidth=2)
    plt.plot(dates, level_pred, label="Random forest (level)", color="#ff7f0e")
    plt.plot(dates, change_pred, label="Random forest (change)", color="#2ca02c")
    plt.plot(
        dates,
        baseline_pred,
        label="Persistence baseline",
        color="#7f7f7f",
        linestyle="--",
    )
    # Skip the threshold line when the whole window is below it. A line at
    # 20,000 above a summer of ~1,000 admissions flattens the curves.
    plotted = np.concatenate(
        [
            np.asarray(actual, dtype=float),
            np.asarray(level_pred, dtype=float),
            np.asarray(change_pred, dtype=float),
            np.asarray(baseline_pred, dtype=float),
        ]
    )
    if np.nanmax(plotted) >= HIGH_ACTIVITY_THRESHOLD:
        plt.axhline(
            HIGH_ACTIVITY_THRESHOLD,
            color="#7f7f7f",
            linestyle=":",
            linewidth=1,
            label=f"High-activity threshold ({HIGH_ACTIVITY_THRESHOLD:,})",
        )
    plt.title(title)
    plt.xlabel("Target week ending")
    plt.ylabel("Weekly hospital admissions")
    plt.legend()
    style_admissions_axis(plotted)
    save_current_figure(path)


def walk_forward(model_data):
    """Score the three forecasts on later weeks only.

    TimeSeriesSplit walks forward. Each fold trains on every earlier row and
    tests on the next TEST_SIZE weeks. The window then steps forward by one
    test length. A row's features are known on its date. Its target is
    admissions HORIZON weeks later. Training rows all come before test rows,
    so the model is not trained on a week it is scored on.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    fold_rows = []
    peak_rows = []
    forecast_frames = []
    last_importance = None

    for fold, (train_index, test_index) in enumerate(tscv.split(model_data), start=1):
        # 1. Earlier block = train, later block = test.
        train_fold = model_data.iloc[train_index]
        test_fold = model_data.iloc[test_index]

        # Keep the last training row.
        # Its target is admissions on the first test date. That number is the
        # first test row's current value, which is already known when we
        # forecast that row. The label we score is the week after that.
        # Dropping this row (train_fold.iloc[:-1]) does not fix leakage. It
        # only throws away one training week.

        # 2. Features are known at the origin date. y is the target week.
        X_train = train_fold[FEATURES]
        X_test = test_fold[FEATURES]
        y_test = test_fold["target"].to_numpy(dtype=float)

        # FluSight dates are the Saturday that ends the week, and this file
        # is one row per week. The target week is HORIZON weeks after the origin.
        target_dates = test_fold["date"] + pd.Timedelta(weeks=HORIZON)

        # 3. Level model: predict next week's admissions directly.
        # A random forest predicts an average of training outcomes, so this
        # forecast cannot go above the highest target in the training block.
        level_model = make_model()
        level_model.fit(X_train, train_fold["target"])
        level_pred = level_model.predict(X_test)

        # 4. Change model: predict the rise or fall from this week to next
        # week, then add it to this week's known admissions. The predicted
        # change is still bounded by training changes, but adding it to the
        # current level can land above the highest level in the training block.
        change_model = make_model()
        change_model.fit(X_train, train_fold["target_change"])
        predicted_change = change_model.predict(X_test)
        change_pred = test_fold["value"].to_numpy(dtype=float) + predicted_change

        # 5. Persistence baseline: guess that next week equals this week.
        baseline_pred = test_fold["value"].to_numpy(dtype=float)

        # 6. MAE on every test week, and again on high-activity weeks only.
        # High-activity MAE is computed for all three models on the same weeks.
        level_mae = float(mean_absolute_error(y_test, level_pred))
        change_mae = float(mean_absolute_error(y_test, change_pred))
        baseline_mae = float(mean_absolute_error(y_test, baseline_pred))

        level_high, n_high = high_activity_mae(y_test, level_pred)
        change_high, _ = high_activity_mae(y_test, change_pred)
        baseline_high, _ = high_activity_mae(y_test, baseline_pred)

        # 7. Peak timing and height for each model, on this fold's target weeks.
        # See peak_error: a perfect baseline height is usually the real peak
        # copied forward one week.
        model_predictions = {
            "level": level_pred,
            "change": change_pred,
            "baseline": baseline_pred,
        }
        for model_name, preds in model_predictions.items():
            stats = peak_error(target_dates, y_test, preds)
            stats["fold"] = fold
            stats["model"] = model_name
            peak_rows.append(stats)

        fold_rows.append(
            {
                "fold": fold,
                "train_start": train_fold["date"].min(),
                "train_end": train_fold["date"].max(),
                "n_train": int(len(train_fold)),
                "test_start": target_dates.min(),
                "test_end": target_dates.max(),
                "n_test": int(len(test_fold)),
                "highest_training_value": float(train_fold["value"].max()),
                "highest_training_target": float(train_fold["target"].max()),
                "highest_test_target": float(y_test.max()),
                "level_pred_max": float(level_pred.max()),
                "change_pred_max": float(change_pred.max()),
                "level_mae": level_mae,
                "change_mae": change_mae,
                "baseline_mae": baseline_mae,
                "level_high_activity_mae": level_high,
                "change_high_activity_mae": change_high,
                "baseline_high_activity_mae": baseline_high,
                "high_activity_weeks": n_high,
            }
        )

        forecast_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "origin_date": test_fold["date"].to_numpy(),
                    "target_date": target_dates.to_numpy(),
                    "actual": y_test,
                    "level_pred": level_pred,
                    "change_pred": change_pred,
                    "baseline_pred": baseline_pred,
                }
            )
        )

        # Feature importance from the last fold's level model. That fold has
        # the longest training history.
        if fold == N_SPLITS:
            last_importance = (
                pd.DataFrame(
                    {
                        "feature": FEATURES,
                        "importance": level_model.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        # One comparison chart per fold. The title uses this fold's dates.
        plot_forecast(
            target_dates,
            y_test,
            level_pred,
            change_pred,
            baseline_pred,
            forecast_title(target_dates),
            os.path.join(FIGURES_DIR, f"fold_{fold}_comparison.png"),
        )

        print(
            f"Fold {fold}: train {train_fold['date'].min():%Y-%m-%d} to "
            f"{train_fold['date'].max():%Y-%m-%d} "
            f"({len(train_fold)} weeks) | "
            f"target {target_dates.min():%Y-%m-%d} to {target_dates.max():%Y-%m-%d}"
        )
        print(
            f"  MAE  level {level_mae:.1f} | change {change_mae:.1f} | "
            f"baseline {baseline_mae:.1f}"
        )
        if n_high:
            print(
                f"  high-activity weeks {n_high} | "
                f"MAE level {level_high:.1f} | change {change_high:.1f} | "
                f"baseline {baseline_high:.1f}"
            )
        else:
            print("  high-activity weeks 0 | MAE not defined for this fold")
        print(
            f"  highest training target {train_fold['target'].max():.0f} | "
            f"highest test target {y_test.max():.0f} | "
            f"level pred max {level_pred.max():.2f} | "
            f"change pred max {change_pred.max():.2f}"
        )

    folds = pd.DataFrame(fold_rows)
    peaks = pd.DataFrame(peak_rows)
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    return folds, peaks, forecasts, last_importance


def with_average_row(folds, forecasts):
    """Add an average row to the per-fold MAE table.

    Overall MAE is the mean of the fold MAEs. Every fold has the same number
    of test weeks, so that equals the MAE on all test weeks pooled together.

    High-activity MAE on the average row is different. It is the MAE on every
    high-activity test week pooled, not the mean of the fold scores. A fold
    with 10 of those weeks should count more than a fold with 4. Folds with
    none are already left blank above. `high_activity_weeks` on this row is
    the total, not a mean.
    """
    high = forecasts.loc[forecasts["actual"] >= HIGH_ACTIVITY_THRESHOLD]
    if high.empty:
        level_high = np.nan
        change_high = np.nan
        baseline_high = np.nan
    else:
        level_high = float(mean_absolute_error(high["actual"], high["level_pred"]))
        change_high = float(mean_absolute_error(high["actual"], high["change_pred"]))
        baseline_high = float(mean_absolute_error(high["actual"], high["baseline_pred"]))

    average = {
        "fold": "average",
        "test_start": folds["test_start"].min(),
        "test_end": folds["test_end"].max(),
        "n_test": int(folds["n_test"].sum()),
        "high_activity_weeks": int(len(high)),
        "level_mae": float(folds["level_mae"].mean()),
        "change_mae": float(folds["change_mae"].mean()),
        "baseline_mae": float(folds["baseline_mae"].mean()),
        "level_high_activity_mae": level_high,
        "change_high_activity_mae": change_high,
        "baseline_high_activity_mae": baseline_high,
    }
    return pd.concat([folds, pd.DataFrame([average])], ignore_index=True)


def format_number(value, digits):
    if pd.isna(value):
        return ""
    return f"{float(value):,.{digits}f}"


def format_plain(value, digits):
    """Fixed decimals for the text summary. Blank scores stay NA."""
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def format_date(value):
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def write_mae_markdown(folds_with_avg, path):
    """A small table of per-fold and average MAE for the README."""
    headers = [
        "Fold",
        "Target weeks",
        "Level MAE",
        "Change MAE",
        "Baseline MAE",
        "High-activity weeks",
        "Level high-activity MAE",
        "Change high-activity MAE",
        "Baseline high-activity MAE",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in folds_with_avg.itertuples(index=False):
        window = f"{format_date(row.test_start)} to {format_date(row.test_end)}"
        if str(row.fold) == "average":
            fold_label = "Average"
        else:
            fold_label = str(int(row.fold))
        cells = [
            fold_label,
            window,
            format_number(row.level_mae, 1),
            format_number(row.change_mae, 1),
            format_number(row.baseline_mae, 1),
            f"{int(row.high_activity_weeks)}",
            format_number(row.level_high_activity_mae, 1),
            format_number(row.change_high_activity_mae, 1),
            format_number(row.baseline_high_activity_mae, 1),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Overall MAE in the average row is the mean of the fold MAEs. "
        "Each fold has the same number of test weeks, so this is also the MAE "
        "on every test week together."
    )
    lines.append("")
    lines.append(
        "High-activity MAE in the average row is the MAE on every test week "
        f"with at least {HIGH_ACTIVITY_THRESHOLD:,} admissions, pooled across folds. "
        "It is not the unweighted mean of the fold high-activity scores, because "
        "those folds do not have the same number of high-activity weeks."
    )
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def write_summary(path, us_flu, model_data, seasons, folds, peaks, importance, average):
    """Plain-text snapshot of the numbers used in the README."""
    peak_idx = us_flu["value"].idxmax()
    lines = []
    lines.append(f"series_start={format_date(us_flu['date'].min())}")
    lines.append(f"series_end={format_date(us_flu['date'].max())}")
    lines.append(f"n_weeks={len(us_flu)}")
    lines.append(f"model_rows={len(model_data)}")
    lines.append(f"peak_admissions={us_flu.loc[peak_idx, 'value']:.0f}")
    lines.append(f"peak_date={format_date(us_flu.loc[peak_idx, 'date'])}")
    lines.append(f"mean_admissions={us_flu['value'].mean():.1f}")
    lines.append(f"horizon_weeks={HORIZON}")
    lines.append(f"n_splits={N_SPLITS}")
    lines.append(f"test_size={TEST_SIZE}")
    lines.append(f"high_activity_threshold={HIGH_ACTIVITY_THRESHOLD}")
    lines.append(f"n_estimators={N_ESTIMATORS}")
    lines.append(f"random_state={RANDOM_STATE}")
    lines.append(f"rolling_weeks={ROLLING_WEEKS}")
    lines.append(f"n_lags={N_LAGS}")
    lines.append("")
    lines.append("seasons")
    for row in seasons.itertuples(index=False):
        status = "complete" if row.complete else "partial"
        lines.append(
            f"  {row.season} {status} weeks={row.weeks} "
            f"start={format_date(row.start_date)} end={format_date(row.end_date)} "
            f"peak={row.peak_admissions:.0f} peak_date={format_date(row.peak_date)} "
            f"total={row.total_admissions:.0f}"
        )
    lines.append("")
    lines.append("folds")
    for row in folds.itertuples(index=False):
        lines.append(
            f"  fold={row.fold} n_train={row.n_train} "
            f"train={format_date(row.train_start)}..{format_date(row.train_end)} "
            f"target={format_date(row.test_start)}..{format_date(row.test_end)} "
            f"n_test={row.n_test} high_weeks={row.high_activity_weeks} "
            f"level_mae={row.level_mae:.1f} change_mae={row.change_mae:.1f} "
            f"baseline_mae={row.baseline_mae:.1f} "
            f"level_high={format_plain(row.level_high_activity_mae, 1)} "
            f"change_high={format_plain(row.change_high_activity_mae, 1)} "
            f"baseline_high={format_plain(row.baseline_high_activity_mae, 1)} "
            f"highest_training_value={row.highest_training_value:.0f} "
            f"highest_training_target={row.highest_training_target:.0f} "
            f"highest_test_target={row.highest_test_target:.0f} "
            f"level_pred_max={row.level_pred_max:.2f} "
            f"change_pred_max={row.change_pred_max:.2f}"
        )
    lines.append("")
    lines.append("averages")
    lines.append(
        f"  level_mae={average['level_mae']:.1f} "
        f"change_mae={average['change_mae']:.1f} "
        f"baseline_mae={average['baseline_mae']:.1f}"
    )
    lines.append(
        "  high_activity_definition=pooled MAE on every test week "
        f"at or above {HIGH_ACTIVITY_THRESHOLD}"
    )
    lines.append(
        f"  high_weeks={int(average['high_activity_weeks'])} "
        f"level_high={average['level_high_activity_mae']:.1f} "
        f"change_high={average['change_high_activity_mae']:.1f} "
        f"baseline_high={average['baseline_high_activity_mae']:.1f}"
    )
    lines.append("")
    lines.append("peaks")
    for row in peaks.itertuples(index=False):
        lines.append(
            f"  fold={row.fold} model={row.model} "
            f"actual={row.actual_peak:.0f} on {format_date(row.actual_peak_date)} "
            f"predicted={row.predicted_peak:.2f} on {format_date(row.predicted_peak_date)} "
            f"magnitude_error={row.magnitude_error:.2f} "
            f"timing_error_weeks={int(row.timing_error_weeks)}"
        )
    lines.append("")
    lines.append("feature_importance")
    for row in importance.itertuples(index=False):
        lines.append(f"  {row.feature}={row.importance:.4f}")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    us_flu = load_us_admissions(DATA_URL)
    us_flu, model_data = add_features(us_flu)

    peak_idx = us_flu["value"].idxmax()
    print(
        f"US weekly flu hospital admissions: {len(us_flu)} weeks, "
        f"{us_flu['date'].min():%Y-%m-%d} to {us_flu['date'].max():%Y-%m-%d}"
    )
    print(
        f"Highest week: {us_flu.loc[peak_idx, 'value']:.0f} "
        f"on {us_flu.loc[peak_idx, 'date']:%Y-%m-%d}"
    )
    print(f"Average week: {us_flu['value'].mean():.1f}")
    print(f"Rows used by the model (after lags and the target shift): {len(model_data)}")
    print()

    plot_series(us_flu)

    seasons = season_summary(us_flu)
    print("October–May seasons (peak and sum of weekly admissions):")
    for row in seasons.itertuples(index=False):
        status = "complete" if row.complete else "partial"
        print(
            f"  {row.season} ({status}, {row.weeks} weeks): "
            f"peak {row.peak_admissions:.0f} on {row.peak_date:%Y-%m-%d}, "
            f"sum {row.total_admissions:.0f}"
        )
    print()

    around = weeks_around_peaks(us_flu, seasons)
    seasons.to_csv(os.path.join(RESULTS_DIR, "season_summary.csv"), index=False)
    around.to_csv(os.path.join(RESULTS_DIR, "weeks_around_peaks.csv"), index=False)

    print(
        f"Walk-forward: {N_SPLITS} folds, {TEST_SIZE} test weeks each, "
        f"horizon {HORIZON} week"
    )
    folds, peaks, forecasts, importance = walk_forward(model_data)
    print()
    scored = with_average_row(folds, forecasts)
    average = scored.loc[scored["fold"] == "average"].iloc[0]
    print(
        "Average MAE: "
        f"level {average['level_mae']:.1f} | "
        f"change {average['change_mae']:.1f} | "
        f"baseline {average['baseline_mae']:.1f}"
    )
    print(
        "High-activity MAE on all such test weeks: "
        f"level {average['level_high_activity_mae']:.1f} | "
        f"change {average['change_high_activity_mae']:.1f} | "
        f"baseline {average['baseline_high_activity_mae']:.1f} "
        f"({int(average['high_activity_weeks'])} weeks)"
    )

    # Main forecast plot: every out-of-fold prediction, on the target week.
    plot_forecast(
        forecasts["target_date"],
        forecasts["actual"],
        forecasts["level_pred"],
        forecasts["change_pred"],
        forecasts["baseline_pred"],
        forecast_title(forecasts["target_date"]),
        os.path.join(FIGURES_DIR, "walk_forward_forecast.png"),
    )

    mae_columns = [
        "fold",
        "test_start",
        "test_end",
        "n_test",
        "high_activity_weeks",
        "level_mae",
        "change_mae",
        "baseline_mae",
        "level_high_activity_mae",
        "change_high_activity_mae",
        "baseline_high_activity_mae",
    ]
    scored[mae_columns].to_csv(os.path.join(RESULTS_DIR, "walk_forward_mae.csv"), index=False)
    folds.to_csv(os.path.join(RESULTS_DIR, "walk_forward_folds.csv"), index=False)
    peaks.to_csv(os.path.join(RESULTS_DIR, "peak_errors.csv"), index=False)
    forecasts.to_csv(os.path.join(RESULTS_DIR, "walk_forward_predictions.csv"), index=False)
    importance.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
    write_mae_markdown(scored, os.path.join(RESULTS_DIR, "walk_forward_mae.md"))
    write_summary(
        os.path.join(RESULTS_DIR, "summary.txt"),
        us_flu,
        model_data,
        seasons,
        folds,
        peaks,
        importance,
        average,
    )

    print()
    print("Feature importance (level model, last fold):")
    for row in importance.itertuples(index=False):
        print(f"  {row.feature}: {row.importance:.3f}")
    print()
    print("Wrote figures/ and results/.")


if __name__ == "__main__":
    main()
