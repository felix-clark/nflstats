#!/usr/bin/env python3

from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from db import plays
from nptyping import NDArray
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from utility import split_test_train

# TODO:
# Control for roof type (outdoors, dome, etc.). Some data is missing.


def get_loss(
    model: LogisticRegression, x_data_test: pd.DataFrame, y_data_test: pd.Series
) -> float:
    """
    Return logistic loss from model
    """
    assert (
        model.classes_ == np.array([0.0, 1.0])
    ).all(), "Y classes have incorrect order"
    test_log_prob: NDArray[float] = model.predict_log_proba(x_data_test)
    # test_prob: NDArray[float] = fit.predict_proba(x_data_test)
    # print(fit.classes_)  # [0., 1.]
    # print(test_log_prob)
    loss_terms: NDArray[float] = -y_data_test * test_log_prob.T[1] - (
        1.0 - y_data_test
    ) * test_log_prob.T[0]
    loss: float = np.sum(loss_terms) / len(y_data_test)
    return loss


def basic_regression(data: pd.DataFrame) -> LogisticRegression:
    """
    Regression of success probability as a function of kick distance
    """
    # TODO: treat blocked FGs separately
    data["fg_make"] = data["field_goal_result"].map(
        # {"made": True, "missed": False, "blocked": False}
        {"made": 1, "missed": 0, "blocked": 0}
    )
    data["kick_distance_sq"] = data["kick_distance"] ** 2
    train_params = ["kick_distance"]
    # train_params = ["kick_distance", 'kick_distance_sq']
    data_test, data_train = split_test_train(data)
    x_data_test = data_test[train_params]
    y_data_test = data_test["fg_make"]
    x_data_train = data_train[train_params]
    y_data_train = data_train["fg_make"]
    # regularization is on by default. Should this be turned off?
    reg = LogisticRegression(penalty="none")
    fit = reg.fit(x_data_train, y_data_train)

    # We need to add a constant for the intercept term
    sm_model = sm.GLM(y_data_train, sm.add_constant(x_data_train), sm.families.Binomial())
    sm_model_fit = sm_model.fit()
    print(sm_model_fit.summary())
    print(sm_model_fit.params)
    print(sm_model_fit.normalized_cov_params)
    # defaults to 95% confidence interval (0.05 argument)
    print(sm_model_fit.conf_int(.1))
    # standard error approximation (95% ~ 2*sigma, double-sided)
    print(0.25*(sm_model_fit.conf_int()[1] - sm_model_fit.conf_int()[0]))
    # print(dir(sm_model_fit))

    print(f"coefficients = {fit.coef_}")
    print(f"intercept = {fit.intercept_}")

    score: float = fit.score(x_data_test, y_data_test)
    print(f"score = {score}")
    loss = get_loss(fit, x_data_test, y_data_test)
    print(f"loss = {loss}")

    return fit


def player_regression(data: pd.DataFrame) -> LogisticRegression:
    """
    Regression of success probability treating each player's whole career separately
    """
    # TODO: treat blocked FGs separately
    data["fg_make"] = data["field_goal_result"].map(
        # {"made": True, "missed": False, "blocked": False}
        {"made": 1.0, "missed": 0.0, "blocked": 0.0}
    )
    #
    kicker_one_hot = pd.get_dummies(data["kicker_player_id"])
    data_encoded = pd.concat([data, kicker_one_hot], axis=1)
    train_params = ["kick_distance"] + kicker_one_hot.keys().tolist()
    data_test, data_train = split_test_train(data_encoded)
    x_data_test = data_test[train_params]
    y_data_test = data_test["fg_make"]
    x_data_train = data_train[train_params]
    y_data_train = data_train["fg_make"]
    # Regularization must be turned off to get a good fit because "yards" is on a different scale
    # Turn off the intercept because we are fitting a separate intercept for each kicker
    reg = LogisticRegression(
        penalty="none", fit_intercept=False, n_jobs=4, max_iter=500,
    )
    fit = reg.fit(x_data_train, y_data_train)

    # don't add_constant() for this x-data
    model = sm.GLM(y_data_train, x_data_train, family=sm.families.Binomial())
    sm_fit = model.fit()
    print(sm_fit.summary2())

    # print(f"coefficients = {fit.coef_}")
    print(f"intercept = {fit.intercept_}")
    # print(f"classes = {fit.classes_}")

    score: float = fit.score(x_data_test, y_data_test)
    print(f"score = {score}")
    loss = get_loss(fit, x_data_test, y_data_test)
    print(f"loss = {loss}")

    kicker_params: List[float] = fit.coef_[0][1:]
    kicker_ids: List[str] = train_params[1:]
    kicker_data: pd.DataFrame = pd.DataFrame(
        data={"score": kicker_params, "id": kicker_ids}
    )

    return fit, kicker_data


def _drop_null(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with exclusively null data
    """
    # Data doesn't need to be null to be dropable.
    # null_cols = [col for col in data.keys() if data[col].isnull().all()]
    # This removes redundant data
    noninfo_cols = [col for col in data.keys() if len(data[col].unique()) == 1]
    for col in noninfo_cols:
        print(col, data[col].unique())
    return data.drop(columns=noninfo_cols)


def _get_data(columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    get and cache test data
    """
    cache_name = "field_goals.parquet"
    try:
        play_data: pd.DataFrame = pd.read_parquet(cache_name)
        return play_data
    except Exception:
        print("Could not find cache")
    if columns and (
        "game_id" not in columns
        or "play_id" not in columns
        or "play_type" not in columns
    ):
        raise RuntimeError("Require game_id and play_id and play_type")
    play_data = plays(
        range(2011, 2019 + 1), [("play_type", "field_goal")], columns=columns
    )
    play_data.set_index(["game_id", "play_id"], inplace=True)
    play_data = _drop_null(play_data)
    play_data.to_parquet(cache_name)
    return play_data


def plot_fg_prob(
    fg_attempts: pd.DataFrame, model: LogisticRegression, label: str = ""
) -> None:
    """
    Make a plot of the fg prob
    """
    label = label or "fg"
    yard_bins = np.arange(15, 65, 1)
    # fg_attempts['dist_bin'] = pd.cut(x=fg_attempts['kick_distance'], bins=yard_bins)
    data_bins = pd.cut(x=fg_attempts["kick_distance"], bins=yard_bins)
    x, y = [], []
    yerr = []
    for yd_bin in data_bins.unique().sort_values():
        bin_data = fg_attempts[data_bins == yd_bin]
        # With a single data point there is no standard error
        if len(bin_data) <= 1:
            continue
        x.append(bin_data["kick_distance"].mean())
        yvals = bin_data["fg_make"]
        yavg = yvals.mean()
        y.append(yavg)
        yerr.append(np.sqrt(yavg * (1 - yavg) / (len(bin_data) - 1)))
    xfit = np.arange(15, 70, 0.1)
    yfit = model.predict_proba(xfit.reshape(-1, 1)).T[1]
    # xfit2 = np.stack([xfit, xfit**2], axis=1)
    # yfit = dist_reg.predict_proba(xfit2).T[1]

    # plt.plot(x, y, label='data')
    plt.errorbar(x, y, yerr=yerr, label=f"{label} data")
    plt.plot(xfit, yfit, label=f"{label} fit")
    plt.legend()
    plt.xlabel("field goal distance (yards)")
    plt.ylabel("success probability")
    # plt.show()


def main():
    keep_columns: List[str] = [
        "game_id",
        "play_id",
        "play_type",
        "desc",
        "field_goal_result",
        "kick_distance",
        "kicker_player_name",
        "kicker_player_id",
        "season",
        "weather",
        "roof",
        "surface",
        "temp",
        "wind",
    ]

    fg_attempts = _get_data(columns=keep_columns)
    # print(fg_attempts)

    # Remove kickers with few attempts
    attempt_ids = fg_attempts["kicker_player_id"]
    id_counts = attempt_ids.groupby(attempt_ids).count()
    print(id_counts)
    good_id_counts = id_counts[id_counts >= 20]
    print(good_id_counts)
    print(len(fg_attempts))
    fg_attempts = fg_attempts[
        fg_attempts["kicker_player_id"].isin(good_id_counts.index)
    ]
    print(len(fg_attempts))

    # test_att = fg_attempts.iloc[0]
    # for key, val in test_att.items():
    #     print(key, val)

    inc_model = basic_regression(fg_attempts)

    player_model, kicker_eval = player_regression(fg_attempts)
    kicker_eval["name"] = kicker_eval["id"].apply(
        lambda d: get_player_name(fg_attempts, d)
    )
    kicker_eval = kicker_eval.sort_values(by='score', ascending=False).reset_index(drop=True)
    print(kicker_eval)

    print(fg_attempts["roof"].unique())

    exit(0)
    print(fg_attempts["kicker_player_name"].unique())
    jt_id: str = "32013030-2d30-3032-3935-39371b9a6ac1"
    jt_data = fg_attempts[fg_attempts["kicker_player_id"] == jt_id].copy()
    jt_model = basic_regression(jt_data)
    print(jt_model.intercept_, jt_model.coef_)

    # Plot
    fig = plt.figure()
    plot_fg_prob(fg_attempts, inc_model)
    plot_fg_prob(jt_data, jt_model, label="J.Tucker")
    fig.savefig(f"fg.png")


def get_player_name(source: pd.DataFrame, player_id: str) -> str:
    """
    Convert ID to name using source as a reference
    """
    player_names = source[source["kicker_player_id"] == player_id]["kicker_player_name"]
    unique_names: NDArray[str] = player_names.unique()
    if len(unique_names) != 1:
        print(
            f"Warning: Incorrect number of unique names for {player_id}: {unique_names}"
        )
    return unique_names[0]


if __name__ == "__main__":
    main()
