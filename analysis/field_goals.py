#!/usr/bin/env python3

from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from db import plays
from nptyping import NDArray
from sklearn.linear_model import LogisticRegression
from statsmodels.genmod.generalized_linear_model import GLMResults

# from utility import split_test_train

# TODO:
# Control for roof type (outdoors, dome, etc.), possibly weather. Some data is missing.


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
    loss_terms: NDArray[float] = -y_data_test * test_log_prob.T[1] - (
        1.0 - y_data_test
    ) * test_log_prob.T[0]
    loss: float = np.sum(loss_terms) / len(y_data_test)
    return loss


def basic_regression(data: pd.DataFrame) -> GLMResults:
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
    # data_test, data_train = split_test_train(data)
    # x_data_test = data_test[train_params]
    # y_data_test = data_test["fg_make"]
    # x_data_train = data_train[train_params]
    # y_data_train = data_train["fg_make"]
    x_data_train = data[train_params]
    y_data_train = data["fg_make"]

    # We need to add a constant for the intercept term
    sm_model = sm.GLM(
        y_data_train, sm.add_constant(x_data_train), sm.families.Binomial()
    )
    sm_fit = sm_model.fit()
    sm_df_resid = sm_fit.df_resid
    print(sm_df_resid)
    sm_loss = sm_fit.llf / sm_df_resid
    sm_deviance = sm_fit.deviance / sm_df_resid
    sm_chi_sq_df = sm_fit.pearson_chi2 / sm_df_resid
    print(f"ll / ndf = {sm_loss}")
    print(f"deviance = {sm_deviance}")
    print(f"chi sq / ndf = {sm_chi_sq_df}")

    # print(sm_model_fit.summary2())
    print(sm_fit.params)
    print(sm_fit.normalized_cov_params)
    # defaults to 95% confidence interval (0.05 argument is alpha)
    print(sm_fit.conf_int(0.1))
    # standard error approximation (95% ~ 2*sigma, double-sided)
    print(0.25 * (sm_fit.conf_int()[1] - sm_fit.conf_int()[0]))
    # print(dir(sm_model_fit))

    print(f"params = {sm_fit.params}")

    return sm_fit


def player_regression_range(data: pd.DataFrame) -> pd.DataFrame:
    """
    Regression of success probability treating each player's whole career separately
    """
    data["fg_make"] = data["field_goal_result"].map(
        {"made": 1.0, "missed": 0.0, "blocked": 0.0}
    )

    kicker_ids: List[str] = data["kicker_player_id"].unique().tolist()
    kicker_results: List[Dict[str, Any]] = []
    for kicker_id in kicker_ids:
        kicker_attempts = data[data["kicker_player_id"] == kicker_id]
        kicker_model = sm.GLM(
            kicker_attempts["fg_make"],
            sm.add_constant(kicker_attempts["kick_distance"]),
            family=sm.families.Binomial(),
        )
        kicker_fit = kicker_model.fit()
        params = kicker_fit.params
        # 95% CI by default: use 1 sigma
        conf_int = kicker_fit.conf_int(0.3173)
        kicker_results.append(
            {
                "id": kicker_id,
                "name": get_player_name(kicker_attempts, kicker_id),
                "const": kicker_fit.params["const"],
                "kick_distance": params["kick_distance"],
                "const_err_low": params["const"] - conf_int.loc["const", 0],
                "const_err_up": conf_int.loc["const", 1] - params["const"],
                "kick_distance_err_low": params["kick_distance"]
                - conf_int.loc["kick_distance", 0],
                "kick_distance_err_up": conf_int.loc["kick_distance", 1]
                - params["kick_distance"],
            }
        )
    kicker_data = pd.DataFrame(kicker_results)
    return kicker_data


def player_regression(data: pd.DataFrame) -> Tuple[GLMResults, pd.DataFrame]:
    """
    Regression of success probability treating each player's whole career separately,
    using the same model for distance effect for everyone.
    """
    # TODO: treat blocked FGs separately
    data["fg_make"] = data["field_goal_result"].map(
        # {"made": True, "missed": False, "blocked": False}
        {"made": 1.0, "missed": 0.0, "blocked": 0.0}
    )
    kicker_one_hot = pd.get_dummies(data["kicker_player_id"])
    data_encoded = pd.concat([data, kicker_one_hot], axis=1)
    # TODO: use a GAM and spline the kick_distance variable. This will re-necessitate
    # training data to find the best alpha penalties.
    train_params = ["kick_distance"] + kicker_one_hot.keys().tolist()
    # data_test, data_train = split_test_train(data_encoded)
    # x_data_test = data_test[train_params]
    # y_data_test = data_test["fg_make"]
    # x_data_train = data_train[train_params]
    # y_data_train = data_train["fg_make"]
    x_data_train = data_encoded[train_params]
    y_data_train = data_encoded["fg_make"]
    # don't add_constant() for this x-data
    model = sm.GLM(y_data_train, x_data_train, family=sm.families.Binomial())
    sm_fit = model.fit()
    # print(sm_fit.summary2())

    sm_df_resid = sm_fit.df_resid
    print(sm_df_resid)
    sm_loss = sm_fit.llf / sm_df_resid
    sm_deviance = sm_fit.deviance / sm_df_resid
    sm_chi_sq_df = sm_fit.pearson_chi2 / sm_df_resid

    print(f"ll / ndf = {sm_loss}")
    print(f"deviance = {sm_deviance}")
    print(f"chi sq / ndf = {sm_chi_sq_df}")
    # print(sm_fit.params.drop("kick_distance"))

    kicker_params: List[float] = sm_fit.params.drop("kick_distance")
    kicker_ids: List[str] = train_params[1:]
    kicker_data: pd.DataFrame = pd.DataFrame(
        data={"score": kicker_params, "id": kicker_ids}
    )

    return sm_fit, kicker_data


def player_regression_season(data: pd.DataFrame) -> Tuple[GLMResults, pd.DataFrame]:
    """
    Regression of success probability where the distance penalty is the same for
    everyone (TODO: should be splined) and each season is treated independently.
    """
    # TODO: treat blocked FGs separately
    data["fg_make"] = data["field_goal_result"].map(
        # {"made": True, "missed": False, "blocked": False}
        {"made": 1.0, "missed": 0.0, "blocked": 0.0}
    )
    # kicker_one_hot = pd.get_dummies(data[["kicker_player_id", "season"]])
    kicker_season_labels = data["kicker_player_id"] + "_" + data["season"].apply(str)
    kicker_one_hot = pd.get_dummies(kicker_season_labels)
    data_encoded = pd.concat([data, kicker_one_hot], axis=1)
    # TODO: use a GAM and spline the kick_distance variable. This will re-necessitate
    # training data to find the best alpha penalties.
    train_params = ["kick_distance"] + kicker_one_hot.keys().tolist()
    # data_test, data_train = split_test_train(data_encoded)
    # x_data_test = data_test[train_params]
    # y_data_test = data_test["fg_make"]
    # x_data_train = data_train[train_params]
    # y_data_train = data_train["fg_make"]
    x_data_train = data_encoded[train_params]
    y_data_train = data_encoded["fg_make"]
    # don't add_constant() for this x-data
    model = sm.GLM(y_data_train, x_data_train, family=sm.families.Binomial())
    fit = model.fit()
    # print(sm_fit.summary2())

    sm_df_resid = fit.df_resid
    print(sm_df_resid)
    sm_loss = fit.llf / sm_df_resid
    sm_deviance = fit.deviance / sm_df_resid
    sm_chi_sq_df = fit.pearson_chi2 / sm_df_resid
    print(f"ll / ndf = {sm_loss}")
    print(f"deviance = {sm_deviance}")
    print(f"chi sq / ndf = {sm_chi_sq_df}")
    # print(sm_fit.params.drop("kick_distance"))

    kicker_year_params: pd.Series = fit.params.drop("kick_distance")
    season_data: List[Dict[str, Any]] = []
    for id_year, accuracy in kicker_year_params.items():
        kicker_id, year = id_year.split("_")
        conf_int = fit.conf_int(0.3173)
        season_data.append(
            {
                "id": kicker_id,
                "season": year,
                "accuracy": accuracy,
                "accuracy_err_low": accuracy - conf_int.loc[id_year, 0],
                "accuracy_err_up": conf_int.loc[id_year, 1] - accuracy,
            }
        )
    kicker_data: pd.DataFrame = pd.DataFrame(season_data)

    return fit, kicker_data


def _drop_null(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with exclusively null data
    """
    # This removes null and single-valued data
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
        # The player IDs are a different format pre-2011
        # range(1999, 2019 + 1),
        range(2011, 2019 + 1),
        [("play_type", "field_goal")],
        columns=columns,
    )
    play_data.set_index(["game_id", "play_id"], inplace=True)
    play_data = _drop_null(play_data)
    play_data.to_parquet(cache_name)
    return play_data


def plot_fg_prob(fg_attempts: pd.DataFrame, model: GLMResults, label: str = "") -> None:
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
    yfit = model.predict(sm.add_constant(xfit.reshape(-1, 1)))
    # yfit = model.predict_proba(xfit.reshape(-1, 1)).T[1]
    # xfit2 = np.stack([xfit, xfit**2], axis=1)
    # yfit = dist_reg.predict_proba(xfit2).T[1]

    # plt.plot(x, y, label='data')
    plt.errorbar(x, y, yerr=yerr, label=f"{label} data")
    plt.plot(xfit, yfit, label=f"{label} fit")
    plt.legend()
    plt.xlabel("field goal distance (yards)")
    plt.ylabel("success probability")
    # plt.show()


def plot_kicker_career(kicker_data: pd.DataFrame) -> None:
    """
    Make scatterplot of individual kicker career parameters
    """
    # NOTE: There is a strong anti-correlation between accuracy and "power". Perhaps a
    # generalized additive model would describe the distance dependence better.
    xvals = kicker_data["kick_distance"].to_numpy()
    yvals = kicker_data["const"].to_numpy()
    print(len(xvals))
    xerrs_low = kicker_data["kick_distance_err_low"].to_numpy()
    xerrs_up = kicker_data["kick_distance_err_up"].to_numpy()
    yerrs_low = kicker_data["const_err_low"].to_numpy()
    yerrs_up = kicker_data["const_err_up"].to_numpy()
    xerrs = np.stack([xerrs_low, xerrs_up], axis=0)
    yerrs = np.stack([yerrs_low, yerrs_up], axis=0)
    plt.errorbar(xvals, yvals, yerrs, xerrs, linestyle="none")
    plt.xlabel("distance (yards^{-1})")
    plt.ylabel("accuracy")


def plot_kicker_seasons(kicker_data: pd.DataFrame) -> None:
    """
    Plot trajectory of several kicker's seasons via single-value accuracy parameter
    """
    ids = kicker_data["id"].unique().tolist()
    for kicker_id in ids:
        kicker_season_data: pd.DataFrame = kicker_data[
            kicker_data["id"] == kicker_id
        ].sort_values("season")
        # if len(kicker_season_data) < 4:
        #     continue
        if (
            kicker_season_data["accuracy_err_low"]
            + kicker_season_data["accuracy_err_up"]
            > 10
        ).any():
            continue
        kicker_name = kicker_season_data["name"].unique()[0]
        xvals = kicker_season_data["season"].to_numpy()
        yvals = kicker_season_data["accuracy"].to_numpy()
        plt.plot(xvals, yvals, label=kicker_name)
        # print(kicker_season_data)
    plt.legend()
    plt.xlabel("season")
    plt.ylabel("accuracy")


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
    # print(id_counts)
    good_id_counts = id_counts[id_counts >= 50]
    # print(good_id_counts)
    # print(len(fg_attempts))
    fg_attempts = fg_attempts[
        fg_attempts["kicker_player_id"].isin(good_id_counts.index)
    ]
    # print(len(fg_attempts))

    # test_att = fg_attempts.iloc[0]
    # for key, val in test_att.items():
    #     print(key, val)

    inc_model = basic_regression(fg_attempts)

    player_model, kicker_eval = player_regression(fg_attempts)
    kicker_eval["name"] = kicker_eval["id"].apply(
        lambda d: get_player_name(fg_attempts, d)
    )
    kicker_eval = kicker_eval.sort_values(by="score", ascending=False).reset_index(
        drop=True
    )
    # print(kicker_eval)

    # print(fg_attempts["roof"].unique())

    kicker_ind_range = player_regression_range(fg_attempts)
    # print(kicker_ind_range[["name", "const", "kick_distance"]])

    _season_model, kicker_season_data = player_regression_season(fg_attempts)
    kicker_season_data["name"] = kicker_season_data["id"].apply(
        lambda d: get_player_name(fg_attempts, d)
    )
    print(kicker_season_data)

    # print(fg_attempts["kicker_player_name"].unique())
    # jt_id: str = "32013030-2d30-3032-3935-39371b9a6ac1"
    # jt_data = fg_attempts[fg_attempts["kicker_player_id"] == jt_id].copy()
    # jt_model = basic_regression(jt_data)
    # print(jt_model.intercept_, jt_model.coef_)

    # Plot
    fig = plt.figure()
    plot_fg_prob(fg_attempts, inc_model)
    # plot_fg_prob(jt_data, jt_model, label="J.Tucker")
    fig.savefig(f"fg.png")

    fig = plt.figure()
    plot_kicker_career(kicker_ind_range)
    fig.savefig(f"kickers.png")

    fig = plt.figure()
    plot_kicker_seasons(kicker_season_data)
    fig.savefig("kicker_seasons.png")


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
    player_name: str = unique_names[0]
    return player_name


if __name__ == "__main__":
    main()
