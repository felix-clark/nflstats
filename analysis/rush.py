#!/usr/bin/env python3
from typing import Any, Callable, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm
from db import plays
from nptyping import NDArray
from statsmodels.gam.api import BSplines
from statsmodels.genmod.generalized_linear_model import GLMResults
from utility import drop_noinfo


def exp_inv(x, k, A, B) -> float:
    val: float = A * np.exp(-k * x) + B / x
    return val


def grad_exp_inv(x, k, A, B) -> NDArray[float]:
    exp_term = np.exp(-k * x)
    return np.array([-A * x * exp_term, exp_term, 1.0 / x,])


def _get_data(columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    get and cache test data
    """
    cache_name = "running_plays.parquet"
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
        queries=[("play_type", "run")],
        columns=columns,
    )
    play_data.set_index(["game_id", "play_id"], inplace=True)
    play_data = drop_noinfo(play_data)
    play_data.sort_values(["game_id", "play_id"], inplace=True)
    play_data.to_parquet(cache_name)
    return play_data


def td_prob(rush_att: pd.DataFrame) -> Tuple[GLMResults, BSplines]:
    """
    TD probability as a function of distance to goal
    """
    rush_att["inv_yards"] = 1 / rush_att["yardline_100"]
    train_params = ["inv_yards"]
    # train_params = ["yardline_100"]
    y_data_train = rush_att["rush_touchdown"]
    x_data_train = rush_att[train_params]
    # TODO: These should be checked by X-val; this situation is more complicated than FGs
    degree, df, alpha = 3, 4, 0.0
    bs = BSplines(x_data_train, df=[df], degree=[degree])
    model = sm.GLMGam(
        y_data_train,
        sm.add_constant(x_data_train[[]]),
        smoother=bs,
        alpha=alpha,
        family=sm.families.Binomial(),
    )
    fit = model.fit()
    print(fit.summary2())
    return fit, bs


def td_prob_exp_inv(rush_attempts: pd.DataFrame) -> opt.OptimizeResult:
    """
    Fit the TD probabilty to A*exp(-kx) + B/x where x is yards to goal
    """
    yard_bins = np.concatenate(
        [np.arange(0, 40, 2), np.arange(40, 60, 5), np.arange(60, 100, 10),]
    )
    yvals = rush_attempts["rush_touchdown"].to_numpy()
    xvals = rush_attempts["yardline_100"].to_numpy()
    k = 0.17621065

    def like(y, x, A, B):
        # def like(y, x, k, A, B):
        like_terms = y * np.log(exp_inv(x, k, A, B)) + (1 - y) * np.log(
            1.0 - exp_inv(x, k, A, B)
        )
        return np.sum(like_terms)

    def func(p):
        return -2.0 * like(yvals, xvals, *p)

    def grad(p):
        # probs = exp_inv(xvals, *p)
        probs = exp_inv(xvals, k, *p)
        # grad_probs = grad_exp_inv(xvals, *p)
        grad_probs = grad_exp_inv(xvals, k, *p)[1:]
        grad_terms = grad_probs.dot(yvals / (probs) - (1 - yvals) / (1 - probs))
        return -2.0 * grad_terms

    # x0 = np.array([0.18, 0.4, 0.2])
    x0 = np.array([0.4, 0.2])
    res = opt.minimize(
        func,
        x0=x0,
        jac=grad,
        # bounds=[(0.05, 0.5), (0.01, 0.9), (0.01, 0.6)],
        bounds=[(0.01, 0.9), (0.01, 0.6)],
    )
    if not res.success:
        raise RuntimeError("Did not converge")
    hess_inv = res.hess_inv
    # k_hess_inv = hess_inv.dot(np.array([1.0, 0.0, 0.0,])).dot(np.array([1.0, 0.0, 0.0]))
    A_hess_inv = hess_inv.dot(np.array([1.0, 0.0,])).dot(np.array([1.0, 0.0]))
    B_hess_inv = hess_inv.dot(np.array([0.0, 1.0,])).dot(np.array([0.0, 1.0]))
    A_err = np.sqrt(2.0 * A_hess_inv)
    B_err = np.sqrt(2.0 * B_hess_inv)
    print(res.x)
    errs = np.array([A_err, B_err])
    print(errs)
    return res.x, errs


def plot_td_prob(
    rush_attempts: pd.DataFrame,
    # model: GLMResults,
    # transform: Optional[Callable[[NDArray[float]], NDArray[float]]] = None,
    pars: Tuple[float, float, float],
    **kwargs: Any,
) -> None:
    """
    Make a plot of the fg prob
    """
    label = kwargs.get("label", "td")
    # dep_par = kwargs.get('dep_par', 'yardline_100')
    # yard_bins = np.arange(0, 100, 1)
    yard_bins = np.concatenate(
        # [np.arange(0, 10, 2), np.arange(10, 40, 5), np.arange(40, 100, 20)]
        [np.arange(0, 10, 5), np.arange(10, 40, 10), np.arange(40, 100, 20)]
    )
    # yard_bins = np.arange(0.01, 1.01, 0.01)
    # fg_attempts['dist_bin'] = pd.cut(x=fg_attempts['kick_distance'], bins=yard_bins)
    data_bins = pd.cut(x=rush_attempts["yardline_100"], bins=yard_bins)
    x, y = [], []
    yerr = []
    for yd_bin in data_bins.unique().sort_values():
        bin_data = rush_attempts[data_bins == yd_bin]
        # With a single data point there is no standard error
        if len(bin_data) <= 1:
            continue
        x.append(bin_data["yardline_100"].mean())
        yvals = bin_data["rush_touchdown"]
        yavg = yvals.mean()
        y.append(yavg)
        yerr.append(np.sqrt(yavg * (1 - yavg) / (len(bin_data) - 1)))
    xfit = np.arange(min(x), max(x), 0.01)
    # xshape = xfit.reshape(-1, 1)
    # if transform is not None:
    #     xshape = transform(xshape)
    # yfit = model.predict(sm.add_constant(xshape))
    # popt, _pcov = opt.curve_fit(
    #     exp_inv, x, y, sigma=yerr, p0=[0.2, 0.4, 0.2], absolute_sigma=True
    # )
    # print(popt)
    # print(np.sqrt(np.diag(_pcov)))

    plt.errorbar(x, y, yerr=yerr, label=f"{label} data")
    # plt.plot(xfit, yfit, label=f"{label} fit")
    plt.plot(xfit, exp_inv(xfit, *pars), label=f"{label} fit")
    plt.yscale("log")
    # plt.xscale("log")
    plt.legend()
    plt.xlabel("yards to goal")
    plt.ylabel("touchdown probability")


def main():
    """
    Execute main analysis
    """
    # There may be more; this is just a first pass of what might be useful
    columns = [
        "posteam",
        "posteam_type",
        "defteam",
        "yardline_100",
        "sp",
        "down",
        "goal_to_go",
        "ydstogo",
        "ydsnet",  # unclear; net yards on drive?
        "desc",
        "yards_gained",
        "shotgun",
        "no_huddle",
        "qb_dropback",
        "qb_scramble",
        "run_location",
        "run_gap",
        "two_point_conv_result",
        "timeout",
        "ep",
        "epa",
        # There are several like these; can also use win probability [added] (wp/wpa)
        # 'total_home_epa',
        # 'total_away_epa',
        # 'total_home_rush_epa',
        # 'total_away_rush_epa',
        "wp",
        "def_wp",
        "wpa",
        "fumble_forced",
        "fumble_not_forced",
        "fumble_out_of_bounds",
        "safety",
        "penalty",
        "tackled_for_loss",
        "fumble_lost",
        # These are not identical; a fumble recovery for a touchdown will count as
        # "touchdown" but not "rush_touchdown".
        "touchdown",
        "rush_touchdown",
        "two_point_attempt",
        "fumble",
        "drive_play_count",
        "drive_time_of_possession",
        "drive_first_downs",
        "passer",
        "rusher",
        "receiver",
        # about 5% of these plays have a pass?
        # Running plays marked as "pass" are typically QB scrambles
        "pass",
        "rush",
        "first_down",
        "passer_id",
        "rusher_id",
        "receiver_id",
    ]
    columns = None

    rush_attempts = _get_data(columns)
    print(rush_attempts)
    for col in rush_attempts.columns:
        print(col)
        # print(rush_attempts[col].describe())

    print(rush_attempts["pass"].unique())
    # weird_plays = rush_attempts[rush_attempts['rush_touchdown'] != rush_attempts['touchdown']]
    # weird_plays = rush_attempts[rush_attempts['pass'] == 1]
    # for _, play in weird_plays.iterrows():
    # print(play[['touchdown', 'rush_touchdown']])
    # print(play['desc'])

    rush_attempts = rush_attempts[
        (rush_attempts["penalty"] == 0) & (rush_attempts["pass"] == 0)
    ]

    td_fit, bs = td_prob(rush_attempts)
    # fig = td_fit.plot_added_variable('yardline_100')
    # fig = td_fit.plot_ceres_residuals('yardline_100')
    # fig.savefig('td_prob.png')

    gurley_id = rush_attempts[rush_attempts["rusher"] == "T.Gurley"][
        "rusher_id"
    ].unique()[0]
    zeke_id = rush_attempts[rush_attempts["rusher"] == "E.Elliott"][
        "rusher_id"
    ].unique()[0]
    bell_id = rush_attempts[rush_attempts["rusher"] == "L.Bell"]["rusher_id"].unique()[
        0
    ]
    print(gurley_id, zeke_id, bell_id)

    fig = plt.figure()
    # plot_td_prob(rush_attempts, td_fit, transform=lambda x: bs.transform(1.0 / x))
    inc_pars, _ = td_prob_exp_inv(rush_attempts)
    # This value is from inclusive and is consistent with individuals within statistical error
    k = 0.17621065
    inc_pars = np.concatenate([[k], inc_pars])
    plot_td_prob(rush_attempts, inc_pars)
    gurley_rush_att = rush_attempts[rush_attempts["rusher_id"] == gurley_id]
    print(len(gurley_rush_att))
    zeke_rush_att = rush_attempts[rush_attempts["rusher_id"] == zeke_id]
    print(len(zeke_rush_att))
    bell_rush_att = rush_attempts[rush_attempts["rusher_id"] == bell_id]
    print(len(bell_rush_att))
    gurley_pars, _ = td_prob_exp_inv(gurley_rush_att)
    gurley_pars = np.concatenate([[k], gurley_pars])
    zeke_pars, _ = td_prob_exp_inv(zeke_rush_att)
    zeke_pars = np.concatenate([[k], zeke_pars])
    bell_pars, _ = td_prob_exp_inv(bell_rush_att)
    bell_pars = np.concatenate([[k], bell_pars])
    plot_td_prob(gurley_rush_att, gurley_pars, label="Gurley")
    plot_td_prob(zeke_rush_att, zeke_pars, label="Zeke")
    plot_td_prob(bell_rush_att, bell_pars, label="Bell")
    fig.savefig("td_prob.png")


if __name__ == "__main__":
    main()
