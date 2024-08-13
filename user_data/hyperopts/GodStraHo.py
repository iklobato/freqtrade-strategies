# GodStra Strategy Hyperopt
# Author: @Mablue (Masoud Azizi)
# github: https://github.com/mablue/
# IMPORTANT: INSTALL TA BEFOUR RUN:
# :~$ pip install ta
# freqtrade hyperopt --hyperopt GodStraHo --hyperopt-loss SharpeHyperOptLossDaily --spaces all --strategy GodStra --config config.json -e 100

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
# import talib.abstract as ta  # noqa
from ta import add_all_ta_features
from ta.utils import dropna
import freqtrade.vendor.qtpylib.indicators as qtpylib

# This is your trading strategy DNA Size
# You can change it and see the results...
DNA_SIZE = 1

GodGenes = ["open", "high", "low", "close", "volume", "volume_adi", "volume_obv",
            "volume_cmf", "volume_fi", "volume_mfi", "volume_em", "volume_sma_em", "volume_vpt",
            "volume_nvi", "volume_vwap", "volatility_atr", "volatility_bbm", "volatility_bbh",
            "volatility_bbl", "volatility_bbw", "volatility_bbp", "volatility_bbhi",
            "volatility_bbli", "volatility_kcc", "volatility_kch", "volatility_kcl",
            "volatility_kcw", "volatility_kcp", "volatility_kchi", "volatility_kcli",
            "volatility_dcl", "volatility_dch", "volatility_dcm", "volatility_dcw",
            "volatility_dcp", "volatility_ui", "trend_macd", "trend_macd_signal",
            "trend_macd_diff", "trend_sma_fast", "trend_sma_slow", "trend_ema_fast",
            "trend_ema_slow", "trend_adx", "trend_adx_pos", "trend_adx_neg", "trend_vortex_ind_pos",
            "trend_vortex_ind_neg", "trend_vortex_ind_diff", "trend_trix",
            "trend_mass_index", "trend_cci", "trend_dpo", "trend_kst",
            "trend_kst_sig", "trend_kst_diff", "trend_ichimoku_conv",
            "trend_ichimoku_base", "trend_ichimoku_a", "trend_ichimoku_b",
            "trend_visual_ichimoku_a", "trend_visual_ichimoku_b", "trend_aroon_up",
            "trend_aroon_down", "trend_aroon_ind", "trend_psar_up", "trend_psar_down",
            "trend_psar_up_indicator", "trend_psar_down_indicator", "trend_stc",
            "momentum_rsi", "momentum_stoch_rsi", "momentum_stoch_rsi_k",
            "momentum_stoch_rsi_d", "momentum_tsi", "momentum_uo", "momentum_stoch",
            "momentum_stoch_signal", "momentum_wr", "momentum_ao", "momentum_kama",
            "momentum_roc", "momentum_ppo", "momentum_ppo_signal", "momentum_ppo_hist",
            "others_dr", "others_dlr", "others_cr"]

class GodStraHo(IHyperOpt):

    @staticmethod
    def indicator_space(prefix: str) -> List[Dimension]:
        """
        Define your Hyperopt space for searching strategy parameters.
        """
        gene = []
        for i in range(DNA_SIZE):
            gene.append(Categorical(GodGenes, name=f'{prefix}-indicator-{i}'))
            gene.append(Categorical(GodGenes, name=f'{prefix}-cross-{i}'))
            gene.append(Integer(-1, 101, name=f'{prefix}-int-{i}'))
            gene.append(Real(-1.1, 1.1, name=f'{prefix}-real-{i}'))
            gene.append(Categorical(["D", ">", "<", "=", "CA", "CB", ">I", "=I", "<I", ">R", "=R", "<R"],
                                    name=f'{prefix}-oper-{i}'))
        return gene

    @staticmethod
    def build_conditions(params: Dict[str, Any], dataframe: DataFrame, prefix: str) -> List:
        """
        Build conditions for strategy generation using operation mapping.
        """
        operations_map = {
            ">": lambda df_ind, df_crs, int_val, real_val: df_ind > df_crs,
            "=": lambda df_ind, df_crs, int_val, real_val: np.isclose(df_ind, df_crs),
            "<": lambda df_ind, df_crs, int_val, real_val: df_ind < df_crs,
            "CA": lambda df_ind, df_crs, int_val, real_val: qtpylib.crossed_above(df_ind, df_crs),
            "CB": lambda df_ind, df_crs, int_val, real_val: qtpylib.crossed_below(df_ind, df_crs),
            ">I": lambda df_ind, df_crs, int_val, real_val: df_ind > int_val,
            "=I": lambda df_ind, df_crs, int_val, real_val: df_ind == int_val,
            "<I": lambda df_ind, df_crs, int_val, real_val: df_ind < int_val,
            ">R": lambda df_ind, df_crs, int_val, real_val: df_ind > real_val,
            "=R": lambda df_ind, df_crs, int_val, real_val: np.isclose(df_ind, real_val),
            "<R": lambda df_ind, df_crs, int_val, real_val: df_ind < real_val,
        }

        conditions = []
        for i in range(DNA_SIZE):
            OPR = params[f'{prefix}-oper-{i}']
            IND = params[f'{prefix}-indicator-{i}']
            CRS = params[f'{prefix}-cross-{i}']
            INT = params[f'{prefix}-int-{i}']
            REAL = params[f'{prefix}-real-{i}']

            condition = operations_map[OPR](dataframe[IND], dataframe[CRS], INT, REAL)
            conditions.append(condition)

        return conditions

    @staticmethod
    def strategy_generator(params: Dict[str, Any], prefix: str, signal: str) -> Callable:
        """
        Define the strategy parameters to be used by Hyperopt.
        """
        def populate_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Strategy Hyperopt will build and use.
            """
            conditions = GodStraHo.build_conditions(params, dataframe, prefix)
            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), signal] = 1
            return dataframe

        return populate_trend

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        return GodStraHo.strategy_generator(params, prefix='buy', signal='enter_long')

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        return GodStraHo.strategy_generator(params, prefix='sell', signal='exit_long')

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return GodStraHo.indicator_space(prefix='buy')

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return GodStraHo.indicator_space(prefix='sell')
