import pandas as pd
import numpy as np
import dateutil
from pmdarima.utils import diff_inv


class Time_Series_Retransformer:
    def __init__(self, df_orig: pd.Series, df_diff: pd.Series):        
        self.df_orig = df_orig.reset_index().drop_duplicates().set_index("date").squeeze()
        self.df_diff = df_diff.reset_index().drop_duplicates().set_index("date").squeeze()

    def __inv_diff1(
        self,
        df_orig_column: pd.Series,
        df_diff_column: pd.Series,
        lagNum: int = 1,
        exclude_orig: bool = True,
    ) -> pd.Series:
        """
        Method for the absolute change inversion

        Parameters
        ----------
        df_orig_column: original series indexed in time order
        df_diff_column: differenced series indexed in time order
        lagNum : an integer > 0 indicating which lag to use.
        exclude_orig: flag indicating whether the 1st value of the
            original data should be included in the result.


        Returns
        -------
        pd.Series: the result of the inverse of the difference arrays.

        """

        # Generate np.array for the diff_inv function - it includes first n values(n =
        # periods) of original data & further diff values of given periods
        _index = [df_diff_column.index.min()]

        for _periods in range(lagNum):
            _index.append(_index[-1] - dateutil.relativedelta.relativedelta(months=1))

        value = np.array(
            df_orig_column.loc[_index[-lagNum:]].sort_index().tolist()
            + df_diff_column.sort_index().tolist()
        )

        if exclude_orig:
            # Generate pd.Series with inverse diff
            inv_diff_vals = diff_inv(value, lagNum, 1)[(lagNum + 1) :]
            inv_diff_series = pd.Series(inv_diff_vals, df_diff_column.index)
            
            # assert (inv_diff_series == df_orig_column.loc[df_diff_column.index]).all()
        else:
            # Include the 1st (original) observation
            inv_diff_vals = diff_inv(value, lagNum, 1)[lagNum:]
            inv_diff_idx = [
                df_diff_column.index.min()
                - dateutil.relativedelta.relativedelta(months=1)
            ] + df_diff_column.index.to_list()

            inv_diff_series = pd.Series(inv_diff_vals, inv_diff_idx)

        return inv_diff_series

    def __inv_diff2(
        self, df_orig_column: pd.Series, df_diff_column: pd.Series, exclude_orig=True
    ):
        """
        Method for the 2nd order change inversion

        Parameters
        ----------
        df_orig_column: original series indexed in time order
        df_diff_column: differenced series indexed in time order
        exclude_orig: flag indicating whether the 1st value of the
            original data should be included in the result.

        Returns
        -------
        pd.Series: the result of the inverse of the difference arrays.

        """
        _df_diff_column = df_orig_column.diff(periods=1).dropna()
        _inv_diff_series = self.__inv_diff1(
            df_orig_column=_df_diff_column,
            df_diff_column=df_diff_column,
            lagNum=1,
            exclude_orig=False,
        )

        inv_diff_series = self.__inv_diff1(
            df_orig_column=df_orig_column,
            df_diff_column=_inv_diff_series,
            lagNum=1,
            exclude_orig=exclude_orig,
        )

        return inv_diff_series
    
    def __inv_perc_diff1(
        self,
        df_orig_column: pd.Series,
        df_diff_column: pd.Series,
        lagNum: int = 1,
        exclude_orig: bool = True,
    ) -> pd.Series:
        """
        Method for the absolute change inversion

        Parameters
        ----------
        df_orig_column: original series indexed in time order
        df_diff_column: differenced series indexed in time order
        lagNum : an integer > 0 indicating which lag to use.
        exclude_orig: flag indicating whether the 1st value of the
            original data should be included in the result.


        Returns
        -------
        pd.Series: the result of the inverse of the difference arrays.

        """

        # Generate np.array for the diff_inv function - it includes first n values(n =
        # periods) of original data & further diff values of given periods
        _index = [df_diff_column.index.min()]

        for _periods in range(lagNum):
            _index.append(_index[-1] - dateutil.relativedelta.relativedelta(months=1))

        value = np.array(
            df_orig_column.loc[_index[-lagNum:]].sort_index().tolist()
            + (df_diff_column.sort_index()/100+1).values.tolist()
        )

        if exclude_orig:
            # Generate pd.Series with inverse diff
            inv_diff_vals = value.cumprod()[lagNum :]
            inv_diff_series = pd.Series(inv_diff_vals, df_diff_column.index)

            # assert (inv_diff_series == df_orig_column.loc[df_diff_column.index]).all()
        else:
            # Include the 1st (original) observation
            inv_diff_vals = value.cumprod()
            inv_diff_idx = [
                df_diff_column.index.min()
                - dateutil.relativedelta.relativedelta(months=1)
            ] + df_diff_column.index.to_list()

            inv_diff_series = pd.Series(inv_diff_vals, inv_diff_idx)

        return inv_diff_series


    def make_inv_diff(
        self,
        order: int,
        exclude_orig=True,
    ):
        """
        Method for the 1st and 2nd order change inversion

        Parameters
        ----------
        df_orig_column: original series indexed in time order
        df_diff_column: differenced series indexed in time order
        order: an integer > 0 indicating the difference order
        exclude_orig: flag indicating whether the 1st value of the
            original data should be included in the result.

        Returns
        -------
        pd.Series: the result of the inverse of the difference arrays.

        """
        if order == 1:
            return self.__inv_diff1(
                df_orig_column=self.df_orig,
                df_diff_column=self.df_diff,
                lagNum=1,
                exclude_orig=exclude_orig,
            )
        elif order == 2:
            return self.__inv_diff2(
                df_orig_column=self.df_orig,
                df_diff_column=self.df_diff,
                exclude_orig=exclude_orig,
            )
        else:
            return pd.Series()
        
    def make_inv_perc_diff(
        self,
        order: int,
        exclude_orig=True,
    ):
        """
        Method for the 1st and 2nd order change inversion

        Parameters
        ----------
        df_orig_column: original series indexed in time order
        df_diff_column: differenced series indexed in time order
        order: an integer > 0 indicating the difference order
        exclude_orig: flag indicating whether the 1st value of the
            original data should be included in the result.

        Returns
        -------
        pd.Series: the result of the inverse of the difference arrays.

        """
        if order == 1:
            return self.__inv_perc_diff1(
                df_orig_column=self.df_orig,
                df_diff_column=self.df_diff,
                lagNum=1,
                exclude_orig=exclude_orig,
            )
        else:
            return pd.Series()