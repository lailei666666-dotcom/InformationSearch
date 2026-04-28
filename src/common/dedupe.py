import pandas as pd


def duplicate_mask(frame: pd.DataFrame, subset: list[str] | tuple[str, ...]) -> pd.Series:
    return frame.duplicated(subset=list(subset), keep="first")


def drop_duplicates(frame: pd.DataFrame, subset: list[str] | tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[~duplicate_mask(frame, subset)].copy()
