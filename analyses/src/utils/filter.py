import re
import json
import pytz
import logging
import  pandas as pd
from datetime import datetime, date


from datasets import Dataset

utc = pytz.UTC
logger = logging.getLogger(__name__)


def extract_first_observation_date(observations_str):
    """
    Extracts the date from the first timestamp key in observations JSON.
    Returns a datetime.date object or None if parsing fails.
    """
    def parse_dict_column(val):
        if isinstance(val, dict): return val
        if pd.isna(val): return {}
        try:
            return json.loads(val)
        except:
            return {}

    obs_dict = parse_dict_column(observations_str)
    if not obs_dict:
        return None
    try:
        # Get the first key (which is the timestamp string)
        first_key = next(iter(obs_dict))
        # Parse the timestamp and extract just the date
        return pd.to_datetime(first_key).date()
    except (StopIteration, ValueError, TypeError):
        return None

def split_by_stage(processed_data, cutoff_date=date(2025, 10, 14)):
    """
    Splits processed data into Stage 1 and Stage 2.
    Stage 1: >= cutoff_date (earlier stage)
    Stage 2: < cutoff_date (later/final stage)
    """
    df = processed_data.to_pandas()
    df['observation_date'] = df['observations'].apply(extract_first_observation_date)

    stage_data = {
        'stage1': df[df['observation_date'] < cutoff_date].copy(),
        'stage2': df[df['observation_date'] >= cutoff_date].copy(),
        'all': df.copy()
    }
    return stage_data

def filter_by_range(
    data: Dataset | list,
    date_column: str = "observations",
    start_date: str = None,
    end_date: str = None,
):
    """
    Filter the ``data`` by the provided date range. Normalizes all dates to UTC.

    Args:
        data (Dataset): a MindGames 2025 challenge dataset.
        date_column (str): the column containing the date.
        start_date (str): the start date in format 'YYYY-MM-DD'. If None, no lower bound is applied.
        end_date (str): the end date in format 'YYYY-MM-DD'. If None, no upper bound is applied.

    Returns:
        Dataset: the filtered dataset
    """
    start_date = utc.localize(datetime.strptime(start_date, "%Y-%m-%d")) if start_date else None
    end_date = utc.localize(datetime.strptime(end_date, "%Y-%m-%d")) if end_date else None

    def _filter(row: Dataset) -> bool:
        """
        Retrieve the appropriate boolean for ``row``.

        Args:
            row (Dataset): the row to consider.

        Returns:
            bool: inclusion indication.
        """
        date_str = list(json.loads(row[date_column]).keys())[0]

        try:
            date = datetime.fromisoformat(date_str)

            if start_date and date < start_date:
                return False
            if end_date and date > end_date:
                return False
            return True
        except (TypeError, ValueError) as e:
            logger.error(f"Could not parse date {date_str}. Excluding it from the dataset.")
            logger.exception(e)
            return False

    # Apply filter to dataset
    return data.filter(_filter)


def filter_by_model(data: Dataset, model: str):
    """
    Filter ``data`` by ``model``.

    Args:
        data (Dataset): a MindGames 2025 challenge dataset.
        model (str): a model name.

    Returns:
        Dataset: the filtered dataset.
    """
    def _filter(row: Dataset) -> bool:
        """
        Retrieve the appropriate boolean for ``row``.

        Args:
            row (Dataset): the row to consider.

        Returns:
            bool: inclusion indication.
        """
        return row["model_name"] == model

    return data.filter(_filter)


def filter_model_response(
    data: Dataset, game: str, trace_column: str = "observations", separator: str = " "
) -> Dataset:
    """
    Filter the response of the player indicated at ``player_id`` from the ``obsesrvations`` column. Returns a new
    Dataset with a ``player_responses`` column, which contains a list of the responses per turn. The responses are
    encoded as strings. If a player didn't speak during a turn, then an empty string is returned.

    Args:
        data (Dataset): a MindGames 2025 challenge dataset.
        game (str): the MindGames 2025 game the dataset corresponds to. Choose from
            {threeplayeripd, colonelblotto, secretmafia, codenames}
        trace_column (str): the name of the column containing the game traces. Default = 'observations'.
        separator (str): the separator to use when a player submits more than two messages per turn.

    Returns:
        Dataset: a new dataset where the players' responses are recorded in a ``player_responses`` column.
    """
    assert game in ["threeplayeripd", "colonelblotto", "secretmafia", "codenames"], (
        f"The chosen game should be threeplayeripd, colonelblotto, secretmafia or codenames, received {game}."
    )

    def _extract_player_messages(row: Dataset) -> Dataset:
        """
        Extracts the messages from player ``player_id`` from the ``trace_column``. Based on a regex-based search on the
        player id. Applicable to ``secretmafia`` and ``threeplayeripd``.

        Args:
            row (Dataset): a row in the dataset.

        Returns:
            Dataset: the updated row.
        """
        player_id = row["player_id"]

        traces = list(json.loads(row[trace_column]).values())
        turns = []

        for trace in traces:
            trace = trace["observation"]
            # Pattern to match [player_index] followed by text until next [index] or end
            pattern = rf"\[{player_id}\]\s*(.*?)(?=\s*\[\d+\]|$)"

            # Find all matches
            matches = re.findall(pattern, trace, re.DOTALL)

            # Strip whitespace from each match and filter out empty strings
            messages = [msg.strip() for msg in matches if msg.strip()]

            turns.append(separator.join(messages))

        row["player_responses"] = turns
        return row

    def _extract_player_actions(row: Dataset) -> Dataset:
        """
        Extracts the messages from player ``player_id`` from the ``trace_column``. Based on the player's action.
        Applicable to ``colonelblotto`` and ``codenames``.

        Args:
            row (Dataset): a row in the dataset.

        Returns:
            Dataset: the updated row.
        """
        player_id = row["player_id"]

        traces = list(json.loads(row[trace_column]).values())
        turns = []

        for trace in traces:
            turns.append(trace["action"].strip())

        row["player_responses"] = turns
        return row

    if game == "secretmafia" or game == "threeplayeripd":
        data = data.map(_extract_player_messages)

    elif game == "colonelblotto" or game == "codenames":
        data = data.map(_extract_player_actions)

    return data
