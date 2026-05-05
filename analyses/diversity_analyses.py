import re
import os
from typing import List
from pathlib import Path
import logging
import argparse

import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Dataset
from sentence_transformers import SentenceTransformer

from llm.embed_openai import EmbeddingTransformer
from src.llm.embedding_model import get_embedding_model
import src.utils.stages as stages
from src.utils.filter import filter_by_range, filter_by_model, filter_model_response, split_by_stage
from src.utils.plot import heatmap, add_patches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DICT = name_mapping = {
    "RLG-Stage2-Heavy": "RLG-S2",
    "RLG-Stage2-Heavy-v2": "RLG-S2-V2",
    "RLG-Model8B-Ver12": "RLG-8B-V12",
    "RLGame-ts-v7": "RLG-TS-V7",
    "Corleone_Reflextion": "Corleone-Ref",
    "STARS Agent Track2 V7": "STARS-T2-V7",
    "Odyssean_Generalization": "Odyssean-Gen",
    "Odyssean_Social2": "Odyssean-Soc-V2",
    "tungsten_social_v2": "Tungsten-Soc-V2",
    "BanMaHeavy": "BanMaHeavy",
    "Revac-online": "Revac-online",
    "Vito-1.0": "Vito-1.0",
    "ZeroR-SecretMafia-Efficient-v4": "ZeroR-SM-V4",
    "OrderOfPhoenix/jarvis_v6": "OrderOP-j-V6",
    "In2AI_model": "In2AI-Base",
    "In2AI/mindgames-in2ai-submission-stage-7-step-30-temperature-1.0": "In2AI-S7-St30-T1.0",
    "In2AI/mindgames-in2ai-submission-stage-7-step-45-temperature-1.0": "In2AI-S7-St45-T1.0",
    "In2AI/mindgames-in2ai-submission-stage-7-step-45-temperature-0.6": "In2AI-S7-St45-T0.6",
    "In2AI/mindgames-in2ai-submission-temperature-1.0": "In2AI-T1.0",
    "In2AI/mindgames-in2ai-submission-temperature-0.8": "In2AI-T0.8",
    "In2AI/mindgames-in2ai-submission-temperature-0.6": "In2AI-T0.6",
    "In2AI/mindgames-in2ai-submission-stage-7-step-30-temperature-0.8": "In2AI-S7-St30-T0.8",
    "In2AI/mindgames-in2ai-submission-mix-v1": "In2AI-Mix-V1",
    "In2AI/mindgames-in2ai-submission-m-v3": "In2AI-Mv3",
}


def analyse_diversity(
    data: DatasetDict, embedding_model: str, game: str, track: str, max_turns: int, within_model: bool
) -> None:
    """
    Analyze the response diversity of the MindGames challenge.
    Args:
        data (DatasetDict): a MindGames 2025 challenge dataset.
        embedding_model (str): the model used for creating text-embeddings. Currently, only Hugging Face models are
            supported.
        game (str): the MindGames 2025 game the dataset corresponds to. Choose from
            {threeplayeripd, colonelblotto, secretmafia, codenames}
        track (str): the models to analyze. Choose from
            ["threeplayeripd_small", "threeplayeripd_large", "colonelblotto_small", "colonelblotto_large",
            "secretmafia_small", "secretmafia_large", "codenames_small", "codenames_large",]
        max_turns (int): the maximal number of turns to consider.

    Returns:
        None
    Notes:
        * Text length and distribution thereof
        * Cosine similarity of text embeddings
            * Within model depending on the number of responses (Make a 2D comparison heatmap / histogram, with the
                response index on the axes)
            * Between models with the average representation    (Make a 2D heatmap, with the model names on the axes)


    """
    # data_dict = split_by_stage(data['train'])
    # print("Vito-1.0" in data_dict['stage2']['model_name'].unique())
    stage_2 = filter_by_range(data["train"], start_date="2025-10-14", end_date=None)
    # print("Vito-1.0" in data["train"].unique("model_name"), "Vito-1.0" in stage_2.unique("model_name"))
    stage_2 = filter_model_response(stage_2, game=game)

    models = f"{game}_{track}"

    path = "./results/similarities" if within_model else "./results/similarities_b"
    embedding_path = path + "_" + embedding_model.replace(".", "-").replace("/", "_")
    logger.info(f"Saving responses to {embedding_path}.")
    plot_response_length(stage_2, game=game, path=path)
    plot_embedding_similarity(
        stage_2, game, models, embedding_model, max_turns=max_turns, path=embedding_path, within_model=within_model
    )


def plot_embedding_similarity(
    data: Dataset,
    game: str,
    models: str,
    model_name_or_path: str,
    within_model: bool,
    path: str = "./results/similarities",
    max_turns: int = 50,
) -> None:
    """
    Plot the cosine similarity of the textual responses.

    Args:
        data (Dataset): a MindGames 2025 challenge dataset.
        game (str): the MindGames 2025 game the dataset corresponds to. Choose from
            {threeplayeripd, colonelblotto, secretmafia, codenames}
        models (str): the models to analyze. Choose from
            ["threeplayeripd_small", "threeplayeripd_large", "colonelblotto_small", "colonelblotto_large",
            "secretmafia_small", "secretmafia_large", "codenames_small", "codenames_large",]
        model_name_or_path (str): the model used for creating text-embeddings. Currently, only Hugging Face and OpenAI
            models are supported.
        within_model (bool): True to calculate the similarities within model. Else calculates the similarities between
            models.
        path (str): the filepath to save the results at.
        max_turns (int): the maximal number of turns to consider.

    Returns:
        None
    """

    embedding_model = get_embedding_model(model_name_or_path)

    logger.info(f"Calculating the {embedding_model.similarity_fn_name} similarity of the embeddings.")

    if within_model:
        # models = data.unique("model_name")
        models = getattr(stages, models)
        for model in models:
            get_within_similarities(model, game, embedding_model, data, max_turns, path)

    else:
        get_between_similarities(data, game, models, embedding_model, max_turns, path)


def get_within_similarities(
    model: str,
    game: str,
    embedding_model: EmbeddingTransformer | SentenceTransformer,
    data: Dataset,
    max_turns: int,
    path: str,
):
    """
    Calculate the similarities between responses of the same model.

    Args:
        model (str): the model to filter on.
        game (str): the MindGames 2025 game the dataset corresponds to. Choose from
            {threeplayeripd, colonelblotto, secretmafia, codenames}
        embedding_model (EmbeddingTransformer | SentenceTransformer): the embedding client.
        data (Dataset): a MindGames 2025 challenge dataset.
        max_turns (int): the maximal number of turns to consider.
        path (str): the filepath to save the results at.

    Returns:
        None
    """

    def _get_self_similarity(
        text: List[str],
    ) -> np.array:
        """
        Calculate the cosine similarity between the responses in ``text``.

        Args:
            text (List[str]): a list of textual responses.

        Returns:
            np.array: A list of cosine similarities.
        """
        embeddings = embedding_model.encode(text, batch_size=4)
        if isinstance(embedding_model, SentenceTransformer):
            return embedding_model.similarity(embeddings, embeddings).detach().numpy()
        else:
            return embedding_model.similarity(embeddings, embeddings)

    filtered_data = filter_by_model(data, model=model)
    if len(filtered_data) < 1:
        return
    flatten_responses = [turn for response in filtered_data["player_responses"] for turn in response if turn != ""][
        :max_turns
    ]
    box_indices = [len(game) for game in filtered_data["player_responses"]]
    similarities = _get_self_similarity(flatten_responses)

    fig, ax = plt.subplots()

    im, cbar = heatmap(
        similarities,
        ax=ax,
        row_labels=np.arange(0, len(flatten_responses), 5),
        col_labels=np.arange(0, len(flatten_responses), 5),
        cmap="Blues",
        cbarlabel="Cosine similarity",
    )

    ax = add_patches(box_indices, ax)
    ax.set_xlabel("Response index")
    ax.set_ylabel("Response index")

    fig.tight_layout()

    folder = os.path.join(path, game)
    Path(folder).mkdir(exist_ok=True, parents=True)

    filename = os.path.join(folder, f"{model.replace('/', '_')}")
    fig.savefig(filename + ".pdf", bbox_inches="tight")
    np.save(filename + ".npy", similarities)
    plt.close(fig)


def get_between_similarities(
    data: Dataset,
    game: str,
    stage: str,
    embedding_model: EmbeddingTransformer | SentenceTransformer,
    max_turns: int,
    path: str,
):
    """
    Calculate the cosine similarity between model responses.

    Args:
        data (Dataset): a MindGames 2025 challenge dataset.
        game (str): the MindGames 2025 game the dataset corresponds to. Choose from
            {threeplayeripd, colonelblotto, secretmafia, codenames}
        stage (str): the models to analyze.
        embedding_model (EmbeddingTransformer | SentenceTransformer): the embedding client.
        max_turns (int): the maximal number of turns to consider.
        path (str): the filepath to save the results at.
    Returns:
        None
    """
    models = getattr(stages, stage)
    folder = os.path.join(path, game)
    Path(folder).mkdir(exist_ok=True, parents=True)
    avg_embeddings = []
    saved_models = []
    for model in models:
        path = os.path.join(folder, f"{model.replace('/', '-')}_average_embedding.npy")
        if not os.path.exists(path):
            logger.info(f"Calculating embeddings using {embedding_model.model_name_or_path}.")
            avg_embedding = get_average_embedding(model, embedding_model, data, max_turns)
            np.save(path, avg_embedding)
        else:
            logger.info(f"Loading embeddings from {path}.")
            avg_embedding = np.load(path)
        if avg_embedding.shape != ():
            avg_embeddings.append(avg_embedding)
            saved_models.append(model)
    avg_embeddings = np.array(avg_embeddings)

    similarities = embedding_model.similarity(avg_embeddings, avg_embeddings)

    labels = [MODEL_DICT[l] if l in MODEL_DICT.keys() else l for l in saved_models]

    fig, ax = plt.subplots()
    im, cbar = heatmap(
        similarities,
        ax=ax,
        rotation=45,
        row_labels=labels,
        col_labels=labels,
        cmap="Blues",
        cbarlabel="Cosine similarity",
        step=1,
        vmin= 0.4,
        vmax=1.0,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    fig.tight_layout()
    filename = os.path.join(folder, f"{stage}_between_model_similarities")
    fig.savefig(filename + ".pdf", bbox_inches="tight")
    np.save(filename + ".npy", similarities)
    plt.close(fig)


def get_average_embedding(
    model: str, embedding_model: EmbeddingTransformer | SentenceTransformer, data: Dataset, max_turns: int
) -> np.ndarray:
    """
    Calculate the average embedding over model responses.

    Args:
        model (str): the model to filter on.
        embedding_model (EmbeddingTransformer | SentenceTransformer): the embedding client.
        data (Dataset): a MindGames 2025 challenge dataset.
        max_turns (int): the maximal number of turns to consider.
    Returns:

    """
    filtered_data = filter_by_model(data, model=model)
    print(model, filtered_data)
    flatten_responses = [turn for response in filtered_data["player_responses"] for turn in response if turn != ""][
        :max_turns
    ]
    embeddings = embedding_model.encode(flatten_responses, batch_size=4)
    if isinstance(embeddings, Tensor):
        embeddings = embeddings.detach.numpy()
    return np.mean(embeddings, axis=0)


def plot_response_length(data: Dataset, game: str, path: str = "./results/") -> None:
    """
    Plot a histogram of the average response length per turn.

    Args:
        data (Dataset): a MindGames 2025 challenge dataset including the ``player_responses`` column.
        game (str): the MindGames 2025 game the dataset corresponds to. Choose from
            {threeplayeripd, colonelblotto, secretmafia, codenames}
        path (str): the filepath to save the results at.
    Returns:
        None
    """
    filename = os.path.join(path, f"response_lengths_{game}.pdf")
    Path(path).mkdir(exist_ok=True, parents=True)

    models = data.unique("model_name")
    avg_response_lengths = np.zeros(len(models))

    for i, model in enumerate(models):
        response_lengths = get_response_length(data, model)
        avg_response_lengths[i] = np.mean(response_lengths)

    fig, axes = plt.subplots(
        1,
        1,
    )
    axes.hist(avg_response_lengths, bins=100)
    axes.set_xlabel("Response length (words)")
    axes.set_ylabel("Number of observations")
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")


def get_response_length(data: Dataset, model: str):
    """
    Get the average response lengths per model and turn.

    Args:
        data (Dataset): a MindGames 2025 challenge dataset including the ``player_responses`` column.
        model (str): the model to filter on.

    Returns:
        list[str]: the list of models
        list[list[float]]: the average response length per turn. Shape n_games, n_turns. Not guaranteed to be symmetric.
    """
    filtered_data = filter_by_model(data, model=model)
    messages = filtered_data["player_responses"]

    text_lengths = np.zeros((len(messages),))
    for i, row in enumerate(messages):
        row = [m for m in row if m != []]
        text_lengths[i] = np.mean([len(re.split(" |:", r)) for r in row if len(r) > 0])
    return text_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="diversity",
        description="Analyse the diversity of the MindGames 2025 Challenge responses",
        epilog="License: MIT",
    )
    parser.add_argument(
        "--game",
        "-g",
        choices=["threeplayeripd", "colonelblotto", "secretmafia", "codenames"],
        help="the game to analyse. Default = 'codenames'.",
        default="codenames",
    )
    parser.add_argument(
        "--embedding_model",
        "-e",
        help="The model used for creating text-embeddings. Currently, only Hugging Face models are supported.",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument(
        "--max_turns",
        help="The maximal number of turns to analyse.",
        default=50,
    )
    parser.add_argument(
        "--track",
        "-t",
        choices=["small", "large"],
        help="the models to analyse. Default = 'small'.",
        default="small",
    )

    parser.add_argument(
        "--within_model",
        "-w",
        help="Set to True for within model evaluation",
        default=False,
    )

    args = parser.parse_args()

    ds = load_dataset("mindgameschallenge/mgc2025", name=args.game)
    analyse_diversity(
        data=ds,
        game=args.game,
        track=args.track,
        embedding_model=args.embedding_model,
        max_turns=args.max_turns,
        within_model=args.within_model,
    )
