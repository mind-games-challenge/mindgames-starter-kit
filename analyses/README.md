# MindGames 2025 game analyses

## Bad Games analyses
`@author = Benjamin Kempinski`

## Response diversity
`@author = Chiara Thöni`

### Summary
A quick analysis of the game responses by calculating the cosine similarities of the model responses. The analysis consists of the following steps:

1. Filter all stage 2 submissions (between `"2025-10-24"` and `"2025-10-27"`). 
2. Extract the model output from the `observations` column. For elaborate observations, where there is a lot of text recorded, I use regular expressions to extract the output of the model from the "model_name" column.
3. Calculate the text embeddings using a LLM-based embedding model. Each model response is translated to a single embedding. The responses are grouped per game.
4. Calculate the cosine similarity between the embedding vectors.
5. Plot the cosine similarities in a heatmap, grouping the responses per game.

### Usage examples
#### Installation & Quick start
You can install all dependencies and run the analyses using the following:

```shell
# Clone the repository
git clone https://github.com/Mindgames-Competition/all_games_analysis
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh  

# If your system doesn't have curl, you can use wget or pip to install it
# wget -qO- https://astral.sh/uv/install.sh | sh
# pip install uv

# Create environment and install Shinka
cd all_games_analysis
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .  # or pip install -r requirements.txt

# Run the analyses
python -m src.diversity_analyses  
```

#### Configuration
The program takes the following parameters

| Flag                  | Description                                    | Type | Default                       | Choices                                                 |
|-----------------------|------------------------------------------------|------|-------------------------------|---------------------------------------------------------|
| -h, --help            | show this help message and exit                |      |                               |                                                         |
| -g, --game            | the game to analyse                            | str  | `'codenames'`                 | `{threeplayeripd,colonelblotto,secretmafia,codenames}`  |
| -e, --embedding_model | The model used for creating text-embeddings.   | str  | `"Qwen/Qwen3-Embedding-0.6B"` |                                                         |
| -t, --max_turns       | The maximal number of turns to analyse.        | int  |                               |                                                         |


>[!NOTE]
> Currently, only [Hugging Face](https://huggingface.co) 🤗 models are supported for creating embeddings. Note that these models will be downloaded upon usage. Hugging Face 🤗 caches all data and models by default in `/.cache/`. This can be troublesome if you are downloading large models on a cluster. Set the environment variables `HF_HOME` and `HF_HUB_CACHE` to a location that is better suited to handling large amount of bits and bytes. 

### Output
The analyses are by default stored in the `./results` folder. Specify the `--game` parameter to analyse different games. For each model in stage 2, the program will generate a diversity heatmap between responses, comparable to the following example from `codenames`:

![gemini_example](./figures/diversity_google_gemini-2.0-flash-001-2.png)

