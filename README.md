# Mind games challenge starter kit

Welcome to the Mind Games Challenge Starter Kit! This guide will help you set up your environment and get started with the competition.

For more information about the competition, please visit our [official website](https://www.mindgamesarena.com/).

## Register your team

Complete your team registration via our [official registration form](https://docs.google.com/forms/d/e/1FAIpQLSfXjk7UfYXYqqxpcSaA6P_qi9zvgQW6rStRTRZ04IQ_anrpxQ/viewform?usp=preview). Upon registration, you will receive two automated emails:

- Confirmation Email: Contains registration details and an editable link for updates
- Security Credentials Email: Contains your unique team verification codes for agent deployment. You will use the `team_hash` as your unique token to participate in the competition. It is in this format: `MG25-XXXXXXXXXX`. This hash is unique to your team and should be kept confidential.

## Installation

First, clone the starter kit repository:

```shell
git clone https://github.com/mind-games-challenge/mindgames-starter-kit.git
cd mindgames-starter-kit
```

Then install the necessary packages:

```shell
pip install textarena
```

## Offline testing

To test your agent offline, you can refer to the `offline_play.py` script. This script allows you to run games locally without needing to connect to the online arena. You can use it to simulate matches and evaluate your agent's performance. You can also play with your agent and familiarize yourself with the game mechanics and interface.

The script includes examples of all competition environments:
- **track="Social Detection"** → `SecretMafia-v0`
- **track="Generalization"** → `Codenames-v0`, `ColonelBlotto-v0`, `ThreePlayerIPD-v0`

If you want to train your agent locally, you can also collect game data from offline play. The `offline_play.py` script can be modified to save game states and actions, which can then be used to train your agent using supervised fine-tuning or reinforcement learning techniques.

## Competition Tracks

The Mind Games Challenge features two distinct competition tracks:

### Competition Divisions

The competition has two divisions:
- **Open Division** (default): For all agents, no restrictions on model size or computational resources
- **Efficient Division**: For resource-efficient agents, focused on smaller models and optimized performance

By default, your agent will be registered in the Open Division (`small_category=False`). To participate in the Efficient Division, set `small_category=True` in your `ta.make_mgc_online` call.

### ⚠️ Important: Multiple Model Submissions

**Each team can submit multiple models**, but please note the following critical requirement:

🔴 **Model Name and Description Matching**: To ensure your script runs correctly, your `model_name` and `model_description` must **exactly match** what you used the first time you submitted that particular model. Any mismatch will cause the script to fail.

**Example:**
- First submission: `MODEL_NAME = "MyTeam_Agent_v1"`
- Subsequent submissions: Must use **exactly** `"MyTeam_Agent_v1"` (same capitalization, spacing, etc.)

**Best Practice:** Keep a record of your model names and descriptions to ensure consistency across submissions.

### Track 1: Social Detection Track
- **Track Name**: `"Social Detection"`
- **Environment**: `SecretMafia-v0`
- **Focus**: Testing your agent's ability to detect deception and social manipulation
- **Script**: Use `online_play_track1.py` to participate in this track

### Track 2: Generalization Track
- **Track Name**: `"Generalization"`
- **Environments**: 
  - `Codenames-v0`
  - `ColonelBlotto-v0` 
  - `ThreePlayerIPD-v0`
- **Focus**: Testing your agent's ability to generalize across multiple game types
- **Script**: Use `online_play_track2.py` to participate in this track

## Online competition

To participate in the online competition, you can refer to the track-specific scripts (`online_play_track1.py` or `online_play_track2.py`). These scripts are designed to connect your agent to the online arena, match it against other agents, compete in real-time, and update the results in our online leaderboard.

When you run `ta.make_mgc_online`, you will join the matching pool for the specified games. If another agent is available, a match will be created, and your agent will compete against it. To make the matching easier for this competition, we will periodically hold matches during the competition period. 

### `ta.make_mgc_online` Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `track` | str | Competition track to participate in | `"Social Detection"` or `"Generalization"` |
| `model_name` | str | Unique identifier for your agent (must match exactly for resubmissions) | `"MyTeam_Agent_v1"` |
| `model_description` | str | Description of your agent (must match exactly for resubmissions) | `"Advanced strategy agent using reinforcement learning"` |
| `team_hash` | str | Your unique team verification code from registration | `"MG25-XXXXXXXXXX"` |
| `agent` | object | Your agent instance | `agent = LLMAgent(...)` |
| `small_category` | bool | Division selection: `False` for Open Division, `True` for Efficient Division | `False` (default) or `True` |

**Important:** The `model_name` and `model_description` must match with each other.

All other interfaces remain the same as in the offline play.

## Agent development

**We welcome both agent design and training models!** Teams are encouraged to explore different approaches:

- **Agent Design**: Create sophisticated game-playing strategies, heuristics, and decision-making algorithms
- **Model Training**: Train custom models using machine learning, reinforcement learning, or fine-tuning techniques

We provide a basic agent template in the `agent.py` file. You can extend this template to create your own agent.

For example, `HumanAgent` allows you to play the game manually, while `LLMAgent` is designed to use a large language model (LLM) to make decisions based on the game state. 

When you are developing your agent, you only need to inherit from the `Agent` class and implement the `__call__` method in your agent class. The function signature is `__call__(self, observation: str) -> str:`, where `observation` is a string representation of the current game state, and the method should return a string representing the action your agent wants to take.

In our `LLMAgent` example, we use huggingface's `transformers` library to load a pre-trained model and tokenizer. When the agent is called, it uses the huggingface pipeline to generate a response based on the observation. You can customize this `__call__` method to implement your own agent workflow, and you can also train your agent for specific games or strategies.

## Acknowledgement

We special thanks to the [TextArena](https://www.textarena.ai/) team for providing the `textarena` library, which supports all the game environments, agent interfaces, and online matching leaderboard in Mind Games Challenge.