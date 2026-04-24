# Mind games challenge starter kit

Welcome to the Mind Games Challenge Starter Kit! This guide will help you set up your environment and test your agent **offline** against the top reference submissions.

**Run with Modal Labs Credits**: Deploy your agent in the cloud with $500 free GPU credits. See [modal_lab/MODAL_SETUP.md](modal_lab/MODAL_SETUP.md) for setup instructions.

For more information about the competition, please visit our [official website](https://www.mindgamesarena.com/).

> 🚧 **Online competition is currently closed.** When it reopens we will update the instructions here on GitHub — watch this repo for changes. For now, use the **offline workflow** below.

## 📌 Reference Models & Release Roadmap

> **These are reference inference models** — use them to **test your agent and play against the top submissions locally**. They are vendored as git submodules under [submissions/](submissions/), see the [Top Reference Submissions](#top-reference-submissions) table below for the full list of models and Hugging Face paths.

### TODO

- [ ] **April 2026 release** — top-performance method check: verify each reference agent runs end-to-end locally against its documented install recipe (uv / conda / vllm / ollama) and reproduces its reported TrueSkill in offline play.
- [ ] **May 2026 release** — publish evaluation scripts that automate head-to-head matches against every reference model in this kit, aggregate results, and print a per-environment scorecard.

## Installation

Clone the starter kit and pull the reference submissions:

```shell
git clone https://github.com/mind-games-challenge/mindgames-starter-kit.git
cd mindgames-starter-kit
git submodule update --init --recursive
```

Then install the necessary packages:

```shell
pip install textarena>=0.7.2
```

## Offline testing

Run the [src/offline_play.py](src/offline_play.py) script to play games locally — you can play against an AI yourself, or pit two agents against each other.

Minimal example (human plays `ColonelBlotto` against a hosted model):

```python
import textarena as ta
from agent import HumanAgent

agents = {
    0: HumanAgent(),
    1: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
}

env = ta.make(env_id="ColonelBlotto-v0")
env.reset(num_players=len(agents))

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, step_info = env.step(action=action)

rewards, game_info = env.close()
print(f"Rewards: {rewards}")
```

Swap `agents[1]` with any of the [reference submissions](#top-reference-submissions) below to benchmark against the top agents. See also [src/offline_evaluation.py](src/offline_evaluation.py) for batch evaluation.

The starter kit covers every competition environment:
- **Social Detection** → `SecretMafia-v0`
- **Generalization** → `Codenames-v0`, `ColonelBlotto-v0`, `ThreePlayerIPD-v0`

You can also collect game data from offline play to train your agent — modify `offline_play.py` to save game states and actions for supervised fine-tuning or RL.

## Competition Tracks

Two tracks, both playable offline:

### Track 1: Social Detection
- **Environment**: `SecretMafia-v0`
- **Focus**: Detecting deception and social manipulation

### Track 2: Generalization
- **Environments**: `Codenames-v0`, `ColonelBlotto-v0`, `ThreePlayerIPD-v0`
- **Focus**: Generalizing across multiple game types

## Agent development

**We welcome both agent design and training models!** Teams are encouraged to explore different approaches:

- **Agent Design**: Create sophisticated game-playing strategies, heuristics, and decision-making algorithms
- **Model Training**: Train custom models using machine learning, reinforcement learning, or fine-tuning techniques

We provide a basic agent template in the `agent.py` file. You can extend this template to create your own agent.

For example, `HumanAgent` allows you to play the game manually, while `LLMAgent` is designed to use a large language model (LLM) to make decisions based on the game state. 

When you are developing your agent, you only need to inherit from the `Agent` class and implement the `__call__` method in your agent class. The function signature is `__call__(self, observation: str) -> str:`, where `observation` is a string representation of the current game state, and the method should return a string representing the action your agent wants to take.

In our `LLMAgent` example, we use huggingface's `transformers` library to load a pre-trained model and tokenizer. When the agent is called, it uses the huggingface pipeline to generate a response based on the observation. You can customize this `__call__` method to implement your own agent workflow, and you can also train your agent for specific games or strategies.

## Top Reference Submissions

> 🎯 **Reference inference models — for local testing and self-play.** These are the top-ranked agents across both tracks. Use them to benchmark your own agent, run head-to-head matches offline, and study competitive strategies. **Do not re-submit them as your own entry.**

Reference agents are vendored under [submissions/](submissions/) as git submodules. After cloning the starter kit, pull them with:

```shell
git submodule update --init --recursive
```

Model weights are **not** vendored — the HF paths below are listed for download via `huggingface-cli download <repo>` (or the repo's own install script).

Rankings and TrueSkill values below are from the **Stage II Efficient division** (per the competition report, Table 2 and Appendix C).

### Generalization Track

| Rank | Team | Model | TS (Final Score) | Submodule | Hugging Face / Model |
|------|------|-------|:----------------:|-----------|----------------------|
| 1 | In2AI    | `In2AI_model`           | **34.2** | [submissions/generalization/in2ai](submissions/generalization/in2ai)       | RL-finetuned Qwen3-8B — [`AlekseyKorshuk/mindgames-in2ai-submission`](https://huggingface.co/AlekseyKorshuk/mindgames-in2ai-submission) (served via SGLang) |
| 2 | STARS    | `STARS Agent Track2 V7` | **26.8** | [submissions/generalization/stars](submissions/generalization/stars)       | Ollama `qwen3:8b` (base, code-augmented reasoning — no fine-tune). `ollama pull qwen3:8b` |
| 3 | RLGaming | `RLG-Model8B-Ver12`     | **25.8** | [submissions/generalization/rlgaming](submissions/generalization/rlgaming) | Base [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) + LoRA [`DanDanStonks/RLG-Generalization-Small`](https://huggingface.co/DanDanStonks/RLG-Generalization-Small) |

### Social Detection Track

| Rank | Team | Model | TS (Stage II) | Submodule | Hugging Face / Model |
|------|------|-------|:-------------:|-----------|----------------------|
| 1 | RLGaming | `RLGame-ts-v7`        | **27.2 ± 2.8** | [submissions/social_deduction/rlgaming](submissions/social_deduction/rlgaming) | [`popo9790/Qwen3-8B-Mafia-v2`](https://huggingface.co/popo9790/Qwen3-8B-Mafia-v2) |
| 2 | tungsten | `tungsten_social_v2`  | **23.8 ± 2.8** | [submissions/social_deduction/tungsten](submissions/social_deduction/tungsten) | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (served via vLLM + LiteLLM) |
| 3 | Odyssean | `Odyssean_Social2`    | **18.4 ± 2.9** | [submissions/social_deduction/odyssean](submissions/social_deduction/odyssean) | SFT Qwen-8B — [`yinita/mg-8b-cot-sft-general-1024`](https://huggingface.co/yinita/mg-8b-cot-sft-general-1024) (shared checkpoint with generalization; differentiation is prompt-based, auto-detected per observation) |
| 4 | Phoebus  | `Revac-online`        | **12.5 ± 2.8** | [submissions/social_deduction/phoebus](submissions/social_deduction/phoebus) | Base [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) + LoRA adapters [`mihirArya/qwafia_10`](https://huggingface.co/mihirArya/qwafia_10) (reviewer) and [`mihirArya/qwafiaB_10`](https://huggingface.co/mihirArya/qwafiaB_10) (executor) |

## Acknowledgement

We special thanks to the [TextArena](https://www.textarena.ai/) team for providing the `textarena` library, which supports all the game environments, agent interfaces, and online matching leaderboard in Mind Games Challenge.