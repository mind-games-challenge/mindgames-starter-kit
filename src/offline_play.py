"""
This script allows you to play against a fixed model.
You (human player) will be player 0, and the AI model will be player 1.
"""

import textarena as ta 
from agent import HumanAgent

# initialize the agents
agents = {
    0: HumanAgent(),
    1: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
}

# initialize the environment
env = ta.make(env_id="TicTacToe-v0-train")
env.reset(num_players=len(agents))

# main game loop
done = False 
while not done:
  player_id, observation = env.get_observation()
  action = agents[player_id](observation)
  done, step_info = env.step(action=action)
rewards, game_info = env.close()

print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")