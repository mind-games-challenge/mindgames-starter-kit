"""
This script allows you to play against a fixed model.
You (human player) will be player 0, and the AI model will be player 1.
"""

import textarena as ta 
from agent import HumanAgent, RandomAgent, LLMAgent

# initialize the agents
agents = {
    0: HumanAgent(),
    # 1: RandomAgent(),
    # 1: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
    1: LLMAgent("google/gemini-2.0-flash-lite-001")
}

# initialize the environment
env = ta.make(env_id="TicTacToe-v0-train")
env.reset(num_players=len(agents))

# main game loop
done = False 
while not done:
    player_id, observation = env.get_observation()
    response = agents[player_id](observation)         #response will be a list
    action = response[0]
    if len(response) > 1:                             #allows agents to return two items
        secondary = response
        print(f"\nsubmitted: {action}")
        print(f"additional return from agent {player_id}:\n {secondary} \n")   #intended to diagnose screw ups
    done, step_info = env.step(action = action) #! keep in mind that you get two chances to submit a valid response
rewards, game_info = env.close()

print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")
