import textarena as ta
from agent import LLMAgent

MODEL_NAME = "Test LLM agent" # Replace with your model name
# The name is used to identify your agent in the online arena and leaderboard.
# It should be unique and descriptive.
# For different versions of your agent, you should use different names.
MODEL_DESCRIPTION = "This agent is for testing purposes only."
team_hash = "MG25-XXXXXXXXXX" # Replace with your team hash

# Initialize your agent
agent = LLMAgent(model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

env = ta.make_mgc_online(
    env_id=["ConnectFour-v0-train","DontSayIt-v0-train"], 
    model_name=MODEL_NAME,
    model_description=MODEL_DESCRIPTION,
    team_hash=team_hash,
    agent=agent
)
env.reset(num_players=1) # always set to 1 when playing online, even when playing multiplayer games.

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
    done, step_info = env.step(action=action)

rewards, game_info = env.close()