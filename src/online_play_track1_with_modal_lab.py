"""
Track 1: Social Detection Track with Modal Labs
This script runs your agent in the cloud using Modal Labs for the social detection track.
Environment: SecretMafia-v0
"""

import modal
import time

app = modal.App("mindgames-track1-online")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "textarena>=0.7.2",
        "transformers",
        "torch",
        "accelerate",
    )
    .add_local_python_source("agent")
)

@app.function(
    image=image,
    gpu="T4",  # T4 GPU for inference
    timeout=60 * 60,  # 60 minutes timeout, you could manually stop the job it the game is over
)
#    secrets=[
#         modal.Secret.from_name("huggingface-secret"),  # Add if using HF models
#     ],

def play_online():
    import textarena as ta
    from agent import LLMAgent
    
    MODEL_NAME = "Test LLM agent - Track 1"
    MODEL_DESCRIPTION = "This agent is for Track 1 - Social Detection (SecretMafia-v0)."
    team_hash = "MG25-XXXXXXXXXX"  # Replace with your team hash
    
    agent = LLMAgent(model_name="Qwen/Qwen3-4B")
    
    env = ta.make_mgc_online(
        track="Social Detection", 
        model_name=MODEL_NAME,
        model_description=MODEL_DESCRIPTION,
        team_hash=team_hash,
        agent=agent,
        small_category=False
    )
    env.reset(num_players=1)
    
    done = False
    step_count = 0
    while not done:
        player_id, observation = env.get_observation()
        print(f"Step {step_count}: Player {player_id}")
        
        action = agent(observation)
        done, step_info = env.step(action=action)
        step_count += 1
    
    rewards, game_info = env.close()
    
    print(f"Game completed in {step_count} steps")
    print(f"Rewards: {rewards}")
    print(f"Game info: {game_info}")
    
    return {
        "rewards": rewards,
        "game_info": game_info,
        "steps": step_count
    }

@app.local_entrypoint()
def main():
    with modal.enable_output():
        result = play_online.remote()
        print(f"Final result: {result}")

if __name__ == "__main__":
    main()