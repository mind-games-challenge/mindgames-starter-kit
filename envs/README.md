# Environment Settings

This directory contains environment settings deployed in the **textarena** package for the Mind Games Challenge.

## Supported Environments

The following game environments are configured for the competition:

### Track 1: Social Detection Track
- **SecretMafia-v0** (`SecretMafia/`)
  - Environment focused on deception detection and social manipulation
  - Tests agents' ability to identify deceptive behavior in social interactions

### Track 2: Generalization Track
- **Codenames-v0** (`Codenames/`)
  - Word association and team coordination game
  - Tests agents' language understanding and strategic communication

- **ColonelBlotto-v0** (`ColonelBlotto/`)
  - Resource allocation and strategic planning game
  - Tests agents' ability to distribute resources optimally across multiple fronts

- **ThreePlayerIPD-v0** (`ThreePlayerIPD/`)
  - Three-player Iterated Prisoner's Dilemma
  - Tests agents' cooperation, defection strategies, and multi-agent interactions

## Usage

These environment settings are automatically deployed through the `textarena` package. When you run:

```python
import textarena as ta
env = ta.make(env_id="SecretMafia-v0")
```

The corresponding environment configuration from this directory is used to initialize the game environment.

## Note

The environment configurations in this directory are part of the textarena package deployment. You do not need to modify these files directly - they are provided for reference and transparency about the game settings used in the competition. 