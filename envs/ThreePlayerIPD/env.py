import itertools, re
from typing import Any, Dict, Optional, Tuple

import textarena as ta


class ThreePlayerIPDEnv(ta.Env):
    def __init__(self, num_rounds: int=5, communication_turns: int=3, cooperate_reward: int=3, defect_reward: int=5, sucker_reward: int=0, mutual_defect_reward: int=1):
        self.num_rounds = num_rounds
        self.conversation_rounds = communication_turns
        self.R, self.T, self.S, self.P = (cooperate_reward, defect_reward, sucker_reward, mutual_defect_reward) # pay-off constants
        self.token_pat = re.compile(r"\[\s*(\d+)\s+(cooperate|defect)\s*\]", re.I)

    def reset(self, num_players: int, seed: Optional[int] = None):
        assert num_players == 3, f"Environment is hard-coded for exactly three players. Received {num_players} players on reset."
        self.state = ta.FFAMultiPlayerState(num_players=num_players, seed=seed)
        game_state = {
            "round": 1, "num_rounds": self.num_rounds, "phase": "conversation", "conversation_round": 0, "total_conversation_rounds": self.conversation_rounds,
            "decisions": {p: {q: None for q in range(num_players) if q != p} for p in range(num_players)},
            "scores": {p: 0 for p in range(num_players)}, "acted": {p: False for p in range(num_players)},
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self.state.add_observation(message=f"─── Starting Round {game_state['round']} ───\tYou can converse freely for the next {game_state['total_conversation_rounds']} rounds.", observation_type=ta.ObservationType.GAME_MESSAGE)


    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in a 3-player Iterated Prisoner's Dilemma. The match lasts {game_state['num_rounds']} rounds.\n"
            f"Round structure:\n"
            f"• {game_state['total_conversation_rounds']} free-chat turns\n"
            f"• 1 decision turn - submit one token per opponent: '[<opp-id> cooperate]' or '[<opp-id> defect]' (i.e. '[1 defect] [2 cooperate]'; the default is 'cooperate'). \n"
            f"Pair-wise payoff matrix (applied to each unordered pair):\n"
            f"  - Both cooperate  ->  {self.R}\n"
            f"  - Both defect     ->  {self.P}\n"
            f"  - You defect, they cooperate -> {self.T}\n"
            f"  - You cooperate, they defect -> {self.S}\n"
            f"The player(s) with the highest score at the end of all rounds wins.\n"
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        cid = self.state.current_player_id
        self.state.add_observation(from_id=cid, to_id=cid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        match self.state.game_state["phase"]:
            case "conversation":    self._conversation_phase(msg=action)
            case "decision":        self._decision_phase(msg=action)
        return self.state.step()
    

    def _clean_message(self, msg: str) -> str: return re.sub(r"\s+", " ", msg) # 1-2. strip() → remove edge whitespace; regex → collapse the rest
    def _conversation_phase(self, msg: str):
        cid = self.state.current_player_id
        # broadcast chat to others
        for pid in range(self.state.num_players):
            if pid != cid:
                self.state.add_observation(from_id=cid, to_id=pid, message=self._clean_message(msg), observation_type=ta.ObservationType.PLAYER_ACTION)

        # increment counter after the last speaker
        if cid == self.state.num_players - 1:
            gs = self.state.game_state
            gs["conversation_round"] += 1
            if gs["conversation_round"] >= gs["total_conversation_rounds"]:
                gs["phase"] = "decision"
                self.state.add_observation(message=f"Chat finished for round {gs['round']}. Submit your decisions, one token per opponent: `[pid cooperate]` or `[pid defect]`.", observation_type=ta.ObservationType.GAME_BOARD)

    def _decision_phase(self, msg: str):
        cid = self.state.current_player_id
        gs = self.state.game_state
        # parse all valid tokens
        for pid_str, choice in self.token_pat.findall(msg):
            tgt = int(pid_str)
            if tgt == cid or tgt not in gs["decisions"][cid]:
                continue  # ignore bad ids / self-targets
            gs["decisions"][cid][tgt] = ("defect" if choice.lower().startswith("d") else "cooperate")
        gs["acted"][cid] = True  # player has taken their decision turn

        # When every player has *acted* once, resolve round.
        if all(gs["acted"].values()):
            # fill unspecified edges with *cooperate*
            for p, row in gs["decisions"].items():
                for q in row:
                    if row[q] is None:
                        row[q] = "cooperate"
            self._resolve_round()

            # next round or finish
            gs["round"] += 1
            if gs["round"] > gs["num_rounds"]:
                self._end_game()
            else:
                # reset for next round
                gs.update({
                    "phase": "conversation", "conversation_round": 0, "acted": {p: False for p in range(self.state.num_players)},
                    "decisions": { p: {q: None for q in range(self.state.num_players) if q != p} for p in range(self.state.num_players)},
                })
                self.state.add_observation(message=f"─── Starting Round {gs['round']} ───\tYou can converse freely for the next {gs['total_conversation_rounds']} rounds.", observation_type=ta.ObservationType.GAME_MESSAGE)

    def _pair_payoff(self, a: str, b: str) -> Tuple[int, int]:
        if a == b == "cooperate":   return self.R, self.R
        if a == b == "defect":      return self.P, self.P
        return (self.S, self.T) if a == "cooperate" else (self.T, self.S) # one cooperates, one defects

    def _resolve_round(self):
        gs = self.state.game_state
        decisions = gs["decisions"]
        round_gain = {p: 0 for p in decisions}

        # compute pair-wise rewards
        message = f"### Round {gs['round']} - Results:"
        for i, j in itertools.combinations(range(self.state.num_players), 2):
            pi, pj = self._pair_payoff(decisions[i][j], decisions[j][i])
            round_gain[i] += pi
            round_gain[j] += pj
            message += f"\n\t Player {i} vs Player {j} chose to {decisions[i][j]} and {decisions[j][i]} respectively (Player {i} gained {pi}, Player {j} gained {pj})"
        message += f"\n-> Current scores: "
        for p, inc in round_gain.items(): gs["scores"][p] += inc # accumulate scores
        message += "; ".join([f"Player {p} ({gs['scores'][p]})" for p in range(self.state.num_players)]) 
        self.state.add_observation(message=message+"\n", observation_type=ta.ObservationType.GAME_MESSAGE)

    def _end_game(self):
        scores = self.state.game_state["scores"]
        ranked = sorted(scores, key=lambda p: (scores[p], -p))  # deterministic
        groups: list[list[int]] = []
        for pid in ranked:
            if not groups or scores[pid] != scores[groups[-1][0]]:  groups.append([pid])
            else:                                                   groups[-1].append(pid)
        G = len(groups)
        reward_dict: Dict[int, float] = {}
        if G == 1: reward_dict = {pid: 0.0 for pid in groups[0]} # complete draw
        else:
            for g_idx, grp in enumerate(groups):   # worst idx 0 … best idx G-1
                r = -1.0 + 2.0 * g_idx / (G - 1)   # evenly spaced in [-1, +1]
                for pid in grp: reward_dict[pid] = r
        self.state.set_game_outcome(reward_dict=reward_dict, reason="Final scores: " + ", ".join(f"P{p}={scores[p]}" for p in sorted(scores))+ f".  Ranking groups (worst→best): {groups}")