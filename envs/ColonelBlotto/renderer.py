def create_game_str(game_state: dict) -> str:
    """Create a formatted string representation of the Colonel Blotto game state"""
    
    lines = []
    
    # Header
    current_round = game_state.get('current_round', 1)
    max_rounds = 10  # Default, should be passed in but keeping simple for now
    phase = game_state.get('phase', 'allocation')
    scores = game_state.get('scores', {0: 0, 1: 0})
    
    lines.append("┌" + "─" * 50 + "┐")
    lines.append(f"│{'COLONEL BLOTTO':^50}│")
    lines.append(f"│{f'Round {current_round} - {phase.title()} Phase':^50}│")
    lines.append("├" + "─" * 50 + "┤")
    lines.append(f"│{f'Score: Alpha {scores[0]} - Beta {scores[1]}':^50}│")
    lines.append("├" + "─" * 50 + "┤")
    
    # Fields display
    fields = game_state.get('fields', [])
    if fields:
        lines.append("│" + " " * 50 + "│")
        lines.append(f"│{'BATTLEFIELD':^50}│")
        lines.append("│" + " " * 50 + "│")
        
        # Field headers
        field_header = "│ Field │ Alpha │ Beta  │ Winner    │"
        lines.append("├" + "─" * 7 + "┼" + "─" * 7 + "┼" + "─" * 7 + "┼" + "─" * 11 + "┤")
        lines.append(field_header)
        lines.append("├" + "─" * 7 + "┼" + "─" * 7 + "┼" + "─" * 7 + "┼" + "─" * 11 + "┤")
        
        # Field data
        for field in fields:
            name = field['name']
            alpha_units = field['player_0_units']
            beta_units = field['player_1_units']
            
            if alpha_units > beta_units:
                winner = "🟢 Alpha"
            elif beta_units > alpha_units:
                winner = "🔴 Beta"
            else:
                winner = "⚪ Tie"
            
            # Handle case where no units are allocated yet
            if alpha_units == 0 and beta_units == 0 and phase == 'allocation':
                winner = "❓ TBD"
            
            field_row = f"│   {name:^3} │  {alpha_units:^3}  │  {beta_units:^3}  │ {winner:^9} │"
            lines.append(field_row)
    
    # Player states (if in allocation phase)
    if phase == 'allocation':
        lines.append("├" + "─" * 50 + "┤")
        lines.append("│" + " " * 50 + "│")
        lines.append(f"│{'ALLOCATION STATUS':^50}│")
        lines.append("│" + " " * 50 + "│")
        
        player_states = game_state.get('player_states', {})
        for player_id in [0, 1]:
            player_name = "Alpha" if player_id == 0 else "Beta"
            if player_id in player_states:
                state = player_states[player_id]
                remaining = state.get('units_remaining', 0)
                complete = state.get('allocation_complete', False)
                
                if complete:
                    status = "✅ Complete"
                    allocation = state.get('current_allocation', {})
                    alloc_str = ", ".join([f"{k}:{v}" for k, v in allocation.items()])
                    if len(alloc_str) > 35:
                        alloc_str = alloc_str[:32] + "..."
                    lines.append(f"│ {player_name}: {status} - {alloc_str:<25} │")
                else:
                    status = f"⏳ {remaining} units left"
                    lines.append(f"│ {player_name}: {status:<40} │")
    
    # Instructions (if in allocation phase and game not over)
    if phase == 'allocation':
        lines.append("├" + "─" * 50 + "┤")
        lines.append("│" + " " * 50 + "│")
        lines.append(f"│{'Enter allocation: A:4, B:2, C:2':^50}│")
        lines.append("│" + " " * 50 + "│")
    
    lines.append("└" + "─" * 50 + "┘")
    
    return "\n".join(lines)


def create_simple_game_str(game_state: dict) -> str:
    """Create a simple text representation for debugging"""
    phase = game_state.get('phase', 'allocation')
    current_round = game_state.get('current_round', 1)
    scores = game_state.get('scores', {0: 0, 1: 0})
    
    lines = [
        f"=== COLONEL BLOTTO - Round {current_round} ===",
        f"Phase: {phase.title()}",
        f"Score: Alpha {scores[0]} - Beta {scores[1]}",
        ""
    ]
    
    fields = game_state.get('fields', [])
    if fields and phase == 'results':
        lines.append("Battle Results:")
        for field in fields:
            alpha = field['player_0_units']
            beta = field['player_1_units']
            winner = "Alpha" if alpha > beta else "Beta" if beta > alpha else "Tie"
            lines.append(f"  Field {field['name']}: Alpha {alpha} vs Beta {beta} -> {winner}")
    
    return "\n".join(lines)