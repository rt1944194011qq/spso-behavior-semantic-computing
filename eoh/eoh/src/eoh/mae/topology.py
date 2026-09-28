from __future__ import annotations


def ring_neighbors(agent_id: int, num_agents: int) -> tuple[int, int]:
    if num_agents < 1:
        raise ValueError("num_agents must be >= 1")
    if not 0 <= agent_id < num_agents:
        raise ValueError("agent_id out of range")
    if num_agents == 1:
        return agent_id, agent_id
    return (agent_id - 1) % num_agents, (agent_id + 1) % num_agents
