"""Episode-bound initial observation admission for G1 motion."""


def initial_observation_allowed(directory, frame_id, job_id=None):
    """Host-bound exception for the first candidate; never alter sensor timestamps."""
    import json
    from pathlib import Path
    try:
        marker=json.loads((Path(directory)/'execution-memory.json').read_text())
        state=json.loads((Path(marker['directory'])/'state.json').read_text())
        admitted=False
        if job_id:
            admission=json.loads((Path(directory)/'initial-motion-admission.json').read_text())
            admitted=(admission['job_id']==job_id and admission['episode']==state['episode']
                      and admission['frame_id']==frame_id)
        return bool(state['episode']==marker['episode'] and state['lifecycle']=='running'
            and (state['first_action'] or admitted) and not state.get('finish_requested')
            and not state['prepare'].get('moved')
            and state['initial_frame']['frame_id']==frame_id)
    except (OSError,ValueError,KeyError,TypeError):
        return False


def reusable_episode_frame(directory,frame_id):
    """No age limit: reuse an unchanged episode view, bound to its current identity."""
    import json
    from pathlib import Path
    try:
        marker=json.loads((Path(directory)/'execution-memory.json').read_text())
        state=json.loads((Path(marker['directory'])/'state.json').read_text())
        return bool(state.get('compact_action_loop') and state['episode']==marker['episode']
            and state['lifecycle']=='running' and not state.get('finish_requested')
            and (state.get('current_frame') or {}).get('frame_id')==frame_id)
    except (OSError,ValueError,KeyError,TypeError):return False
