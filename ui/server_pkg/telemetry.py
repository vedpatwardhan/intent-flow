import os


def get_body_ids(sim):
    """
    Returns robust MuJoCo body link IDs for fingertips and target object.
    Matches exact training rollout resolution.
    """
    r_index_id = sim.model.body("R_index_tip_link").id
    r_thumb_id = sim.model.body("R_thumb_tip_link").id
    l_index_id = sim.model.body("L_index_tip_link").id
    l_thumb_id = sim.model.body("L_thumb_tip_link").id
    cube_id = sim.model.body("red_cube").id

    return {
        "r_index_id": r_index_id,
        "r_thumb_id": r_thumb_id,
        "l_index_id": l_index_id,
        "l_thumb_id": l_thumb_id,
        "cube_id": cube_id,
    }
