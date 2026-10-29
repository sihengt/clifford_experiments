from TrajProc.models.DSKBM import action_type

ACCELERATION_LABEL = "Acceleration ($m/s^2$)"
SPEED_LABEL = "Speed command ($m/s$)"


def first_action_label(model_cfg):
    """y label of the first MPC action. model_cfg is mpc_params['model'] (None or {}: acceleration)."""
    if action_type(model_cfg or {}) == 'speed_target':
        return SPEED_LABEL
    return ACCELERATION_LABEL
