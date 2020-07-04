def update_ema_variables(ema_model, model, alpha=0.9):
    # Use the true average until the exponential average is more correct
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)