import json

def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        model_config = json.load(file)
    return model_config


def cyclical_kl_weight(epoch, warmup_epochs, cycle_length, max_beta=1.0):
    """
    带 warmup 的循环退火
    """
    if epoch < warmup_epochs:
        return 0.1
    else:
        cycle_pos = (epoch - warmup_epochs) % cycle_length
        return max_beta * (cycle_pos / cycle_length)