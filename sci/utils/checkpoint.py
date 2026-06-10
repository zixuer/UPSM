from collections import OrderedDict

import torch


def unwrap_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def strip_module_prefix(state_dict):
    stripped = OrderedDict()
    for key, value in state_dict.items():
        stripped[key[7:] if key.startswith("module.") else key] = value
    return stripped


def load_compatible_state_dict(model, checkpoint_state, strict_shape=True, verbose=True):
    checkpoint_state = strip_module_prefix(unwrap_checkpoint_state(checkpoint_state))
    model_state = model.state_dict()

    compatible_state = OrderedDict()
    skipped_keys = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if strict_shape and tuple(model_state[key].shape) != tuple(value.shape):
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    load_msg = model.load_state_dict(compatible_state, strict=False)

    if verbose:
        print("=" * 80)
        print(f"[Load Model] loaded keys: {len(compatible_state)}")
        print(f"[Load Model] skipped keys: {len(skipped_keys)}")
        print(f"[Load Model] missing keys: {len(load_msg.missing_keys)}")
        print(f"[Load Model] unexpected keys: {len(load_msg.unexpected_keys)}")
        if skipped_keys:
            print("[Load Model] first skipped keys:", skipped_keys[:20])
        if load_msg.missing_keys:
            print("[Load Model] first missing keys:", load_msg.missing_keys[:20])
        if load_msg.unexpected_keys:
            print("[Load Model] first unexpected keys:", load_msg.unexpected_keys[:20])
        print("=" * 80)

    return {
        "loaded_keys": list(compatible_state.keys()),
        "skipped_keys": skipped_keys,
        "missing_keys": list(load_msg.missing_keys),
        "unexpected_keys": list(load_msg.unexpected_keys),
    }


def load_checkpoint_file(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return load_compatible_state_dict(model, checkpoint)
