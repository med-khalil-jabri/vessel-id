dataset_norms: dict = {
    'SOP': {
        'mean': [0.5807, 0.5396, 0.5044],
        'std': [0.2901, 0.2974, 0.3095]
    },
    'Hotels-50k': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'GLM': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'imagenet21k': { # TODO this are false statistics. Need to change them.
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}