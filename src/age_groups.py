"""
Age group configurations for ad attention scoring.
Different age groups have different attention patterns and preferences.
"""

AGE_GROUPS = {
    "gen_z": {
        "name": "Gen Z (18-27)",
        "description": "Digital natives, high attention to motion and fast pacing",
        "weights": {
            "saliency": 0.40,  # Lower weight on static saliency
            "motion": 0.35,    # Higher weight on motion
            "clip": 0.15,
            "pacing": 0.10
        },
        "time_decay": {"start": 1.3, "end": 0.9},  # Stronger early attention
        "thresholds": {
            "excellent": 0.70,  # Lower threshold (more critical)
            "good": 0.55,
            "fair": 0.40,
            "hook_excellent": 0.65,
            "hook_good": 0.50,
            "hook_fair": 0.35
        },
        "pacing_preferences": {
            "hook": {"f_star": 0.6, "lambda": 0.4},      # Faster pacing preferred
            "explainer": {"f_star": 0.4, "lambda": 1.0},
            "calm_brand": {"f_star": 0.3, "lambda": 0.8}
        }
    },
    "millennial": {
        "name": "Millennials (28-43)",
        "description": "Balanced attention, values both engagement and information",
        "weights": {
            "saliency": 0.50,
            "motion": 0.25,
            "clip": 0.15,
            "pacing": 0.10
        },
        "time_decay": {"start": 1.2, "end": 1.0},
        "thresholds": {
            "excellent": 0.75,
            "good": 0.60,
            "fair": 0.45,
            "hook_excellent": 0.70,
            "hook_good": 0.55,
            "hook_fair": 0.40
        },
        "pacing_preferences": {
            "hook": {"f_star": 0.5, "lambda": 0.5},
            "explainer": {"f_star": 0.3, "lambda": 1.2},
            "calm_brand": {"f_star": 0.2, "lambda": 1.0}
        }
    },
    "gen_x": {
        "name": "Gen X (44-59)",
        "description": "Prefers clarity and moderate pacing, less tolerance for rapid cuts",
        "weights": {
            "saliency": 0.55,  # Higher weight on visual clarity
            "motion": 0.20,    # Lower weight on motion
            "clip": 0.15,
            "pacing": 0.10
        },
        "time_decay": {"start": 1.1, "end": 1.0},
        "thresholds": {
            "excellent": 0.75,
            "good": 0.60,
            "fair": 0.45,
            "hook_excellent": 0.70,
            "hook_good": 0.55,
            "hook_fair": 0.40
        },
        "pacing_preferences": {
            "hook": {"f_star": 0.4, "lambda": 0.6},      # Slower pacing preferred
            "explainer": {"f_star": 0.25, "lambda": 1.4},
            "calm_brand": {"f_star": 0.15, "lambda": 1.2}
        }
    },
    "boomer": {
        "name": "Boomers (60+)",
        "description": "Prefers slower pacing, clear visuals, and less rapid changes",
        "weights": {
            "saliency": 0.60,  # Highest weight on visual clarity
            "motion": 0.15,    # Lowest weight on motion
            "clip": 0.15,
            "pacing": 0.10
        },
        "time_decay": {"start": 1.0, "end": 1.0},  # No time decay
        "thresholds": {
            "excellent": 0.70,
            "good": 0.55,
            "fair": 0.40,
            "hook_excellent": 0.65,
            "hook_good": 0.50,
            "hook_fair": 0.35
        },
        "pacing_preferences": {
            "hook": {"f_star": 0.3, "lambda": 0.8},      # Much slower pacing
            "explainer": {"f_star": 0.2, "lambda": 1.5},
            "calm_brand": {"f_star": 0.1, "lambda": 1.3}
        }
    },
    "children": {
        "name": "Children (5-17)",
        "description": "Very high attention to motion and fast pacing, shorter attention spans",
        "weights": {
            "saliency": 0.35,  # Lower weight on static saliency
            "motion": 0.40,    # Highest weight on motion
            "clip": 0.15,
            "pacing": 0.10
        },
        "time_decay": {"start": 1.4, "end": 0.8},  # Strongest early attention, rapid decay
        "thresholds": {
            "excellent": 0.65,  # Lower threshold (easier to lose attention)
            "good": 0.50,
            "fair": 0.35,
            "hook_excellent": 0.60,
            "hook_good": 0.45,
            "hook_fair": 0.30
        },
        "pacing_preferences": {
            "hook": {"f_star": 0.7, "lambda": 0.3},      # Fastest pacing preferred
            "explainer": {"f_star": 0.5, "lambda": 0.8},
            "calm_brand": {"f_star": 0.4, "lambda": 0.6}
        }
    },
    "general": {
        "name": "General Audience",
        "description": "Default settings for mixed demographics",
        "weights": {
            "saliency": 0.50,
            "motion": 0.25,
            "clip": 0.15,
            "pacing": 0.10
        },
        "time_decay": {"start": 1.2, "end": 1.0},
        "thresholds": {
            "excellent": 0.75,
            "good": 0.60,
            "fair": 0.45,
            "hook_excellent": 0.70,
            "hook_good": 0.55,
            "hook_fair": 0.40
        },
        "pacing_preferences": {
            "hook": {"f_star": 0.5, "lambda": 0.5},
            "explainer": {"f_star": 0.3, "lambda": 1.2},
            "calm_brand": {"f_star": 0.2, "lambda": 1.0}
        }
    }
}

def get_age_group_config(age_group="general"):
    """
    Get configuration for a specific age group.
    
    Args:
        age_group: Age group key (gen_z, millennial, gen_x, boomer, children, general)
        
    Returns:
        dict with age group configuration
    """
    return AGE_GROUPS.get(age_group, AGE_GROUPS["general"])

def get_pacing_for_age_group(age_group, goal):
    """
    Get pacing preferences (f_star, lambda) for an age group and goal.
    
    Args:
        age_group: Age group key
        goal: Creative goal (hook, explainer, calm_brand)
        
    Returns:
        dict with f_star and lambda
    """
    config = get_age_group_config(age_group)
    return config["pacing_preferences"].get(goal, config["pacing_preferences"]["hook"])

