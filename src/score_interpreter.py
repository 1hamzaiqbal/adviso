"""
Score interpretation and explanation module.
Provides detailed analysis of ad performance scores.
"""

def interpret_score(overall_score, first5s_retention, avg_cut_rate, goal, pacing_f_star, age_group=None):
    """
    Interpret the overall attention score and provide detailed explanation.
    
    Args:
        overall_score: Overall attention score (0-1)
        first5s_retention: Average attention in first 5 seconds (0-1)
        avg_cut_rate: Average cut rate (cuts per second)
        goal: Creative goal ('hook', 'explainer', 'calm_brand')
        pacing_f_star: Target cut rate for the goal
        age_group: Optional age group key to use age-specific thresholds
        
    Returns:
        dict with rating, grade, explanation, strengths, weaknesses, recommendations
    """
    # Get age group-specific thresholds
    if age_group is not None:
        from .age_groups import get_age_group_config
        age_config = get_age_group_config(age_group)
        thresholds = age_config["thresholds"]
        excellent_thresh = thresholds["excellent"]
        good_thresh = thresholds["good"]
        fair_thresh = thresholds["fair"]
        hook_excellent = thresholds["hook_excellent"]
        hook_good = thresholds["hook_good"]
        hook_fair = thresholds["hook_fair"]
        age_group_name = age_config["name"]
    else:
        excellent_thresh = 0.75
        good_thresh = 0.60
        fair_thresh = 0.45
        hook_excellent = 0.70
        hook_good = 0.55
        hook_fair = 0.40
        age_group_name = None
    
    # Determine rating and grade
    if overall_score >= excellent_thresh:
        rating = "Excellent"
        grade = "A"
        performance_desc = f"This ad is highly likely to perform well and capture viewer attention effectively{' for ' + age_group_name if age_group_name else ''}."
    elif overall_score >= good_thresh:
        rating = "Good"
        grade = "B"
        performance_desc = f"This ad shows solid potential but has room for improvement to maximize engagement{' for ' + age_group_name if age_group_name else ''}."
    elif overall_score >= fair_thresh:
        rating = "Fair"
        grade = "C"
        performance_desc = f"This ad may struggle to maintain attention and could benefit from significant optimization{' for ' + age_group_name if age_group_name else ''}."
    else:
        rating = "Needs Improvement"
        grade = "D"
        performance_desc = f"This ad is unlikely to perform well and requires substantial changes to improve engagement{' for ' + age_group_name if age_group_name else ''}."
    
    # Analyze first 5 seconds (critical hook period) with age-specific thresholds
    if first5s_retention >= hook_excellent:
        hook_analysis = "Excellent hook - the opening seconds are highly engaging and likely to capture attention immediately."
    elif first5s_retention >= hook_good:
        hook_analysis = "Good hook - the opening captures attention but could be more compelling."
    elif first5s_retention >= hook_fair:
        hook_analysis = "Weak hook - the opening may not be strong enough to prevent viewers from skipping."
    else:
        hook_analysis = "Poor hook - the opening fails to grab attention, risking immediate viewer drop-off."
    
    # Analyze pacing relative to goal
    goal_names = {
        'hook': 'Hook (fast-paced, attention-grabbing)',
        'explainer': 'Explainer (moderate pacing, informative)',
        'calm_brand': 'Calm Brand (slow-paced, contemplative)'
    }
    goal_name = goal_names.get(goal, goal)
    
    pacing_diff = abs(avg_cut_rate - pacing_f_star)
    if pacing_diff <= 0.3:
        pacing_analysis = f"Pacing aligns well with {goal_name} goal (target: {pacing_f_star:.1f} cuts/sec, actual: {avg_cut_rate:.2f} cuts/sec)."
    elif avg_cut_rate > pacing_f_star + 0.3:
        pacing_analysis = f"Pacing is too fast for {goal_name} goal. Consider slowing down cuts (target: {pacing_f_star:.1f} cuts/sec, actual: {avg_cut_rate:.2f} cuts/sec)."
    else:
        pacing_analysis = f"Pacing is too slow for {goal_name} goal. Consider increasing cut frequency (target: {pacing_f_star:.1f} cuts/sec, actual: {avg_cut_rate:.2f} cuts/sec)."
    
    # Generate strengths (using age-specific thresholds)
    strengths = []
    strong_threshold = (excellent_thresh + good_thresh) / 2
    if overall_score >= strong_threshold:
        strengths.append("Strong overall attention capture")
    if first5s_retention >= hook_good:
        strengths.append("Effective opening hook")
    if pacing_diff <= 0.3:
        strengths.append("Well-matched pacing for creative goal")
    if overall_score >= fair_thresh and first5s_retention < overall_score * 0.9:
        strengths.append("Maintains engagement beyond initial hook")
    
    if not strengths:
        strengths.append("Identified areas for improvement")
    
    # Generate weaknesses (using age-specific thresholds)
    weaknesses = []
    if first5s_retention < hook_good:
        weaknesses.append("Weak opening hook - first 5 seconds need more impact")
    if overall_score < good_thresh:
        weaknesses.append("Overall attention score below optimal threshold")
    if pacing_diff > 0.5:
        weaknesses.append(f"Pacing doesn't match {goal_name} goal effectively")
    if first5s_retention > overall_score * 1.2:
        weaknesses.append("Attention drops significantly after initial hook")
    
    if not weaknesses:
        weaknesses.append("Minor optimizations possible")
    
    # Generate recommendations (using age-specific thresholds)
    recommendations = []
    if first5s_retention < hook_good:
        recommendations.append("Strengthen the opening 3-5 seconds with more compelling visuals, motion, or contrast")
    if pacing_diff > 0.4:
        if avg_cut_rate > pacing_f_star:
            recommendations.append(f"Reduce cut frequency to better match {goal_name} pacing (aim for ~{pacing_f_star:.1f} cuts/sec)")
        else:
            recommendations.append(f"Increase cut frequency to create more dynamic pacing (aim for ~{pacing_f_star:.1f} cuts/sec)")
    if overall_score < strong_threshold:
        recommendations.append("Increase visual saliency in key frames - use contrast, color, and composition to draw attention")
    if first5s_retention < overall_score * 0.85:
        recommendations.append("Maintain engagement throughout - avoid attention drop-off after the hook")
    
    # Add age-specific recommendations
    if age_group_name:
        if age_group == "gen_z":
            recommendations.append("Consider increasing motion and faster pacing - Gen Z responds well to dynamic content")
        elif age_group == "boomer":
            recommendations.append("Consider slower pacing and clearer visuals - Boomers prefer less rapid changes")
        elif age_group == "gen_x":
            recommendations.append("Balance clarity with engagement - Gen X values both visual clarity and moderate pacing")
        elif age_group == "children":
            recommendations.append("Maximize motion and fast pacing - Children have high attention to movement and prefer dynamic, fast-paced content")
    
    if not recommendations:
        recommendations.append("Continue monitoring performance and A/B test variations")
    
    # Performance prediction (using age-specific thresholds)
    if overall_score >= excellent_thresh:
        prediction = f"High likelihood of strong performance{' for ' + age_group_name if age_group_name else ''}: expect above-average view-through rates, engagement, and conversion potential."
    elif overall_score >= good_thresh:
        prediction = f"Moderate performance expected{' for ' + age_group_name if age_group_name else ''}: competitive view-through rates with potential for optimization gains."
    elif overall_score >= fair_thresh:
        prediction = f"Below-average performance likely{' for ' + age_group_name if age_group_name else ''}: may struggle with viewer retention and may need significant revisions."
    else:
        prediction = f"Poor performance expected{' for ' + age_group_name if age_group_name else ''}: high risk of low engagement, view-through, and conversion rates."
    
    result = {
        'rating': rating,
        'grade': grade,
        'overall_score': round(overall_score, 3),
        'performance_prediction': prediction,
        'detailed_explanation': {
            'summary': performance_desc,
            'hook_analysis': hook_analysis,
            'pacing_analysis': pacing_analysis,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations
        }
    }
    
    if age_group_name:
        result['age_group'] = age_group_name
    
    return result

