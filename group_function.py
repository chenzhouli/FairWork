from functions.group import *
from config import INDUSTRIES

def prepare_analysis_data(
    users_file, jobs_file, apps_file, num_users, N, pos_ratio, cancel_event
):

    try:
        df_users = load_users(users_file)
        df_jobs = load_jobs(jobs_file)
        try:
            df_apps = load_apps(apps_file) 
        except Exception:
            df_apps = None 
    except Exception as e:
        print(e)
    
    original_predictions = []
    n_keywords= 5
    n_clusters = 12
    similarity_threshold = 0.2
    if 'Industry' not in df_users.columns:
        df_users = cluster_data(df_users, n_keywords, INDUSTRIES, similarity_threshold, n_clusters, cancel_event)
        df_users.to_csv("data/users_indus.tsv", index=False, sep="\t")

    if 'Industry' not in df_jobs.columns:
        df_jobs = cluster_data(df_jobs, n_keywords, INDUSTRIES, similarity_threshold, n_clusters, cancel_event)
        df_jobs.to_csv("data/jobs_indus.tsv", index=False, sep="\t")

    df_users_sample = sample_users_by_cluster(df_users, num_users=num_users)
    current_user_ids = df_users_sample["UserID"].unique().tolist()


    user_jobs_map = {}
    original_predictions = []
    if df_apps is not None:
        for uid in current_user_ids:
            industry = df_users.loc[df_users["UserID"] == uid, 'Industry'].iloc[0]
            df_chosen, original_prediction = sample_jobs_for_user(df_jobs, df_apps, uid, industry, N, pos_ratio)
            user_jobs_map[uid] = df_chosen
            original_predictions.extend(original_prediction)
    else:
        for uid in current_user_ids:
            df_chosen, original_prediction = sample_jobs_for_user_simi(df_jobs, df_users=df_users, user_id=uid, N=N, pos_ratio=pos_ratio)
            user_jobs_map[uid] = df_chosen
            original_predictions.extend(original_prediction)
    
    return {
        "df_users_sample": df_users_sample,
        "current_user_ids": current_user_ids,
        "user_jobs_map": user_jobs_map,
        "original_predictions": original_predictions,
        "df_jobs": df_jobs 
    }
    

def run_full_analysis(
    prepared_data: dict,
    llm_function,
    threshold,
    sensitive_attr_dict, 
    cancel_event
):
    df_users_sample = prepared_data['df_users_sample']
    current_user_ids = prepared_data['current_user_ids']
    user_jobs_map = prepared_data['user_jobs_map']
    original_predictions = prepared_data['original_predictions']
    df_jobs = prepared_data['df_jobs']
    all_predictions = []
    attr_dict = sensitive_attr_dict
    k = len(attr_dict)

    for m in range(1, k+1):
        subsets = get_subsets_of_attributes(attr_dict, m)

        for subset in subsets:
            combos = generate_combinations_for_subset(attr_dict, subset)

            for uid in current_user_ids:
                user_row = df_users_sample[df_users_sample["UserID"] == uid].iloc[0]
                user_profile = {
                    "Education": user_row.get("DegreeType", ""),
                    "Major": user_row.get("Major", ""),
                    "GraduationDate": user_row.get("GraduationDate", ""),
                    "Work Experience": user_row.get("WorkHistoryCount", ""),
                    "Work Years": user_row.get("TotalYearsExperience", ""),
                    "Work History": user_row.get("JobTitle", "")
                }
                df_chosen_jobs = user_jobs_map[uid]

                _, all_predict = compute_sensitivity_after_averaging(uid, user_profile, df_chosen_jobs, combos, llm_function, cancel_event)
                all_predictions.extend(all_predict)

    if all_predictions:
        single_model_results = analyze_llm_performance(
            all_predictions=all_predictions,
            original_predictions=original_predictions,
            df_jobs=df_jobs,
            threshold=threshold
        )
    
    return single_model_results
