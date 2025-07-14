import pandas as pd
import random
from itertools import combinations, product
from typing import Dict, List, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.ticker as mticker
from config import PROMPT_TEMPLATE
import streamlit as st
class AnalysisCancelledError(Exception):
    pass

seed = 42
random.seed(seed)

def load_users(user_path: str) -> pd.DataFrame:
    df_users = pd.read_csv(user_path, sep='\t')
    return df_users

def load_jobs(jobs_path: str) -> pd.DataFrame:
    df_jobs = pd.read_csv(jobs_path, sep='\t')
    return df_jobs

def load_apps(apps_path: str) -> pd.DataFrame:
    df_apps = pd.read_csv(apps_path, sep='\t')
    return df_apps

def sample_users(df_users: pd.DataFrame, num_users: int) -> pd.DataFrame:
    if num_users >= len(df_users):
        return df_users
    return df_users.sample(n=num_users, random_state=seed)

def cluster_users(df_users: pd.DataFrame) -> pd.DataFrame:
    text_columns = [col for col in df_users.columns if df_users[col].dtype == 'object']
    if not text_columns:
        raise ValueError("DataFrame has no text columns for clustering.")
    df_users['combined_text'] = df_users[text_columns].fillna('').agg(' '.join, axis=1)

    tfu = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
    tfidfu_matrix = tfu.fit_transform(df_users['combined_text'])
    num_clusters = 14
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidfu_matrix)
    df_users.loc[:, "Cluster"] = kmeans.labels_
    return df_users

def cluster_data(df_data, n_keywords, industries, similarity_threshold, n_clusters=12, cancel_event=None):
    text_columns = [col for col in df_data.columns if df_data[col].dtype == 'object']
    if not text_columns:
        raise ValueError("data has no text columns for clustering.")
    df_data['combined_text'] = df_data[text_columns].fillna('').agg(' '.join, axis=1)

    if cancel_event.is_set():
        raise AnalysisCancelledError("Analysis cancelled during clustering step 1.")

    tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2), min_df=0.0, max_features=600000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_data['combined_text'])

    if cancel_event.is_set():
        raise AnalysisCancelledError("Analysis cancelled during clustering step 2.")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    df_data.loc[:, "Cluster"] = kmeans.labels_

    terms = tfidf.get_feature_names_out()
    cluster_keywords = {}
    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        top_indices = cluster_center.argsort()[-n_keywords:][::-1]
        cluster_keywords[i] = [terms[j] for j in top_indices]
    # return cluster_keywords

    industry_df = pd.DataFrame(list(industries.items()), columns=['Industry', 'Description'])
    all_texts = [" ".join(keywords) for keywords in cluster_keywords.values()] + list(industry_df['Description'])
    tfidfaaa = TfidfVectorizer(analyzer='word',ngram_range=(1, 2), min_df=0.0, max_features=600000, stop_words='english')
    tfidf_matrixaaa = tfidfaaa.fit_transform(all_texts)
    
    cluster_vectors = tfidf_matrixaaa[:len(cluster_keywords)]
    industry_vectors = tfidf_matrixaaa[len(cluster_keywords):]
    assert cluster_vectors.shape[1] == industry_vectors.shape[1], "TF-IDF vectors must have the same number of features."

    similarity_matrix = cosine_similarity(cluster_vectors, industry_vectors)
    
    industry_mapping = {}
    for i, row in enumerate(similarity_matrix):
        max_similarity = row.max()
        if max_similarity >= similarity_threshold:
            industry_mapping[i] = industry_df.iloc[row.argmax()]['Industry']
        else:
            industry_mapping[i] = "Unclassified"
    
    df_data.loc[:, 'Industry'] = df_data['Cluster'].map(industry_mapping)
    return df_data

def sample_users_by_cluster(df_users: pd.DataFrame, num_users: int, seed: int = 42) -> pd.DataFrame:

    clusters = df_users["Industry"].unique()
    n_clusters = len(clusters)
    
    if num_users >= len(df_users):
        return df_users

    base_count = num_users // n_clusters
    remainder = num_users % n_clusters
    
    target_counts = {cluster: base_count for cluster in clusters}
    random_state = np.random.RandomState(seed)
    if remainder:
        extra_clusters = random_state.choice(clusters, size=remainder, replace=False)
        for cluster in extra_clusters:
            target_counts[cluster] += 1

    sampled_dict = {}
    for cluster in clusters:
        cluster_users = df_users[df_users["Industry"] == cluster]
        n_sample = min(target_counts[cluster], len(cluster_users))
        sampled_dict[cluster] = cluster_users.sample(n=n_sample, random_state=random_state)
    
    total_sampled = sum(len(df) for df in sampled_dict.values())
    
    while total_sampled < num_users:
        added = False
        for cluster in clusters:
            cluster_users = df_users[df_users["Industry"] == cluster]
            current_sampled = sampled_dict[cluster]
            remaining = cluster_users.drop(current_sampled.index)
            if not remaining.empty:
                extra_sample = remaining.sample(n=1, random_state=random_state)
                sampled_dict[cluster] = pd.concat([current_sampled, extra_sample])
                total_sampled += 1
                added = True
                if total_sampled == num_users:
                    break
        if not added:
            break

    return pd.concat(sampled_dict.values())

def sample_jobs_for_user(
    df_jobs: pd.DataFrame,
    df_apps: pd.DataFrame,
    user_id: Any,
    industry: str,
    N: int,
    pos_ratio: float = 0.5
) -> pd.DataFrame:

    random.seed(42)

    original_predictions = []
    user_apps = df_apps[df_apps["UserID"] == user_id]["JobID"].unique()
    # user_industries = df_jobs[df_apps["JobID"].isin(user_apps)]['Industry']
    # print(industry)

    same_industry_jobs = df_jobs[df_jobs["Industry"] == industry]
    all_jobs = same_industry_jobs["JobID"].unique()
    # all_jobs = df_jobs["JobID"].unique()

    pos_jobs = list(set(user_apps).intersection(all_jobs))  
    neg_jobs = list(set(all_jobs) - set(pos_jobs))   

    num_pos = int(N * pos_ratio)
    num_pos = min(num_pos, len(pos_jobs)) 
    num_neg = N - num_pos
    num_neg = min(num_neg, len(neg_jobs))

    chosen_pos = random.sample(pos_jobs, num_pos) if len(pos_jobs) >= num_pos else pos_jobs
    chosen_neg = random.sample(neg_jobs, num_neg) if len(neg_jobs) >= num_neg else neg_jobs

    chosen_jobs = chosen_pos + chosen_neg
    
    df_chosen = df_jobs[df_jobs["JobID"].isin(chosen_jobs)].copy()
    df_chosen.loc[:, "label"] = 0
    df_chosen.loc[df_chosen["JobID"].isin(chosen_pos), "label"] = 1
    for job in chosen_jobs:
        original_predictions.append({
            "user_id": int(user_id),
            "job_id": int(job),
            "label": int(job in chosen_pos)
        })
    return df_chosen, original_predictions

def sample_jobs_for_user_simi(
    df_jobs: pd.DataFrame,
    df_users: pd.DataFrame,
    user_id: Any,
    N: int,
    pos_ratio: float = 0.5,
    seed: int = 42
):
    random.seed(seed)
    original_predictions = []
    user_industry = df_users[df_users["UserID"] == user_id]["Industry"].iloc[0]

    industry_jobs = df_jobs[df_jobs["Industry"] == user_industry]
    
    if industry_jobs.empty:
        industry_jobs = df_jobs

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, max_features=600000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(industry_jobs["combined_text"])

    user_vector = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, max_features=600000, stop_words="english").fit_transform(
        df_users[df_users["UserID"] == user_id]["combined_text"]
    )
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    ranked_indices = similarity_scores.argsort()[::-1]
    num_pos = min(int(N * pos_ratio), len(ranked_indices))
    num_neg = N - num_pos

    chosen_pos_indices = ranked_indices[:num_pos]
    chosen_pos = industry_jobs.iloc[chosen_pos_indices]["JobID"].tolist()

    all_jobs_in_industry = industry_jobs["JobID"].unique()
    neg_jobs = list(set(all_jobs_in_industry) - set(chosen_pos))
    chosen_neg = random.sample(neg_jobs, min(num_neg, len(neg_jobs)))

    chosen_jobs = chosen_pos + chosen_neg
    df_chosen = industry_jobs[industry_jobs["JobID"].isin(chosen_jobs)]
    df_chosen["label"] = 0
    df_chosen.loc[df_chosen["JobID"].isin(chosen_pos), "label"] = 1

    for job in chosen_jobs:
        original_predictions.append({
            "user_id": int(user_id),
            "job_id": int(job),
            "label": int(job in chosen_pos)
        })

    return df_chosen, original_predictions

def load_sensitive_attributes() -> Dict[str, List[Any]]:
    return {
        "Gender": ["Male", "Female"],
        "Age": ["<35", ">35"]
        # "race": ["Black", "White", "Asian"]
    }

def get_subsets_of_attributes(attr_dict: Dict[str, List[Any]], m: int) -> List[List[str]]:
    keys = list(attr_dict.keys())
    subsets = list(combinations(keys, m))
    return [list(s) for s in subsets]

def generate_combinations_for_subset(
    attr_dict: Dict[str, List[Any]],
    subset: List[str]
) -> List[Dict[str, Any]]:
    selected_attrs = {k: attr_dict[k] for k in subset}
    keys = list(selected_attrs.keys())
    value_lists = [selected_attrs[k] for k in keys]

    combos = []
    for vals in product(*value_lists):
        combo = {}
        for k, v in zip(keys, vals):
            combo[k] = v
        combos.append(combo)
    return combos

def design_prompt(user_profile: Dict[str, Any], job_info: Dict[str, Any]) -> str:
    up_str = "; ".join(f"{k}:{v}" for k, v in user_profile.items())
    jb_str = "; ".join(f"{k}:{v}" for k, v in job_info.items())
    prompt = f"""{PROMPT_TEMPLATE}
### Input:
Candidate Profile: {up_str}
Job Description: {jb_str}

### Output:
"""
    return prompt

def predict_binary(prompt: str, llm_function, cancel_event) -> int: 
    if cancel_event.is_set():
        raise AnalysisCancelledError("Cancelled during LLM call.")
    output = llm_function(prompt)
    if output is None:
        raise ValueError("LLM API returned None, indicating an error or no response.")
    resp = output.get("score", 0.0)
    print("score", resp)
    return resp

def inject_attributes(
    original_profile: Dict[str, Any],
    attrs: Dict[str, Any]
) -> Dict[str, Any]:
    new_profile = original_profile.copy()
    for k, v in attrs.items():
        new_profile[k] = v
    return new_profile

def adaptive_sampling(
    user_sensitivity_scores: Dict[Any, float],
    current_user_ids: List[Any],
    ratio: float,
    threshold: float
) -> List[Any]:
    high_group = []
    low_group = []
    for uid in current_user_ids:
        score = user_sensitivity_scores.get(uid, 0.0)
        if score >= threshold:
            high_group.append(uid)
        else:
            low_group.append(uid)

    sample_high = random.sample(high_group, int(len(high_group)*ratio)) if len(high_group) > 0 else []
    sample_low  = random.sample(low_group,  int(len(low_group)*ratio))  if len(low_group) > 0  else []

    return sample_high + sample_low

def average_prediction_for_combo(
    uid,
    user_profile: Dict[str, Any],
    df_jobs_for_user: pd.DataFrame,
    combo: Dict[str, Any],
    llm_function,
    cancel_event
) -> float:
    all_predictions = []
    if df_jobs_for_user.empty:
        return 0.0
    preds = []
    injected_profile = inject_attributes(user_profile, combo)
    for _, row in df_jobs_for_user.iterrows():
        if cancel_event.is_set():
            raise AnalysisCancelledError("Analysis cancelled by user.")
        job_info = {
            "Position Titile": row.get("Title",""),
            "Description": row.get("Description",""),
            "Requirements": row.get("Requirements","")
        }
        # print(f"--> Sending request for UID: {uid}, JOBID: {row["JobID"]}")
        prompt = design_prompt(injected_profile, job_info)
        pred = predict_binary(prompt, llm_function, cancel_event)
        preds.append(pred)
        all_predictions.append({
            "user_id": uid,
            "job_id": row["JobID"],
            "combo": combo,
            "prediction": pred
        })
        if cancel_event.is_set():
            raise AnalysisCancelledError("Analysis cancelled by user.")
    return sum(preds)/(len(preds)*10), all_predictions

def analyze_llm_performance(
    all_predictions: List[Dict],
    original_predictions: List[Dict],
    df_jobs: pd.DataFrame,
    threshold: float
) -> Dict[str, Any]:
    if not all_predictions:
        return {}


    spd_by_attr = calc_spd(all_predictions, threshold)
    eo_by_attr = calc_eo(all_predictions, original_predictions, threshold)
    ppv_diff_by_attr = calc_ppv_diff(all_predictions, original_predictions, threshold)
    avg_scores = calculate_avg_scores_by_sensitive_group(all_predictions)

    sensitive_levels = {}
    for keys, value_dict in avg_scores.items():
        if not value_dict: continue
        max_diff = max(value_dict.values()) - min(value_dict.values())
        if max_diff >= 6:
            sensitive_levels[keys] = 3
        elif max_diff >= 3:
            sensitive_levels[keys] = 2
        else:
            sensitive_levels[keys] = 1
    
    df_preds_with_industry = add_industry_to_predictions(all_predictions, df_jobs)
    fairness_by_industry = calc_fairness_by_industry(df_preds_with_industry, original_predictions, threshold)
    avg_scores_by_industry = calculate_industry_avg_scores(all_predictions, df_jobs)
    # print("Fairness by industry:", fairness_by_industry)


    if not spd_by_attr:
        most_biased_attribute_tuple = tuple()
    else:
        most_biased_attribute_tuple = max(spd_by_attr, key=spd_by_attr.get)
    
    most_biased_attribute_str = ", ".join(k.capitalize() for k in most_biased_attribute_tuple)

    biased_industries = set()
    max_spd_in_industry = -1.0
    
    if most_biased_attribute_str:
        for industry, attr_metrics in fairness_by_industry.items():
            if most_biased_attribute_str in attr_metrics:
                current_spd = attr_metrics[most_biased_attribute_str]['SPD']
                if current_spd > max_spd_in_industry:
                    max_spd_in_industry = current_spd
                    biased_industries = {industry}
                elif current_spd == max_spd_in_industry:
                    biased_industries.add(industry)

    overview_bias_data = {}
    all_attributes = {
        combo_type
        for industry_metrics in fairness_by_industry.values()
        for combo_type in industry_metrics.keys()
    }

    for attr_name in all_attributes:
        max_spd = -1.0
        most_biased_industries_for_attr = []
        for industry, metrics in fairness_by_industry.items():
            if attr_name in metrics:
                current_spd = metrics[attr_name]['SPD']
                if current_spd > max_spd:
                    max_spd = current_spd
                    most_biased_industries_for_attr = [industry]
                # elif current_spd == max_spd and max_spd > 0:
                #     most_biased_industries_for_attr.append(industry)
        if most_biased_industries_for_attr:
            overview_bias_data[attr_name] = sorted(list(set(most_biased_industries_for_attr)))

    overview_metrics_data = {
            "SPD": max(spd_by_attr.values()) if spd_by_attr else 0.0,
            "EO": max(eo_by_attr.values()) if eo_by_attr else 0.0,
            "PPV_diff": max(ppv_diff_by_attr.values()) if ppv_diff_by_attr else 0.0,
            "Biased Attribute": most_biased_attribute_str,
            "Biased Industry": ", ".join(sorted(list(biased_industries)))
        }

    return {
        "all_predictions": all_predictions,
        "spd": spd_by_attr,
        "eo": eo_by_attr,
        "ppv_diff": ppv_diff_by_attr,
        "avg_scores": avg_scores,
        "sensitive_levels": sensitive_levels,
        "fairness_results_by_industry": fairness_by_industry,
        "avg_scores_by_industry": avg_scores_by_industry,
        "overview_metrics": overview_metrics_data,
        "overview_bias": overview_bias_data
    }

def calculate_avg_scores_by_sensitive_group(all_predictions: List[Dict]) -> Dict[str, Dict[str, float]]:
    df = pd.DataFrame(all_predictions)
    
    df['combo_keys'] = df['combo'].apply(lambda x: tuple(sorted(k.capitalize() for k in x.keys())))
    
    avg_scores = {}
    
    for keys, group_df in df.groupby('combo_keys'):
        group_df = group_df.copy()
        
        group_df['combo_values'] = group_df['combo'].apply(lambda d: tuple(d[k] for k in keys))
        
        avg_scores[keys] = group_df.groupby('combo_values')['prediction'].mean().to_dict()
    
    return avg_scores

def calculate_industry_avg_scores(
    all_predictions: List[Dict],
    df_jobs: pd.DataFrame
) -> Dict[str, Dict[tuple, Dict[tuple, float]]]:

    df = pd.DataFrame(all_predictions)
    df = df.rename(columns={"job_id": "JobID"})
    df = df.merge(df_jobs[['JobID', 'Industry']], on='JobID', how='left')
    
    df['combo'] = df['combo'].apply(
        lambda d: {k.lower(): v for k, v in d.items()}
    )
    
    df['combo_keys'] = df['combo'].apply(
        lambda x: tuple(sorted(k.capitalize() for k in x.keys()))
    )
    
    industry_scores = {}
    for industry, industry_df in df.groupby('Industry'):
        industry_scores[industry] = {}
        
        for keys, key_group in industry_df.groupby('combo_keys'):
            key_group = key_group.copy()
            
            key_group['combo_values'] = key_group['combo'].apply(
                lambda d: tuple(d[k.lower()] for k in keys) 
            )
            
            avg_scores = key_group.groupby('combo_values')['prediction'].mean().to_dict()
            industry_scores[industry][keys] = avg_scores
    
    return industry_scores

def compute_sensitivity_after_averaging(
    uid,
    user_profile: Dict[str, Any],
    df_jobs_for_user: pd.DataFrame,
    combos: List[Dict[str, Any]], 
    llm_function,
    cancel_event
) -> float:

    all_predictions = []
    if not combos:
        return 0.0
    combo_avg_preds = []
    for c in combos:
        if cancel_event.is_set():
            raise AnalysisCancelledError("Analysis cancelled by user.")
        avg_pred, all_prediction = average_prediction_for_combo(uid, user_profile, df_jobs_for_user, c, llm_function, cancel_event)
        combo_avg_preds.append(avg_pred)
        all_predictions.extend(all_prediction)
        # time.sleep(1)

    max_diff = 0.0
    n = len(combo_avg_preds)
    for i in range(n):
        for j in range(i+1, n):
            diff = abs(combo_avg_preds[i] - combo_avg_preds[j])
            if diff > max_diff:
                max_diff = diff
    return max_diff, all_predictions

def partition_users_by_ratio(
    user_scores: Dict[Any, float],
    ratio_high: float
) -> Tuple[List[Any], List[Any]]:
    sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
    cutoff = int(len(sorted_users) * ratio_high)
    high_part = sorted_users[:cutoff]
    low_part  = sorted_users[cutoff:]

    high_group = [u[0] for u in high_part]
    low_group  = [u[0] for u in low_part]
    return high_group, low_group



def calc_spd(all_predictions: List[Dict], threshold: float = 5.0) -> Dict[str, float]:

    df = pd.DataFrame(all_predictions)
    
    df['combo_keys'] = df['combo'].apply(lambda x: tuple(sorted(x.keys())))
    
    spd_results = {}
    
    for keys, group_df in df.groupby('combo_keys'):
        group_df = group_df.copy()

        group_df['combo_values'] = group_df['combo'].apply(lambda d: tuple(d[k] for k in keys))
        
        group_positive_rates = group_df.groupby('combo_values')['prediction'].apply(
            lambda x: (x >= threshold).mean()
        )
        
        if len(group_positive_rates) <= 1:
            spd = 0.0
        else:
            rates = group_positive_rates.values
            spd = max(abs(p1 - p2) for p1 in rates for p2 in rates)
        
        spd_results[keys] = spd
        
    return spd_results

def calc_eo(
    all_predictions: List[Dict],
    original_predictions: List[Dict],
    threshold: float = 5.0
) -> Dict[str, float]:

    df_pred = pd.DataFrame(all_predictions)
    df_orig = pd.DataFrame(original_predictions)
    
    df_combined = pd.merge(
        df_pred, df_orig, on=["user_id", "job_id"], how="inner"
    )
    
    df_combined['combo_keys'] = df_combined['combo'].apply(lambda x: tuple(sorted(x.keys())))
    
    eo_results = {}
    
    for keys, group_df in df_combined.groupby('combo_keys'):
        group_df = group_df.copy()
        group_df['combo_values'] = group_df['combo'].apply(lambda d: tuple(d[k] for k in keys))
        
        group_tprs = group_df.groupby('combo_values').apply(
            lambda sub_df: (
                ((sub_df['prediction'] >= threshold) & (sub_df['label'] == 1)).sum() /
                (sub_df['label'] == 1).sum()
                if (sub_df['label'] == 1).sum() > 0 else 0
            )
        )
        
        if len(group_tprs) <= 1:
            eo = 0.0
        else:
            rates = group_tprs.values
            eo = max(abs(tpr1 - tpr2) for tpr1 in rates for tpr2 in rates)
        
        eo_results[keys] = eo
        
    return eo_results

def calc_ppv_diff(
    all_predictions: List[Dict],
    original_predictions: List[Dict],
    threshold: float = 5.0
) -> Dict[str, float]:

    df_pred = pd.DataFrame(all_predictions)
    df_orig = pd.DataFrame(original_predictions)
    
    df_combined = pd.merge(
        df_pred, df_orig, on=["user_id", "job_id"], how="inner"
    )
    
    df_combined['combo_keys'] = df_combined['combo'].apply(lambda x: tuple(sorted(x.keys())))
    
    ppv_diff_results = {}
    
    for keys, group_df in df_combined.groupby('combo_keys'):
        group_df = group_df.copy()
        group_df['combo_values'] = group_df['combo'].apply(lambda d: tuple(d[k] for k in keys))
        
        group_ppvs = group_df.groupby('combo_values').apply(
            lambda sub_df: (
                ((sub_df['prediction'] >= threshold) & (sub_df['label'] == 1)).sum() /
                (sub_df['prediction'] >= threshold).sum()
                if (sub_df['prediction'] >= threshold).sum() > 0 else 0
            )
        )
        
        if len(group_ppvs) <= 1:
            ppv_diff = 0.0
        else:
            rates = group_ppvs.values
            ppv_diff = max(abs(r1 - r2) for r1 in rates for r2 in rates)
        
        ppv_diff_results[keys] = ppv_diff
        
    return ppv_diff_results

def generate_wordcloud(text: str, title: str = None):

    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        collocations=False,  
        max_words=200
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=16)
    plt.show()

@st.cache_data
def create_score_difference_chart(subgroup_labels, sensitivity_scores, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(subgroup_labels, sensitivity_scores, color='lightblue', width=0.4)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=20, fontname='DejaVu Sans')
    
    ax.set_title(title, fontsize=24, fontname='DejaVu Sans')
    ax.set_xlabel(xlabel, fontsize=24, fontname='DejaVu Sans')
    ax.set_ylabel(ylabel, fontsize=24, fontname='DejaVu Sans')
    if sensitivity_scores:
        ax.set_ylim(0, max(sensitivity_scores) + 1)
    plt.xticks(rotation=45, ha="right", fontsize=24)
    plt.tight_layout()
    return fig

@st.cache_data
def generate_wordcloud_figure(text: str):
    custom_phrases_to_filter = {
        "manager", "Manager", "Assistant", "assistant", "Management", 
        "service", "worker", "sales", "customer", 
        "assistant manager", "customer service", "administrative assistant"
    }

    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        stop_words="english"
    )
    X = vectorizer.fit_transform([text])
    freqs = X.toarray().sum(axis=0)
    vocab = vectorizer.get_feature_names_out()
    ngram_freq_dict = {
        ngram: freq for ngram, freq in zip(vocab, freqs)
        if ngram not in custom_phrases_to_filter
    }

    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        max_words=200
    ).generate_from_frequencies(ngram_freq_dict)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Work History", fontsize=18, fontweight='bold')

    fig.tight_layout(pad=1.5)

    return fig

@st.cache_data
def get_education_pie_chart(df_filtered):
    edu_counts = df_filtered["DegreeType"].dropna().value_counts()
    colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3'][:len(edu_counts)]

    fig, ax = plt.subplots(figsize=(8, 2))
    wedges, texts, autotexts = ax.pie(
        edu_counts,
        labels=edu_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        radius=1.0, 
        colors=colors,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 6}
    )
    ax.set_title("Education Distribution", fontsize=6, fontweight='bold')
    fig.tight_layout(pad=1.5)  
    return fig

@st.cache_data
def get_major_bar_chart(df_filtered, top_n=3):
    major_counts = df_filtered["Industry"].dropna().value_counts()
    top_majors = major_counts[:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_majors.index[::-1], top_majors.values[::-1])
    ax.set_title("Top Related Industries", fontsize=20, fontweight='bold')
    ax.set_xlabel("Number of Users", fontsize=14)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    return fig


def extract_top_phrases(texts, top_k=10):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(texts)
    tfidf_scores = X.mean(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    ranked = sorted(zip(vocab, tfidf_scores), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, score in ranked[:top_k]]

def calc_fairness_by_industry(
    df_preds: pd.DataFrame,
    original_predictions: List[Dict],
    threshold: float = 5.0
) -> Dict:

    df_orig = pd.DataFrame(original_predictions)
    df_preds = df_preds.rename(columns={"JobID": "job_id"})
    df_combined = pd.merge(df_preds, df_orig, on=["user_id", "job_id"], how="inner")
    
    df_combined['combo_type'] = df_combined['combo'].apply(
        lambda x: ", ".join(sorted([k.capitalize() for k in x.keys()]))
    )

    results_dict = {}
    
    for (industry, combo_type), group in df_combined.groupby(["Industry", "combo_type"]):
        subgroup_dict = {}
        for sg_name, sg_data in group.groupby(group['combo'].apply(str)):
            pred_pos_mask = sg_data['prediction'] >= threshold
            label_pos_mask = sg_data['label'] == 1
            
            true_positives = (pred_pos_mask & label_pos_mask).sum()
            predicted_positives = pred_pos_mask.sum()
            actual_positives = label_pos_mask.sum()
            
            subgroup_dict[sg_name] = {
                "positive_rate": pred_pos_mask.mean(),
                "tpr": (true_positives / actual_positives) if actual_positives > 0 else 0.0,
                "ppv": (true_positives / predicted_positives) if predicted_positives > 0 else 0.0
            }
        
        metrics = {"SPD": 0.0, "EO": 0.0, "PPV_diff": 0.0}
        if len(subgroup_dict) >= 2:
            values = list(subgroup_dict.values())
            metrics["SPD"] = max(abs(a["positive_rate"] - b["positive_rate"]) for a in values for b in values)
            metrics["EO"] = max(abs(a["tpr"] - b["tpr"]) for a in values for b in values)
            metrics["PPV_diff"] = max(abs(a["ppv"] - b["ppv"]) for a in values for b in values)
        
        if industry not in results_dict:
            results_dict[industry] = {}
        results_dict[industry][combo_type] = metrics
    
    return results_dict


def calc_fairness_by_cluster(
    df_preds: pd.DataFrame,
    original_predictions: List[Dict],
    threshold: float = 5.0
):

    df_orig = pd.DataFrame(original_predictions)

    df_combined = pd.merge(
        df_preds, df_orig, on=["user_id", "job_id"], how="inner"
    )

    cluster_groups = df_combined.groupby("Cluster")

    results = []
    for cluster, group in cluster_groups:
        sensitive_groups = group.groupby(group["combo"].apply(str))

        group_positive_rates = {
            group_name: (subgroup["prediction"] >= threshold).mean()
            for group_name, subgroup in sensitive_groups
        }
        spd = max(abs(p1 - p2) for p1 in group_positive_rates.values() for p2 in group_positive_rates.values())

        group_tprs = {
            group_name: ((subgroup["prediction"] >= threshold) & (subgroup["label"] == 1)).sum() /
                        (subgroup["label"] == 1).sum() if (subgroup["label"] == 1).sum() > 0 else 0
            for group_name, subgroup in sensitive_groups
        }
        eo = max(abs(tpr1 - tpr2) for tpr1 in group_tprs.values() for tpr2 in group_tprs.values())

        group_ppvs = {
            group_name: ((subgroup["prediction"] >= threshold) & (subgroup["label"] == 1)).sum() /
                        (subgroup["prediction"] >= threshold).sum() if (subgroup["prediction"] >= threshold).sum() > 0 else 0
            for group_name, subgroup in sensitive_groups
        }
        ppv_diff = max(abs(ppv1 - ppv2) for ppv1 in group_ppvs.values() for ppv2 in group_ppvs.values())

        results.append({
            "Cluster": cluster,
            "SPD": spd,
            "EO": eo,
            "PPV_diff": ppv_diff
        })

    return pd.DataFrame(results)


def add_industry_to_predictions(all_predictions: List[Dict], jobs: List[Dict]) -> pd.DataFrame:

    df_jobs = pd.DataFrame(jobs)  
    df_preds = pd.DataFrame(all_predictions)
    df_preds = df_preds.rename(columns={"job_id": "JobID"})

    df_preds = pd.merge(df_preds, df_jobs[["JobID", "Industry"]], on="JobID", how="left")
    return df_preds

def add_cluster_to_predictions(
    all_predictions: List[Dict],
    users: List[Dict]
) -> pd.DataFrame:

    df_users = pd.DataFrame(users)  
    df_preds = pd.DataFrame(all_predictions)

    df_preds = pd.merge(df_preds, df_users[["UserID", "Cluster"]], left_on="user_id", right_on="UserID", how="left")
    return df_preds.drop(columns=["UserID"])  

def stringify_keys(d):
    return {str(k): v for k, v in d.items()}

def stringify_keys_recursively(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys_recursively(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_keys_recursively(elem) for elem in obj]
    else:
        return obj

# if __name__ == "__main__":
    # main()
    # cal()