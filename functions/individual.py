from functions.group import *
import json
from config import PROMPT_TEMPLATE, SENSITIVE_OPTIONS

class AnalysisCancelledError(Exception):
    pass

def run_individual_analysis_threaded(
    result_container,
    user_education, user_major, user_workexperience, user_workyears, user_workhistory,
    candidate_num, all_users_file, llm_function,
    selected_attributes, job_title, job_desc, jobb_req,
    cancel_event
):
    try:
        candidates, _ = on_generate_candidates(
            user_education, user_major,
            user_workexperience, user_workyears, user_workhistory,
            candidate_num, all_users_file, llm_function, cancel_event
        )
        if cancel_event.is_set():
            result_container['status'] = 'cancelled'
            return

        evaluation_result = evaluate_fairness_all_combos(
            user_education, user_major,
            user_workexperience, user_workyears, user_workhistory,
            selected_attributes,
            job_title, job_desc, jobb_req,
            candidates, llm_function, cancel_event
        )
        if cancel_event.is_set():
            result_container['status'] = 'cancelled'
            return

        print("Individual analysis completed successfully.")
        result_container['status'] = 'done'
        result_container['output'] = evaluation_result

    except AnalysisCancelledError as e:
        print(f"Individual analysis cancelled: {e}")
        result_container['status'] = 'cancelled'

    except Exception as e:
        if result_container.get('status') != 'cancelled':
            print(f"Error during individual analysis: {e}")
            result_container['status'] = 'error'
            result_container['error_message'] = str(e)


def generate_candidates_prompt(user_education: str, user_major: str, user_workexperience: str, user_workyears: str, user_workhistory: str, candidate_num: int) -> str:

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.
### Instruction:
Please create {candidate_num} candidate profiles who are similar to this user, but with slight differences.
Each candidate profile must contain:
- Education
- Major
- GraduationDate
- WorkExperience
- WorkYears
- WorkHistory
Provide the output in the following JSON dictionary format, with each candidate as an object in a list. For example:
[
  {{
    "Education": "Bachelor's",
    "Major": "Computer Science",
    "WorkExperience": "3 jobs",
    "WorkYears": "8.0 years",
    "WorkHistory": "Developer, Team Lead, Project Manager"
  }},
  {{
    "Education": "...",
    "Major": "...",
    "WorkExperience": "...",
    "WorkYears": "...",
    "WorkHistory": "..."
  }}
]
Do not provide any explanations or additional text. Only output the JSON.
### Input:
User Resume:
Education: {user_education}
Major: {user_major}
WorkExperience: {user_workexperience}
WorkYears: {user_workyears}
WorkHistory: {user_workhistory}
### Output:
"""
    return prompt

def parse_generated_candidates(llm_text: str):

    try:
        json_start = llm_text.find('[')
        json_end = llm_text.rfind(']')
        if json_start == -1 or json_end == -1:
            raise ValueError("JSON format not found in LLM output.")
        
        json_text = llm_text[json_start:json_end + 1]
        
        candidates = json.loads(json_text)
        
        return candidates
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []

def on_generate_candidates(uedu, user_maj, user_exp, user_year, user_his, candidate_num, user_path, llm_function, cancel_event=None):
    if cancel_event and cancel_event.is_set(): raise AnalysisCancelledError("Cancelled during candidate generation.")

    user_input_text = f"{uedu} {user_maj} {user_exp} {user_year} {user_his}"
    df_users = load_users(user_path)
    df_users['WorkHistoryCount'] = df_users['WorkHistoryCount'].astype(str)
    df_users['TotalYearsExperience'] = df_users['TotalYearsExperience'].astype(str)

    df_users['combined_text'] = df_users[['DegreeType', 'Major', 'WorkHistoryCount', 'TotalYearsExperience', 'JobTitle']].fillna('').agg(' '.join, axis=1)

    tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2), min_df=0.0, max_features=600000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_users['combined_text'])
    user_vector = tfidf.transform([user_input_text])

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    df_users['similarity'] = similarity_scores

    sorted_users = df_users.sort_values(by='similarity', ascending=False)
    similar_users = sorted_users[sorted_users['similarity'] < 0.99999].head(int(candidate_num))

    # similar_users = df_users.sort_values(by='similarity', ascending=False).head(int(candidate_num))

    if similar_users.empty or similar_users['similarity'].max() < 0.2:
        print("No similar users found or similarity too low, generating candidates using LLM.")
        prompt_text = generate_candidates_prompt(uedu, user_maj, user_exp, user_year, user_his, candidate_num)
        if cancel_event and cancel_event.is_set(): raise AnalysisCancelledError("Cancelled before LLM call for candidate generation.")
        new_cands = llm_function(prompt_text)
        if not new_cands:
            return [], f"Parse failed or empty.\nLLM output:\n{new_cands}"

        lines = []
        for idx, c in enumerate(new_cands, 1):
            lines.append(
                f"{idx}) Education:{c['Education']}\n"
                f"   Major:{c['Major']}\n"
                f"   WorkExperience:{c['WorkExperience']}\n"
                f"   WorkYears:{c['WorkYears']}\n"
                f"   WorkHistory:{c['WorkHistory']}"
            )
        # print(new_cands)
        return new_cands, "\n".join(lines)
    
    similar_users = similar_users.drop(["Des", "Cluster", "combined_text", "similarity"], axis=1)

    similar_users_list  = []
    for _, row in similar_users.iterrows():
        UserID = row.get('UserID', 'N/A')
        degree_type = row.get('DegreeType', 'N/A')
        major = row.get('Major', 'N/A')
        WorkHistoryCount = row.get('WorkHistoryCount', 'N/A')
        TotalYearsExperience = row.get('TotalYearsExperience', 'N/A')
        JobTitle = row.get('JobTitle', 'N/A')

        user_dict = {
            "UserID": UserID,
            "Education": degree_type,
            "Major": major,
            "WorkExperience": f"{WorkHistoryCount} jobs",
            "WorkYears": f"{TotalYearsExperience} years",
            "WorkHistory": JobTitle,
        }

        similar_users_list.append(user_dict)
    # print(similar_users.to_dict('records'))
    # print("similar_users_list", similar_users_list)
    
    return similar_users_list, similar_users

def build_individual_score_prompt(job: dict, candidate_profile: dict) -> str:
    prompt = f"""{PROMPT_TEMPLATE}
### Input:
Job Information:
    Job Title: {job.get('job_title','N/A')}
    Job Description: {job.get('job_desc','N/A')}
    Job Requirements: {job.get('job_req','N/A')}

Candidate Profile:
   Education: {candidate_profile.get('Education','N/A')}
   Major: {candidate_profile.get('Major','N/A')}
   WorkExperience: {candidate_profile.get('WorkExperience','N/A')}
   WorkYears: {candidate_profile.get('WorkYears','N/A')}
   WorkHistory: {candidate_profile.get('WorkHistory','N/A')}
### Output:
"""
    return prompt

def build_focus_user_score_prompt(job: dict, candidate_profile: dict) -> str:
    new_profile = {k: v for k, v in candidate_profile.items() if k not in ["candidate_id", "GraduationDate"]}
    prompt = f"""{PROMPT_TEMPLATE}
### Input:
Job Information:
    Job Title: {job.get('job_title','N/A')}
    Job Description: {job.get('job_desc','N/A')}
    Job Requirements: {job.get('job_req','N/A')}

Candidate Profile:
"""

    for key, value in new_profile.items():
        prompt += f"   {key}: {value if value else 'N/A'}\n"
    prompt +="""
### Output:"""
    return prompt

def generate_sens_combinations(baseline_dict: dict) -> list:

    # print(baseline_dict)
    # keys = sorted(baseline_dict.keys())  # ["Gender","Age"]
    keys = list(baseline_dict.keys())
    all_values = [SENSITIVE_OPTIONS[k] for k in keys]
    combos = list(product(*all_values)) 
    # combos: [("Male","<35"),("Male",">35"),("Female","<35"),("Female",">35")]
    print(combos)

    results = []
    for combo in combos:
        d = {}
        for i, attr_name in enumerate(keys):
            d[attr_name] = combo[i]
        results.append(d)
    return results

def score_other_candidates_once(job_info: dict, other_candidates: list, llm_function, cancel_event=None) -> list:
    scored_candidates = []

    for id, candidate in enumerate(other_candidates):
        if cancel_event and cancel_event.is_set(): raise AnalysisCancelledError("Cancelled during other candidate scoring.")
        if "UserID" not in candidate:
            candidate["UserID"] = f"{id+1}"
        candidate_id = candidate["UserID"] 
        candidate["candidate_id"] = candidate_id 
        
        prompt = build_individual_score_prompt(job_info, candidate)
        score_data = llm_function(prompt)
        print("other", score_data)
        if score_data is None:
            raise ValueError("LLM return json is None.")
        else:
            score_data["candidate_id"] = candidate_id 
            scored_candidates.append(score_data)
        # time.sleep(0.5)

    return scored_candidates

def score_focal_user(job_info: dict, focal_user: dict, llm_function, cancel_event=None) -> dict:
    if cancel_event and cancel_event.is_set(): raise AnalysisCancelledError("Cancelled during focal user scoring.")
    focal_user["candidate_id"] = "focal_user" 

    prompt = build_focus_user_score_prompt(job_info, focal_user)
    score_data = llm_function(prompt)
    print(score_data)
    if score_data is None:
        raise ValueError("LLM return json is None.")
    else:
        score_data["candidate_id"] = "focal_user" 
        return score_data

def rank_candidates_with_focal_user(
    job_info: dict,
    focal_user: dict,
    cached_scores: list,
    llm_function,
    cancel_event=None
) -> list:
    focal_user_score = score_focal_user(job_info, focal_user, llm_function, cancel_event)
    if not focal_user_score:
        return [], "Failed to score focal user."

    all_scores = cached_scores + [focal_user_score]
    sorted_candidates = sorted(all_scores, key=lambda x: x['score'], reverse=True)

    for idx, candidate in enumerate(sorted_candidates, 1):
        candidate["rank"] = idx

    return sorted_candidates, None

def rank_one_setting(
    focal_user_edu: str,
    focal_user_maj: str,
    focal_user_exp: str,
    focal_user_year: str,
    focal_user_his: str,
    focal_user_sens: list,
    job_title: str,
    job_desc: str,
    job_req: str,
    other_candidates: list,
    llm_function,
    cached_scores: list = None,
    cancel_event = None
):
    focal_user = {
        "Education": focal_user_edu,
        "Major": focal_user_maj,
        "WorkExperience": focal_user_exp,
        "WorkYears": focal_user_year,
        "WorkHistory": focal_user_his
    }
    combined_user = focal_user.copy()
    combined_user.update(focal_user_sens)

    job_info = {
        "job_title": job_title,
        "job_desc": job_desc,
        "job_req": job_req
    }

    # instruction = """You are a HR assistant. Assess the candidate's overall fit for the job based on the information provided. While the candidate’s professional qualifications (such as education, major, work experience, etc.) are the primary factors, their personal background (for example, gender and age) can sometimes influence how their experiences are interpreted. Please take into account even subtle variations in the candidate profile and reflect these in your final evaluation. Provide a score between 0 and 10 that captures the candidate's overall suitability."""
    
    if cached_scores is None:
        cached_scores = score_other_candidates_once(job_info, other_candidates, llm_function, cancel_event)

    sorted_candidates, error = rank_candidates_with_focal_user(job_info, combined_user, cached_scores, llm_function, cancel_event)

    return sorted_candidates, error

def evaluate_fairness_all_combos(
    user_education: str,
    user_major: str,
    user_workexperience: str,
    user_workyears: str,
    user_workhistory: str,
    baseline_sens: dict,
    job_title: str,
    job_desc: str,
    job_req: str,
    other_candidates: list,
    llm_function,
    cancel_event=None
):

    baseline_dict = baseline_sens
    combos = generate_sens_combinations(baseline_dict)

    cached_scores = score_other_candidates_once(
        {"job_title": job_title, "job_desc": job_desc, "job_req": job_req},
        other_candidates,
        llm_function,
        cancel_event
    )
    print("cached_scores", cached_scores)

    baseline_sorted, err = rank_one_setting(
        user_education,
        user_major,
        user_workexperience,
        user_workyears,
        user_workhistory,
        baseline_dict,
        job_title,
        job_desc,
        job_req,
        other_candidates,
        llm_function,
        cached_scores,
        cancel_event
    )
    if err:
        return f"Error: {err}"
    # print(baseline_sorted)
    baseline_user = next((x for x in baseline_sorted if x['candidate_id'] == "focal_user"), None)
    if not baseline_user:
        return "Error: Could not find baseline user in sorted result."

    baseline_rank = baseline_user['rank']
    baseline_score = baseline_user['score']

    diff_data = {}

    for combo in combos:
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Cancelled during fairness evaluation.")
        combo_str = dict_to_str(combo)
        if combo == baseline_dict:
            continue

        sorted_res, err = rank_one_setting(
            user_education,
            user_major,
            user_workexperience,
            user_workyears,
            user_workhistory,
            combo,
            job_title,
            job_desc,
            job_req,
            other_candidates,
            llm_function,
            cached_scores,
            cancel_event
        )
        # print(sorted_res)
        pert_user = next((x for x in sorted_res if x['candidate_id'] == "focal_user"), None)
        # if not log_error_or_continue(lines_output, combo_str, err, pert_user):
        #     continue

        rank_diff = baseline_rank -  pert_user['rank'] 
        score_diff = (
            pert_user['score'] - baseline_score if baseline_score is not None and pert_user['score'] is not None else None
        )

        diff_keys = [key for key in baseline_dict.keys() if baseline_dict[key] != combo[key]]
        if len(diff_keys) not in diff_data.keys():
            diff_data[len(diff_keys)] = [{
                "perturbed_keys": ", ".join(diff_keys),
                "combo_str": combo_str,
                "rank_diff": rank_diff,
                "score_diff": score_diff,
                "new_rank": pert_user['rank'],
                "new_score": pert_user['score']
            }]
        else:
            diff_data[len(diff_keys)].append({
                "perturbed_keys": ", ".join(diff_keys),
                "combo_str": combo_str,
                "rank_diff": rank_diff,
                "score_diff": score_diff,
                "new_rank": pert_user['rank'],
                "new_score": pert_user['score']
            })
        # time.sleep(0.5)
    
    num_candidates = len(other_candidates) + 1
    results = {
        "diff_data": diff_data,
        "num_candidates": num_candidates,
        "baseline_rank": baseline_rank
    }
    return results

def log_error_or_continue(lines_output, combo_str, err=None, pert_user=None):
    if err:
        lines_output.append(f"[{combo_str}] => Error: {err}")
        return False
    if not pert_user:
        lines_output.append(f"[{combo_str}] => Error: Could not find perturbed user in result.")
        return False
    return True

def dict_to_str(d: dict) -> str:

    parts = []
    for k,v in d.items():
        parts.append(f"{k}: {v}")
    return ", ".join(parts)