# individual_page.py
import streamlit as st
import pathlib
import json
import threading
import time

from functions.individual import run_individual_analysis_threaded
from config import DATA_DIR, DEFAULT_INDIVIDUAL_SELECTION, DEFAULT_SENSITIVE_VALUES
from ui_pages.ui_components import (
    get_color,
    render_highlighted_attr,
    display_rank_as_html_table,
    extract_perturbations
)

def render_page(llm_function, demo_mode_i=False):
    # =========== Individual Analysis ===========
    if "analysis_status" not in st.session_state:
        st.session_state.analysis_status = "not_started"
    if "error_message" not in st.session_state:
        st.session_state.error_message = ""
    if "thread_results_ind" not in st.session_state:
        st.session_state.thread_results_ind = {}
    if "cancel_event" not in st.session_state:
        st.session_state.cancel_event = None
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None

    st.session_state.all_users_file = DATA_DIR / "users.tsv"
    colll = st.columns([4,8])
    with colll[0]:
        st.title("Individual Fairness")
    with colll[1]:
        st.markdown("    ")
        st.markdown("    ")
        st.markdown(
            """
            <style>
            .custom-title {
                font-size: 26px;
                font-family: 'DejaVu Sans', sans-serif;
                font-weight: bold;
                color: #2C3E50;
                padding-bottom: 5px;
                display: inline-block;
            }
            </style>
            <p class="custom-title">💡 Upload your personal details to uncover potential biases in your job recommendations.</p>
            """,
            unsafe_allow_html=True
        )

    st.header("⚙️ Settings")
    st.markdown("---")
    col1, col2, col3 = st.columns([1.2, 1.8, 1.5])

    with col1:
        st.subheader("1) User Profile")
        with st.container():
            user_education = st.text_input("Education", "Master of Science")
            user_major = st.text_input("Major", "Computer Science")
            user_workexperience = st.text_input("WorkExperience", "3 jobs")
            user_workyears = st.text_input("WorkYears", "15.0 years")
            user_workhistory = st.text_input("WorkHistory", "Lead Software Engineer at IBM for 10 years, specializing in C++ and Java systems. Took a 12-year career gap to raise a family. Recently completed an intensive AI & Machine Learning bootcamp from MIT (2024) and have multiple personal projects on GitHub using TensorFlow and PyTorch.")

    with col2:
        st.subheader("2) Job Description")
        with st.container():
            job_title = st.text_input("Position Title", "Senior AI/ML Engineer")
            job_desc = st.text_area("Position Description", height=120, value="""Join our disruptive, fast-paced AI startup as we revolutionize the industry. We're looking for a 'rockstar' developer and a true 'code ninja' to join our dynamic, 'work hard, play hard' team culture. You'll be working on cutting-edge models and need to thrive under pressure to meet aggressive deadlines.""")
            # jobb_req = st.text_area("Position Requirements", height=120, value="Bachelor's Degree in software engineering or equivalent. 5+ years of combined framework and application software development experience are required. A focused/driven attitude toward the support and development of mission critical systems. The candidate should have some exposure to investing/banking technology sector. Expert working knowledge of Microsoft .NET platform. Strong design and development skills in framework, GUI components and widgets in WPF are required. Experience with a number of the following technologies is strongly suggested: prism, Silverlight, Ibatis, MEF, XML, Soap Web service and Clearcase. Strong technical knowledge, excellent writing and communication skills, and experience in all phases of application development are required.")
            jobb_req = st.text_area("Position Requirements", height=120, value="""Master's or PhD in Computer Science. 5+ years of hands-on experience in Python, TensorFlow, and PyTorch. Must have experience building and deploying large-scale machine learning models from the ground up.""")

    with col3:
        st.subheader("3) Select sensitive attributes")
            
        with st.container():

            selected_attributes = {}
            for attribute, options_str in DEFAULT_SENSITIVE_VALUES.items():
                options = [
                    "\>35" if option.strip() == ">35" else option.strip()
                    for option in options_str.split(",")
                ] # streamlit > 35 is not supported, so we replace it with \>35
                options.insert(0, "not select")

                default_selection = DEFAULT_INDIVIDUAL_SELECTION.get(attribute, "not select")
                if default_selection in options:
                    default_index = options.index(default_selection)
                else:
                    default_index = 0

                col = st.columns([2, 9])
                with col[0]:
                    st.markdown("  ")
                    st.markdown("  ")
                    st.markdown(f"###### **{attribute}:**") #
                with col[1]:
                    selected_option = st.radio(" ", options, index=default_index, key=attribute, horizontal=True)
                        
                if selected_option != "not select":
                    if selected_option == "\>35":
                        selected_option = ">35"
                    selected_attributes[attribute] = selected_option
    
    co = st.columns([3, 10])
    with co[0]:
        candidate_num = st.number_input("Candidate numbers", 1, 5, 3, step=1)    

    if st.button("Evaluate"):
        if demo_mode_i:
            demo_path = pathlib.Path("data/individual_sample.json")
            with open(demo_path, "r", encoding="utf-8") as f:
                evaluation_result = json.load(f)

            st.session_state.evaluation_result  = evaluation_result
            st.session_state.selected_attributes = selected_attributes

            st.info(
                "Demo mode: a pre‑computed example is displayed; "
                "no live LLM calls were made.",
                icon="⚠️"
            )
        else:
            st.session_state.analysis_status = "running"
            st.session_state.thread_results_ind = {'status': 'running'}
            st.session_state.cancel_event = threading.Event()
            st.session_state.selected_attributes = selected_attributes

            args_for_thread = (
                st.session_state.thread_results_ind,
                user_education, user_major, user_workexperience, user_workyears, user_workhistory,
                candidate_num, st.session_state.all_users_file, llm_function,
                selected_attributes, job_title, job_desc, jobb_req,
                st.session_state.cancel_event
            )
            thread = threading.Thread(target=run_individual_analysis_threaded, args=args_for_thread)
            thread.start()
            st.rerun()

    current_status = st.session_state.analysis_status

    if current_status == "running":
        st.info("🔄 Analyzing in the background... This may take a few minutes depending on the number of attributes.")
        if st.button("Cancel Analysis"):
            if st.session_state.cancel_event:
                st.session_state.cancel_event.set()
            st.rerun()
        
        thread_status = st.session_state.thread_results_ind.get('status')
        if thread_status in ['done', 'error', 'cancelled']:
            if thread_status == 'done':
                st.session_state.analysis_status = 'done'
                st.session_state.evaluation_result = st.session_state.thread_results_ind.get('output')
            elif thread_status == 'error':
                st.session_state.analysis_status = 'error'
                st.session_state.error_message = st.session_state.thread_results_ind.get('error_message', 'Unknown error')
            elif thread_status == 'cancelled':
                st.session_state.analysis_status = 'cancelled'
            
            st.session_state.thread_results_ind = {}
            st.session_state.cancel_event = None
            st.rerun()
        else:
            with st.spinner("Processing..."):
                time.sleep(3)
            st.rerun()

    elif current_status == "cancelled":
        st.warning("Analysis was cancelled by the user.")
        if st.button("Reset"):
            st.session_state.analysis_status = "not_started"
            st.session_state.evaluation_result = None
            st.rerun()

    elif current_status == "error":
        st.error(f"An error occurred during analysis: {st.session_state.error_message}")
        if st.button("Reset"):
            st.session_state.analysis_status = "not_started"
            st.session_state.evaluation_result = None
            st.rerun()

    elif current_status == "done" and st.session_state.evaluation_result:
        result = st.session_state.evaluation_result
        selected_attributes = st.session_state.selected_attributes
        diff_data = result['diff_data']
        num_candidates = result['num_candidates']
        baseline_rank = result['baseline_rank']
            
        st.markdown("   ")
        st.header("📊 Evaluation Results")
        st.markdown("---")

        sensitivity_html = f"""
            <div style="text-align: right; font-size: 20px; font-weight: bold;">
                Sensitivity level: 
                <span style="background-color: {get_color(1)}; padding: 2px 6px; border-radius: 4px;">low</span> →
                <span style="background-color: {get_color(2)}; padding: 2px 6px; border-radius: 4px;">medium</span> →
                <span style="background-color: {get_color(3)}; padding: 2px 6px; border-radius: 4px;">high</span>
            </div>
        """
        st.markdown(sensitivity_html, unsafe_allow_html=True)

        diff_data = dict(sorted(diff_data.items()))

        for idx, (diff_count, items) in enumerate(diff_data.items()):
            groups = {}
            for item in items:
                key = item['perturbed_keys']
                groups.setdefault(key, []).append(item)
            sum_score_diff = {}
            max_rank_diff = {}
            for perturbed_keys, group_items in groups.items():
                sum_score_diff[perturbed_keys] = sum(abs(item['score_diff']) for item in group_items)
                max_rank_diff[perturbed_keys] = max(abs(item['rank_diff']) for item in group_items)
                
            sensitivity_level = {}
            for key, rank in max_rank_diff.items():
                if rank >= num_candidates * 0.6:
                    sensitivity_level[key] = 3
                elif rank >= num_candidates * 0.4:
                    sensitivity_level[key] = 2
                else:
                    sensitivity_level[key] = 1


            st.markdown(f''' 
                <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <span style="font-size: 24px; font-weight: bold; line-height: 30px;">Level {diff_count}:</span>
                    {'  '.join([render_highlighted_attr(key, sensitivity_level[key], 24) for key in groups.keys()])}
            ''', unsafe_allow_html=True)


            for perturbed_keys, group_items in groups.items():
                # st.markdown(render_highlighted_attr(perturbed_keys, sensitivity_level[perturbed_keys], 16))
                extrated_attr = extract_perturbations(selected_attributes, perturbed_keys)
                with st.expander(f"🏷️ {perturbed_keys} - **{extrated_attr}**", expanded=True):
                    cols = st.columns([5, 2])
                    with cols[0]:
                        display_rank_as_html_table(group_items, baseline_rank)
                    with cols[1]:
                        max_change = max_rank_diff[perturbed_keys]

                        # card_display("Max Absolute Rank Change", max_change)
                        st.markdown(f"""
                            <style>
                                .rank-card {{
                                    background-color: white;
                                    padding: 18px;
                                    border-radius: 12px;
                                    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
                                    text-align: center;
                                    font-size: 16px;
                                    font-weight: bold;
                                    color: #333;
                                    margin-top: 10px;
                                }}
                                .rank-value {{
                                    font-size: 30px;
                                    font-weight: bold;
                                    color: {'#333'};
                                }}
                                .rank-label {{
                                    font-size: 16px;
                                    color: #666;
                                    margin-bottom: 10px;
                                }}
                            </style>
                            <div class="rank-card">
                                <p class="rank-label">Max Absolute Rank Change</p>
                                <p class="rank-value">{max_change}</p>
                            </div>
                        """, unsafe_allow_html=True)
  