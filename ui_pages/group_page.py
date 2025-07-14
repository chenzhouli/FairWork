# group_page.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import json
import time
import threading

from functions.analysis import run_full_analysis_threaded
from functions.group import *

from config import DATA_DIR, INDUSTRIES, DEFAULT_SENSITIVE_VALUES
from functions.data_utils import (
    load_overview_metrics,
    load_overview_bias,
    load_details,
    restore_keys_recursively
)
from ui_pages.ui_components import (
    render_model_comparison_cards,
    render_structured_findings,
    get_color,
    render_highlighted_attr,
    card_display
)

def render_page(llm_provider, selected_configs, demo_mode_g=False):
    # =========== Group-level Analysis ===========
    colll = st.columns([4,8])
    with colll[0]:
        st.title("Group Fairness")
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
            <p class="custom-title">💡 Upload your datasets for an in-depth evaluation of any potential biases in the system.</p>
            """,
            unsafe_allow_html=True
        )
            
    if "users_file" not in st.session_state:
        st.session_state.users_file = None
    if "jobs_file" not in st.session_state:
        st.session_state.jobs_file = None
    if "apps_file" not in st.session_state:
        st.session_state.apps_file = None
    if "default_used" not in st.session_state:
        st.session_state.default_used = False
    

    st.header("⚙️ Settings")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Upload data")
        col11, col12 = st.columns([1,2])
        with col11:
            if st.button("Use default dataset"):
                st.session_state.default_used = True
                # print(st.session_state.users_file)
        with col12:
            if st.session_state.get("default_used", False):
                st.markdown("**Using livecareer.com dataset.**")

        users_file = st.file_uploader("Upload Users Dataset (.tsv)", disabled=demo_mode_g)
        
        jobs_file = st.file_uploader("Upload Jobs Dataset (.tsv)", disabled=demo_mode_g)
        
        apps_file = st.file_uploader("Upload interaction Dataset (.tsv)", disabled=demo_mode_g)
        
        if not demo_mode_g:
            if st.session_state.default_used:
                st.session_state.users_file = DATA_DIR / "users_150.tsv"
                st.session_state.jobs_file = DATA_DIR / "jobs_150.tsv"
                st.session_state.apps_file = DATA_DIR / "apps.tsv"
            if users_file:
                st.session_state.users_file = users_file 
            if jobs_file:
                st.session_state.jobs_file = jobs_file
            if apps_file:
                st.session_state.apps_file = apps_file      
        else:
            st.session_state.users_file = DATA_DIR / "users_150.tsv"
            st.session_state.jobs_file = DATA_DIR / "jobs_150.tsv"
            st.warning("Demo mode: only example results are displayed, and the upload functionality is disabled.", icon="⚠️") 

    with col2:
        if "custom_attributes" not in st.session_state:
            st.session_state.custom_attributes = {}

        st.subheader("Add custom attributes")
        st.markdown("Please enter the attribute name and its corresponding values below (separate multiple values with commas):")

        custom_attr_name = st.text_input("Attribute name", placeholder="e.g., Education Level")
        custom_attr_values = st.text_input("Values (separate with commas)", placeholder="e.g., High School, Bachelor's, Master's")

        if st.button("Add custom attrbutes"):
            if custom_attr_name and custom_attr_values:
                st.session_state.custom_attributes[custom_attr_name] = custom_attr_values
                st.success(f"Added custom attrbutes{custom_attr_name}")
            else:
                st.warning("Please provide both attribute name and values.", icon="⚠️")

    # st.markdown("---")
    with col3:

        st.subheader("Select sensitive attributes")

        all_attributes = DEFAULT_SENSITIVE_VALUES.copy()
        all_attributes.update(st.session_state.custom_attributes)

        selected_attributes = {}
        for attr, values in all_attributes.items():
            if isinstance(values, str):
                options = [v.strip() for v in values.split(",")]
            else:
                options = values
            selected = st.multiselect(f"{attr}", options, default=options)
            if selected != []:
                selected_attributes[attr] = selected
        st.session_state.group_sensitive_attributes = selected_attributes
    
    st.subheader("Sample setting")

    colss = st.columns(4)
    with colss[0]:
        num_users = st.slider("Users to Sample", 1, 30, 30)
        st.markdown(f"👉 Number of users to be evaluated.")
    with colss[1]:
        n_jobs = st.slider("Jobs per User to Sample", 3, 5, 5)
        st.markdown(f"👉 Number of jobs per user to be evalutated.")
    with colss[2]:
        pos_ratio = st.slider("Positive Sample Ratio in Jobs", 0, 10, 4)
        st.markdown(f"👉 **{pos_ratio*10} percent** job samples per user are positive.")
    with colss[3]:
        # threshold = st.slider("Threshold score for Positive Prediction [e.g. >5.0 -> 1 (positive), <5.0 -> 0 (negative)]", 0.0, 10.0, 5.0)
        threshold = st.slider("Threshold Score", min_value=0.0, max_value=10.0, value=5.0)
        st.markdown(f"👉 Scores **above {threshold}** are **positive**, scores **below {threshold}** are **negative**.")

    if "analysis_status" not in st.session_state:
        st.session_state.analysis_status = "not_started"
    if "error_message" not in st.session_state:
        st.session_state.error_message = ""
    if "thread_results" not in st.session_state:
        st.session_state.thread_results = {}
    if "cancel_event" not in st.session_state:
        st.session_state.cancel_event = None
    if "data_generated" not in st.session_state:
        st.session_state.data_generated = False

    if st.button("Evaluate"):
        st.session_state.error_message = ""
        if "data_generated" in st.session_state:
            del st.session_state.data_generated
        
        if demo_mode_g:
            st.session_state.analysis_status = "done"
        else:
            st.session_state.analysis_status = "running"
            st.session_state.thread_results = {'status': 'running'}
            st.session_state.cancel_event = threading.Event()
            args_for_analysis = (
                st.session_state.thread_results, st.session_state.users_file, st.session_state.jobs_file, st.session_state.apps_file,
                num_users, n_jobs, pos_ratio, threshold,
                selected_attributes, llm_provider, selected_configs,
                st.session_state.cancel_event
            )
            thread = threading.Thread(
                target=run_full_analysis_threaded,
                args=args_for_analysis
            )
            thread.start()
        
        st.rerun()
    
    print(f"Analysis status: {st.session_state.analysis_status}")
    if st.session_state.analysis_status == "running":
        if st.button("Cancel Analysis"):
            if st.session_state.cancel_event:
                st.session_state.cancel_event.set()
            st.rerun()

    current_status = st.session_state.analysis_status

    if current_status == "running":
        # Check the status reported by the background thread
        thread_status = st.session_state.thread_results.get('status')
        
        if thread_status == 'done':
            st.session_state.analysis_status = 'done'
            results = st.session_state.thread_results.get('output', {})
            for key, value in results.items():
                st.session_state[key] = value
            st.session_state.data_generated = True 
            # Clean up session state
            st.session_state.thread_results = {}
            st.session_state.cancel_event = None
            st.rerun() 
        
        elif thread_status == 'error':
            st.session_state.analysis_status = 'error'
            st.session_state.error_message = st.session_state.thread_results.get('error_message', 'Unknown error')
            # Clean up session state
            st.session_state.thread_results = {}
            st.session_state.cancel_event = None
            st.rerun()

        elif thread_status == 'cancelled':
            st.session_state.analysis_status = 'cancelled'
            # Clean up session state
            st.session_state.thread_results = {}
            st.session_state.cancel_event = None
            st.rerun()

        else: # Status is still 'running'
            st.info("🔄 Analyzing in the background... This may take several minutes.")
            with st.spinner("Processing..."):
                time.sleep(3) # Poll every 3 seconds
            st.rerun()
    
    elif current_status == "cancelled":
        st.warning("Analysis was cancelled by the user.")
        if st.button("Reset"):
            st.session_state.analysis_status = "not_started"
            if "data_generated" in st.session_state:
                del st.session_state.data_generated
            st.rerun()

    elif current_status == "error":
        st.error(f"An error occurred during analysis: {st.session_state.error_message}")
        if st.button("Reset"):
            st.session_state.analysis_status = "not_started"
            if "data_generated" in st.session_state:
                del st.session_state.data_generated
            st.rerun()

    elif current_status == "done":
        st.session_state.selected_industry = None
        if demo_mode_g:
            attr_dict = {'Gender': ['Male', 'Female'], 'Age': ['<35', '>35']}

            threshold = 5.0

            st.header("📊 Evaluation Results")
            st.markdown("---")

            st.markdown("### 🔍 Fairness Analysis Summary")
            
            metrics_dict = load_overview_metrics()
            bias_dict = load_overview_bias()

            render_model_comparison_cards(metrics_dict)
            render_structured_findings(bias_dict)

            st.markdown("               ")
            st.markdown("               ")

            st.markdown("### 🤖 Please select the model for details:")
            selected_model = st.radio(
                label="  ",
                options=["Llama3-8b", "deepseek"],
                index=0,
                horizontal=True,
                key="selected_model"
            )

            original_path = DATA_DIR / selected_model / "original_predictions.json"
            all_path = DATA_DIR / selected_model / "all_predictions.json"

            with open(original_path, "r") as f:
                original_predictions = json.load(f)

            with open(all_path, "r") as f:
                all_predictions = json.load(f)

            details = load_details(selected_model)
            avg_scores = restore_keys_recursively(details["avg_scores"])
            avg_scores_by_industry = restore_keys_recursively(details["avg_scores_by_industry"])
            fairness_results_by_industry = restore_keys_recursively(details["fairness_results_by_industry"])
            spd, eo, ppv_diff = restore_keys_recursively(details["spd"]), restore_keys_recursively(details["eo"]), restore_keys_recursively(details["ppv_diff"])
        else:
            if not st.session_state.get("data_generated", False):
                    st.warning("Analysis process finished, but no data was generated. Please check the logs or try again.")
                    return
            
            st.header("📊 Evaluation Results")
            st.markdown("---")

            st.markdown("### 🔍 Fairness Analysis Summary")
            metrics_dict = st.session_state.overview_metrics
            bias_dict = st.session_state.overview_bias

            render_model_comparison_cards(metrics_dict)
            render_structured_findings(bias_dict)

            st.markdown("               ")
            st.markdown("               ")
            
            all_results = st.session_state.analysis_results
            model_names_with_results = list(all_results.keys())

            if len(model_names_with_results) > 1:
                st.markdown("### 🤖 Please select the model for details:")
                selected_model = st.radio(
                    label="  ",
                    options=model_names_with_results,
                    index=0,
                    horizontal=True,
                    key="selected_model"
                )
            elif model_names_with_results:
                selected_model = model_names_with_results[0]
            else:
                st.warning("Analysis returned no results.")
                return

            results_for_model = all_results.get(selected_model)
            if not results_for_model:
                st.warning(f"Results for the selected model '{selected_model}' are not available.", icon="⚠️")
                

            if results_for_model:
                spd = results_for_model['spd']
                eo = results_for_model['eo']
                ppv_diff = results_for_model['ppv_diff']
                avg_scores = results_for_model['avg_scores']
                avg_scores_by_industry = results_for_model['avg_scores_by_industry']
                fairness_results_by_industry = results_for_model['fairness_results_by_industry']
                attr_dict = st.session_state.attr_dict
                original_predictions = st.session_state.original_predictions
                all_predictions = results_for_model['all_predictions']   


        sensitive_levels = {}
                
        for keys, value_dict in avg_scores.items():
            max_diff = max(value_dict.values()) - min(value_dict.values())
            if max_diff >=0.1:
                sensitive_levels[keys] = 3
            elif max_diff >=0.05:
                sensitive_levels[keys] = 2
            else:
                sensitive_levels[keys] = 1

        st.markdown("               ")
        st.markdown("               ")

        sensitivity_html = f"""
                <div style="text-align: right; font-size: 20px; font-weight: bold;">
                    Sensitivity level: 
                    <span style="background-color: {get_color(1)}; padding: 2px 6px; border-radius: 4px;">low</span> →
                    <span style="background-color: {get_color(2)}; padding: 2px 6px; border-radius: 4px;">medium</span> →
                    <span style="background-color: {get_color(3)}; padding: 2px 6px; border-radius: 4px;">high</span>
                </div>
            """
        st.markdown(sensitivity_html, unsafe_allow_html=True)

        for m in range(1, len(attr_dict)+1):
            subsets = get_subsets_of_attributes(attr_dict, m)
            st.markdown(f''' 
                    <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <span style="font-size: 24px; font-weight: bold; line-height: 30px;">Level {m}:</span>
                        {'  '.join([render_highlighted_attr(", ".join(key), sensitive_levels[tuple(sorted(key))], 24) for key in subsets])}
                ''', unsafe_allow_html=True)
            for subset in subsets:
                a = ", ".join(subset)
                with st.expander(f"🏷️ {a}", expanded=True):
                    co1, _, co2 = st.columns([5,1,6])
                    with co1:
                        st.subheader("⚖️ Fairness Metrics")
                        cols = st.columns(3)
                        with cols[0]:
                            # print(type(spd[tuple(sorted(subset))]))
                            card_display("SPD", spd[tuple(sorted(subset))])

                        with cols[1]:
                            card_display("EO", eo[tuple(sorted(subset))])

                        with cols[2]:
                            card_display("PPV_diff", ppv_diff[tuple(sorted(subset))])

                        # st.markdown("---")
                    with co2:
                        st.subheader("📊 Prediction Scores Differences")

                        cols2 = st.columns(2)
                        with cols2[0]:
                            subgroups = avg_scores[tuple(sorted(subset))]
                            subgroup_labels = list(subgroups.keys())
                            subgroup_labels = [", ".join(list(label)) for label in subgroup_labels]
                            sensitivity_scores = list(subgroups.values())
                            ttt = ", ".join(sorted(subset))

                            fig = create_score_difference_chart(
                                subgroup_labels, 
                                sensitivity_scores,
                                title=f"Average Scores: {ttt}",
                                xlabel=f"{ttt} Subgroups",
                                ylabel="Prediction Score"
                            )
                            st.pyplot(fig)

                        with cols2[1]:
                            att = max(avg_scores[tuple(sorted(subset))], key=avg_scores[tuple(sorted(subset))].get)
                            card_display("✅ The model demonstrates favor in", f"""🎯 {", ".join(map(str, att))}""")
                    
                    st.markdown("---")
                    st.subheader("🏢 Industry-specific Detail")
                    industry_key = f"selected_industry_{a.replace(' ', '_')}"
                        
                    if industry_key not in st.session_state:
                        st.session_state[industry_key] = list(INDUSTRIES.keys())[0]
                    selected_industry = st.selectbox(
                        "Please select industry",
                        INDUSTRIES,
                        index=INDUSTRIES.index(st.session_state.selected_industry) if st.session_state.selected_industry else 0,
                        help="select the industry to be evaluated",
                        key=f"selectbox_{a}", 
                    )
                    st.session_state[industry_key] = selected_industry 

                    industry_data = fairness_results_by_industry.get(selected_industry, {})
                    industry_attrubutes_data = industry_data.get(", ".join(sorted(subset)), {})
                        
                    if not industry_attrubutes_data:
                        st.warning(f"Industry [{selected_industry}] no valid data")
                    # print(industry_attrubutes_data)

                    if industry_attrubutes_data != {}:
                        coo = st.columns([5,1,6])
                        with coo[0]:
                            st.markdown("#### ⚖️ Fairness Metrics")
                            colsss = st.columns(3)
                            with colsss[0]:
                                card_display("SPD", industry_attrubutes_data['SPD'])

                            with colsss[1]:
                                card_display("EO", industry_attrubutes_data['EO'])

                            with colsss[2]:
                                card_display("PPV_diff", industry_attrubutes_data['PPV_diff'])

                        avg_scores_by_industry_select = avg_scores_by_industry[selected_industry]
                        with coo[2]:
                            st.markdown("#### 📊 Prediction Scores Differences")

                            cols2 = st.columns(2)
                            with cols2[0]:
                                subgroups = avg_scores_by_industry_select[tuple(sorted(subset))]
                                subgroup_labels = list(subgroups.keys())
                                subgroup_labels = [", ".join(list(label)) for label in subgroup_labels]
                                sensitivity_scores = list(subgroups.values())
                                ttt = ", ".join(sorted(subset))

                                fig = create_score_difference_chart(
                                    subgroup_labels, 
                                    sensitivity_scores,
                                    title=f"Average Scores: {ttt}",
                                    xlabel=f"{ttt} Subgroups",
                                    ylabel="Prediction Score"
                                )
                                st.pyplot(fig)
                            with cols2[1]:

                                att = max(avg_scores_by_industry_select[tuple(sorted(subset))], key=avg_scores_by_industry_select[tuple(sorted(subset))].get)
                                card_display("✅ The model demonstrates favor in", f"""🎯 {", ".join(list(att))}""")

                    st.markdown("---")
                    st.subheader("🎭 High-Sensitive Groups Background")
                    co = st.columns(3)
                    # with co[0]:
                    #     st.image(wordcloud_images[", ".join(subset)], use_container_width=True)

                    df_pred = pd.DataFrame(all_predictions)

                    df_users = load_users(st.session_state.users_file)

                    def get_subset_key(combo_dict, subset_attrs):
                        subset_set = set(subset_attrs)
                        combo_keys = set(combo_dict.keys())
                        
                        if combo_keys != subset_set:
                            return None
                        
                        try:
                            values = [combo_dict[attr] for attr in subset_attrs]
                            return tuple(values)
                        except KeyError:
                            return None

                    df_pred["subset_key"] = df_pred["combo"].apply(lambda c: get_subset_key(c, subset))
                    df_pred_filtered = df_pred.dropna(subset=["subset_key"])

                    grouped = df_pred_filtered.groupby(["user_id", "subset_key"])["prediction"].mean().reset_index(name="avg_prediction")

                    grouped_diff = grouped.groupby("user_id")["avg_prediction"].agg(["min","max"])
                    grouped_diff["diff"] = grouped_diff["max"] - grouped_diff["min"]
                    grouped_diff = grouped_diff.reset_index()
                    user_scores = grouped_diff[["user_id", "diff"]].rename(columns={"diff":"score"})

                    user_scores_dict = dict(zip(user_scores["user_id"], user_scores["score"]))

                    high_group, low_group = partition_users_by_ratio(user_scores_dict, ratio_high=0.4)

                    df_high_users = df_users[df_users["UserID"].isin(high_group)].copy()
                    education_img = get_education_pie_chart(df_high_users)
                    major_img = get_major_bar_chart(df_high_users)
                    high_user_background = df_users.loc[
                        df_users["UserID"].isin(high_group),
                        "JobTitle"
                    ]
                    # high_sensitivity_text = " ".join(high_user_background.dropna().tolist())
                    # wordcloud_img = generate_wordcloud_figure(high_sensitivity_text)

                    # with co[0]:
                    #     st.pyplot(education_img)
                    # with co[1]:
                    #     st.pyplot(wordcloud_img)
                    # with co[2]:
                    #     st.pyplot(major_img)
                    high_sensitivity_text = " ".join(high_user_background.dropna().tolist())

                    if high_sensitivity_text.strip():
                        wordcloud_img = generate_wordcloud_figure(high_sensitivity_text)
                        
                        with co[0]:
                            st.pyplot(education_img)
                        with co[1]:
                            st.pyplot(wordcloud_img)
                        with co[2]:
                            st.pyplot(major_img)
                    else:
                        with co[0]:
                            st.pyplot(education_img)
                        with co[1]:
                            st.warning("No enough words!")
                        with co[2]:
                            st.pyplot(major_img)

        download_file_path = None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        download_file_path = tmp.name
        with open(download_file_path, "w", encoding="utf-8") as f:
            json.dump({
                    "all_predictions": all_predictions,
                    "original_predictions": original_predictions
            }, f, indent=4, ensure_ascii=False)
        tmp.close()

        if download_file_path:
            st.subheader("📥 Download Predictions")
            st.markdown("You can download the analyzed prediction results in JSON format for further analysis.")
                    
            with open(download_file_path, "rb") as f:
                st.download_button("📂 Download Predictions", data=f, file_name="predictions.json", mime="application/json")
