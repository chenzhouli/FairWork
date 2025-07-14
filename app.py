# app.py
import streamlit as st
import os
import argparse
from groq import Groq

from functions.llm_services import LLMProvider, AVAILABLE_LLMS
from ui_pages.individual_page import render_page as render_individual_page
from ui_pages.group_page import render_page as render_group_page

st.set_page_config(page_title="FairWork", layout="wide") 

@st.cache_resource
def get_llm_provider():
    return LLMProvider()

class App:
    def __init__(self, demo_mode_individual=False, demo_mode_group=False):
        self.demo_mode_i = demo_mode_individual
        self.demo_mode_g = demo_mode_group
        self.llm_provider = get_llm_provider()

    def run(self):
        # st.set_page_config(page_title="FairWork", layout="wide")

        st.sidebar.markdown(
            """<div style="font-size: 20px;">
                <span style="font-size: 29px; font-weight: bold;">FairWork</span> <br> Fairness Evaluation of LLM-based Job Recommender System <br> <br>
            </div>""",
            unsafe_allow_html=True
        )
        if not AVAILABLE_LLMS:
            st.error("LLM configuration is missing or empty in `config.yaml`. Please configure it first.")
            return

        selected_level = st.sidebar.radio('Choose analyse level', ("Individual Fairness", "Group Fairness"))
        llm_display_names = [llm['name'] for llm in AVAILABLE_LLMS]
        selected_llm_names = st.sidebar.multiselect('Choose LLM model', options=llm_display_names, default=[llm_display_names[0]] if llm_display_names else [])

        if not selected_llm_names:
            st.warning("Please select at least one LLM model to proceed.")
            return
        
        selected_configs = [llm for llm in AVAILABLE_LLMS if llm['name'] in selected_llm_names]
        
        
        if selected_level == "Individual Fairness":
            if len(selected_configs) > 1:
                st.warning("Please select only one model for Individual Fairness analysis.")
                return
            single_config = selected_configs[0]
            def unified_llm_call(prompt: str):
                return self.llm_provider.get_llm_response(single_config, prompt)
            
            render_individual_page(unified_llm_call, self.demo_mode_i)
 
            
        elif selected_level == "Group Fairness":
            render_group_page(self.llm_provider, selected_configs, self.demo_mode_g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FairWork Streamlit App.")
    parser.add_argument('--demo_individual', action='store_true', help='Run the Individual Fairness tab in demo mode.')
    parser.add_argument('--demo_group', action='store_true', help='Run the Group Fairness tab in demo mode.')
    args = parser.parse_args()

    app = App(demo_mode_individual=args.demo_individual, demo_mode_group=args.demo_group)
    app.run()
