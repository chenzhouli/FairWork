# ui_components.py
import streamlit as st
import streamlit.components.v1 as components
import re
from html import escape

def get_color(level):
    """
    Map sensitivity level (1-3) to a background color.
    The highest sensitivity (3) gets the darkest color.
    """
    if level == 1:
        return "#a3d9a5"  
    elif level == 2:
        return "#ffe066"  
    elif level == 3:
        return "#f5b7b1"
    else:
        return "#ffffff"  
    
def display_rank_as_html_table(data, baseline_rank, table_title=None):
    html = """
    <style>
        table.custom-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 16px;
            background-color: #ffffff;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        table.custom-table thead th {
            background-color: #f7f7f7;
            font-weight: bold;
            padding: 12px;
            text-align: center;
            font-size: 16px;
            border-bottom: 3px solid #ddd;
        }
        table.custom-table tbody td {
            padding: 12px;
            text-align: center;
            font-size: 15px;
            border-bottom: 1px solid #ddd;
        }
        table.custom-table tbody tr:hover {
            background-color: #f1f1f1;
        }
        .rank-up { color: green; font-weight: bold; }
        .rank-down { color: red; font-weight: bold; }
        .rank-same { color: gray; font-weight: bold; }
        .score-up { color: green; font-weight: bold; }
        .score-down { color: red; font-weight: bold; }
        .score-same { color: gray; font-weight: bold; }
        .perturbed {
            background-color: yellow;
            padding: 2px 4px;
            border-radius: 4px;
        }
        h2 {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
    </style>
    """

    if table_title:
        html += f"<h2>{table_title}</h2>"

    html += """
    <table class="custom-table">
        <thead>
            <tr>
                <th>Pertrubed Attributes</th>
                <th>Rank Change</th>
                <th>Score Change</th>
            </tr>
        </thead>
        <tbody>
    """

    for item in data:
        new_rank = item['new_rank']
        rank_change_class = "rank-same"
        score_change_class = "score-same"
        if new_rank > baseline_rank:
            rank_change_class = "rank-down" 
        elif new_rank < baseline_rank:
            rank_change_class = "rank-up" 
        if item['score_diff'] > 0:
            score_change_class = "score-up"
        elif item['score_diff'] < 0:
            score_change_class = "score-down"
        
        highlighted_text = extract_perturbations_table(item['combo_str'], item['perturbed_keys'])

        html += f"""
            <tr>
                <td>{highlighted_text}</td>
                <td class="{rank_change_class}">{baseline_rank} → {new_rank}</td>
                <td class="{score_change_class}">{item['new_score']-item['score_diff']} → {item['new_score']}</td>
            </tr>
        """
    html += """
        </tbody>
    </table>
    """

    components.html(html)

def highlight_perturbations(text, perturbed_keys):

    perturbed_keys_lists = [part.strip() for part in perturbed_keys.split(",")]
        
    for key in perturbed_keys_lists:
        pattern = rf"({key}:\s*[^,]*)"
        text = re.sub(pattern, r'<span class="perturbed">\1</span>', text)
    return text

def extract_perturbations_table(text, perturbed_keys):

    perturbed_keys_lists = [part.strip() for part in perturbed_keys.split(",")]
    
    result = []
    
    for key in perturbed_keys_lists:
        pattern = rf"({key}:\s*[^,]*)" 
        match = re.search(pattern, text)
        if match:
            result.append(match.group(0))  

    return ", ".join(result)

def extract_perturbations(selected_attributes, perturbed_keys):

    perturbed_keys_lists = [part.strip() for part in perturbed_keys.split(",")]

    attr = []
    for key in perturbed_keys_lists:
        attr.append(key + ": " + selected_attributes[key])
    
    return ", ".join(attr)
    
def render_highlighted_attr(attr, sensitivity, font_size):
    color = get_color(sensitivity)

    return f'''
    <span style="
        background-color: {color};
        padding: 5px 10px;
        margin: 4px;
        border-radius: 8px;
        font-size: {font_size}px;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.15);
        text-align: center;
        border: 1px solid {color};
        display: inline-block;
        white-space: nowrap;
    "> {attr} </span>
    '''

def card_display(name, value):
    if type(value) == str:
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
                    <p class="rank-label">{name}</p>
                    <p class="rank-value">{value}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
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
                    <p class="rank-label">{name}</p>
                    <p class="rank-value">{value:.4f}</p>
                </div>
            """, unsafe_allow_html=True)
    
def display_dict_as_html_table(data, Name, Value, table_title=None):

    html = """
    <style>
        table.custom-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 16px;
        }
        table.custom-table thead th {
            background-color: #f0f0f0;
            font-weight: bold;
            border: 3px solid #000000;
            padding: 10px;
            text-align: center;
        }
        table.custom-table tbody td {
            border: 3px solid #000000;
            padding: 10px;
            text-align: center;
        }
    </style>
    """

    if table_title:
        html += f"<h2>{table_title}</h2>"

    html += f"""
    <table class="custom-table">
        <thead>
            <tr>
                <th>{Name}</th>
                <th>{Value}</th>
            </tr>
        </thead>
        <tbody>
    """
    for key, value in data.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = value
        html += f"""
            <tr>
                <td>{key}</td>
                <td>{formatted_value}</td>
            </tr>
        """
    html += """
        </tbody>
    </table>
    """

    components.html(html)

def render_model_comparison_cards(metrics_dict: dict):
    import streamlit.components.v1 as components
    from html import escape

    rows = ["SPD", "EO", "PPV_diff", "Biased Attribute", "Biased Industry"]

    metric_names = {
        "SPD": "Statistical Parity Diff",
        "EO": "Equal Opportunity",
        "PPV_diff": "Predictive Parity Value Diff",
        "Biased Attribute": "Most Biased Attribute",
        "Biased Industry": "Most Biased Industry"
    }

    # Calculate dynamic height based on content
    base_height = 100  # Base padding and header
    metric_row_height = 68  # Height per metric row
    model_header_height = 38  # Model header height
    explanation_section_height = 200  # Approximate height for explanation cards
    padding_buffer = 50  # Extra buffer for safety
    
    # Calculate total content height
    total_metrics_height = len(rows) * metric_row_height
    total_height = base_height + model_header_height + total_metrics_height + explanation_section_height + padding_buffer
    
    # Adjust height based on number of models (for potential wrapping)
    num_models = len(metrics_dict)
    if num_models > 3:  # If many models, add extra height for potential wrapping
        total_height += 100

    indicators_html = "".join(
        f"<div style='padding: 14px 12px; font-size: 16px; font-weight: 500; color: #1E417D; height: 88px; "
        f"border-top: 1px solid #E0E0E0; display: flex; align-items: center; box-sizing: border-box;'>"
        f"{escape(metric_names[r])}</div>"
        for r in rows
    )

    model_cards_html = ""
    for model_name, metrics in metrics_dict.items():
        model_header = f"""
            <div style='padding: 14px 12px; font-size: 16px; font-weight: 700; color: #1E417D;
                        border-bottom: 1px solid #E0E0E0; text-align: center; background-color: #F2F7FC;
                        height: 48px; display: flex; align-items: center; justify-content: center;
                        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                        box-sizing: border-box;'>
                {escape(model_name)}
            </div>
        """
        score_rows = ""
        for r in rows:
            val = metrics[r]
            text = f"{val:.4f}" if isinstance(val, float) else escape(str(val))
            score_rows += f"""
                <div style='padding: 14px 12px; font-size: 14px; font-weight: 500;
                            text-align: center; color: #2A2A2A; font-variant-numeric: tabular-nums;
                            height: 88px; display:flex; align-items:center; justify-content:center;
                            border-top: 1px solid #E0E0E0; box-sizing: border-box;'>
                    {text}
                </div>
            """

        model_cards_html += f"""
        <div style='display: flex; flex-direction: column;
                    border-radius: 10px; background-color: #ffffff;
                    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08);
                    min-width: 130px; max-width: 200px; margin: 0 12px;
                    border: 1.5px solid #D6E1F0;'>
            {model_header}
            {score_rows}
        </div>
        """

    html = f"""
        <div style='padding: 20px; font-family: "Segoe UI", "Helvetica Neue", sans-serif; max-width: 100%; box-sizing: border-box; min-height: fit-content;'>
            <div style='background: linear-gradient(to bottom right, #F6FAFE, #ffffff);
                        padding: 24px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); box-sizing: border-box; min-height: fit-content;'>
                <div style='width: 100%; margin: 0 auto; box-sizing: border-box;'>

                    <p style="font-size: 24px; font-weight: 700; color: #1E417D; margin-bottom: 24px;">
                        📊 Model Comparison
                    </p>

                    <div style='display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px; box-sizing: border-box; min-height: fit-content;'>

                        <div style='flex: 2; display: flex; flex-direction: row; min-width: 0; box-sizing: border-box;'>
                            <div style='display: flex; flex-direction: column; width: 110px; flex-shrink: 0; box-sizing: border-box; margin-right: 15px;'>
                            <div style='height: 48px;'></div>
                                {indicators_html}
                            </div>
                            <div style='display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; flex: 1; min-width: 0; box-sizing: border-box;'>
                                {model_cards_html}
                            </div>
                        </div>

                        <div style='flex: 1; display: flex; flex-direction: column; min-width: 0; box-sizing: border-box;'>
                            <div style="background-color: #F2F7FC; border: 1px solid #D6E1F0; border-radius: 10px; padding: 16px; font-size: 16px; color: #1E417D; text-align: left; margin-bottom: 12px; box-sizing: border-box;">
                                <strong>Statistical Parity Diff (SPD)</strong><br>
                                Difference in positive outcome rates between groups.<br><em>Range: 0 ~ 1.</em> The closer to 0, the fairer.
                            </div>
                            <div style="background-color: #F2F7FC; border: 1px solid #D6E1F0; border-radius: 10px; padding: 16px; font-size: 16px; color: #1E417D; text-align: left; margin-bottom: 12px; box-sizing: border-box;">
                                <strong>Equal Opportunity (EO)</strong><br>
                                Difference in true positive rates between groups.<br><em>Range: 0 ~ 1.</em> The closer to 0, the fairer.
                            </div>
                            <div style="background-color: #F2F7FC; border: 1px solid #D6E1F0; border-radius: 10px; padding: 16px; font-size: 16px; color: #1E417D; text-align: left; margin-bottom: 12px; box-sizing: border-box;">
                                <strong>Predictive Parity Value Diff (PPV)</strong><br>
                                Precision difference across demographic groups.<br><em>Range: 0 ~ 1.</em> The closer to 0, the fairer.
                            </div>
                            <div style="background-color: #F2F7FC; border: 1px solid #D6E1F0; border-radius: 10px; padding: 16px; font-size: 16px; color: #1E417D; text-align: left; margin-bottom: 12px; box-sizing: border-box;">
                                <strong>Most Biased Attribute</strong><br>
                                Attributes (e.g., Age, Gender) with highest fairness disparity.
                            </div>
                            <div style="background-color: #F2F7FC; border: 1px solid #D6E1F0; border-radius: 10px; padding: 16px; font-size: 16px; color: #1E417D; text-align: left; box-sizing: border-box;">
                                <strong>Most Biased Industry</strong><br>
                                Industries (e.g., Sales, Tech) most affected by unfair treatment.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    """
    
    # Use calculated height instead of fixed 900
    components.html(html, height=total_height)

def render_structured_findings(model_bias_dict):
    import streamlit.components.v1 as components
    from html import escape

    def format_industries(industries):
        if len(industries) == 1:
            return f"<strong>{industries[0]}</strong>"
        else:
            return " and ".join([f"<strong>{ind}</strong>" for ind in industries])

    # Calculate dynamic height based on content
    base_height = 150  # Base padding and header
    model_block_height = 60  # Base height per model block
    items_per_model_height = 30  # Additional height per item in each model
    
    total_height = base_height
    
    model_blocks = ""
    for model_name, attr_to_inds in model_bias_dict.items():
        list_items = ""
        item_count = 0
        for attr, inds in attr_to_inds.items():
            industry_text = format_industries(inds)
            list_items += f"<li>In <strong>{attr}</strong>, {industry_text} industry show highest bias impact.</li>"
            item_count += 1
        
        # Add height for this model block
        total_height += model_block_height + (item_count * items_per_model_height)
        
        model_blocks += f"""
            <div style='margin-bottom: 20px;'>
                <p style="font-size: 18px; font-weight: 700; color: #1E417D;">🧠 {escape(model_name)}</p>
                <ul style='font-size: 18px; color: #2A2A2A; padding-left: 20px;'>
                    {list_items}
                </ul>
            </div>
        """

    html = f"""
    <div style='padding: 20px; font-family: "Segoe UI", "Helvetica Neue", sans-serif; min-height: fit-content;'>
        <div style='background: linear-gradient(to bottom right, #F6FAFE, #ffffff);
                    padding: 24px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); min-height: fit-content;'>

            <div style='max-width: 1000px; padding-left: 12px;'>

                <p style="font-size: 24px; font-weight: 700; color: #1E417D; margin-bottom: 24px;">
                    🔍 Key Fairness Findings per Model
                </p>
                {model_blocks}
            </div>
        </div>
    </div>
    """
    
    # Use calculated height instead of fixed formula
    components.html(html, height=total_height)
