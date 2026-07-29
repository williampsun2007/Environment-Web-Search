'''
Pollution Event Web Search - Streamlit entry point.

This is the app's UI layer: sidebar (API key entry, model choice), the
Single Search tab, and the Batch Search tab. It holds no research logic or
Excel-building logic itself - it calls into search_pipeline.py to find, rank,
score, and synthesize an explanation for a pollutant spike, and into
excel_export.py to validate batch uploads and build the downloadable report.

Requires a DeepSeek API key (https://platform.deepseek.com/api_keys) and a
Serper API key (https://serper.dev), entered by the user in the sidebar at
runtime - never stored or logged.
'''

import streamlit as st
from openai import OpenAI
import datetime
import pandas as pd
import openpyxl
import search_pipeline
import excel_export

if "location" not in st.session_state:
    st.session_state.location = ""
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.date.today()
if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.date.today()
if "pollutant" not in st.session_state:
    st.session_state.pollutant = ""
if "sources" not in st.session_state:
    st.session_state.sources = {}
if "synthesis" not in st.session_state:
    st.session_state.synthesis = None
if "deepseek_key" not in st.session_state:
    st.session_state.deepseek_key = ""
if "serper_key" not in st.session_state:
    st.session_state.serper_key = ""
if "search_model" not in st.session_state:
    st.session_state.search_model = "deepseek-reasoner"
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "batch_excel" not in st.session_state:
    st.session_state.batch_excel = None
if "_pending_single" not in st.session_state:
    st.session_state._pending_single = False
if "_pending_batch" not in st.session_state:
    st.session_state._pending_batch = False
if "_batch_rows" not in st.session_state:
    st.session_state._batch_rows = []

# Display the results for a single search
def _display_single_results(sources, synthesis, start_date, end_date):
    st.subheader("Summary")
    if isinstance(synthesis, dict):
        st.info(f"**Most Likely Cause:** {synthesis.get('most_likely_cause', '')}")
        if synthesis.get("top_entity"):
            st.caption(f"Determined cause (entity): **{synthesis['top_entity']}**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Evidence**")
            for point in synthesis.get("key_evidence", []):
                st.markdown(f"- {point}")
        with col2:
            st.markdown("**Gaps & Alternatives**")
            for point in synthesis.get("gaps_and_alternatives", []):
                st.markdown(f"- {point}")
        confidence = synthesis.get("confidence", "")
        score = synthesis.get("score", 0)
        color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(confidence.upper(), "gray")
        st.markdown(
            f'<span style="color:{color};font-weight:bold">Confidence: {confidence}</span>'
            f' &nbsp;·&nbsp; <span style="font-size:1.1em"><b>{score}/100</b></span>',
            unsafe_allow_html = True
        )
        st.markdown(synthesis.get("confidence_rationale", ""))
    else:
        st.info(str(synthesis))

    groups = {}
    for source in sources.values():
        et = source.get("event_type", "Unknown Event")
        groups.setdefault(et, []).append(source)

    for event_type, group_sources in groups.items():
        st.header(event_type)
        gs = synthesis.get("group_scores", {}).get(event_type, {}) if isinstance(synthesis, dict) else {}
        gc = synthesis.get("group_confidence", {}).get(event_type, {}) if isinstance(synthesis, dict) else {}
        if gs:
            g_conf = gs.get("confidence", "")
            g_score = gs.get("score", 0)
            color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(g_conf.upper(), "gray")
            group_entities = gs.get("entities", [])
            is_single_entity = gs.get("is_single_entity", True)
            label = "Confidence this is the cause" if is_single_entity else \
                f"Category confidence (spans {len(group_entities)} entities — not a single cause)"
            st.markdown(
                f'<span style="color:{color};font-weight:bold">{label}: {g_conf}</span>'
                f' &nbsp;·&nbsp; <b>{g_score}/100</b>'
                f' — {gc.get("rationale", "")}',
                unsafe_allow_html = True
            )
            
            if not is_single_entity:
                top_entity = synthesis.get("top_entity", "") if isinstance(synthesis, dict) else ""
                for e in group_entities:
                    marker = " ⭐ (Determined most likely cause)" if e["entity"] == top_entity else ""
                    st.caption(
                        f'- {e["entity"]}: {e["score"]}/100 ({e["confidence"]}) '
                        f'· {e["source_count"]} source(s){marker}'
                    )
                    
            siblings = gs.get("sibling_event_types", [])
            if siblings:
                entity = gs.get("entity", "")
                combined = synthesis.get("entity_scores", {}).get(entity, {}) if isinstance(synthesis, dict) else {}
                st.caption(
                    f'Note: these sources concern the same underlying entity ("{entity}") as '
                    f'the {", ".join(siblings)} group(s) — combined confidence: '
                    f'{combined.get("score", "")}/100 ({combined.get("confidence", "")}).'
                )

        for i, source in enumerate(group_sources, 1):
            st.subheader(f"Source {i}: {source['title']}")
            st.write(f"**Type:** {source['type']}")
            if source.get("date"):
                date_str = source["date"]
                date_warning = " **(⚠ outside query window)**" if search_pipeline._source_in_window(date_str, start_date, end_date) is False else ""
                st.write(f"**Date:** {date_str}{date_warning}")
            st.write(f"**Summary:** {source['summary']}")
            st.write(f"**URL:** {source['url']}")
            st.divider()


# Sidebar for API keys
with st.sidebar:
    locked = st.session_state.is_running
    st.header("API Keys")
    st.text_input("DeepSeek API Key", type = "password", key = "deepseek_key", disabled = locked)
    st.caption("Get your key at [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)")
    st.text_input("Google Serper API Key", type = "password", key = "serper_key", disabled = locked)
    st.caption("Get your key at [serper.dev](https://serper.dev)")
    st.caption(
        "Keys are used only for this session and are sent directly to DeepSeek and Serper. "
        "They are never stored, logged, or routed through any intermediate server."
    )
    st.divider()
    st.header("Settings")
    model_choice = st.selectbox(
        "Search model",
        ["deepseek-reasoner (thorough)", "deepseek-chat (fast, cheaper)"],
        disabled = locked
    )
    st.session_state.search_model = model_choice.split()[0]


# Code for main page
st.title("Pollution Event Web Search")
tab1, tab2 = st.tabs(["Single Search", "Batch Search"])


# Code for Tab #1 - Single Search
with tab1:
    locked = st.session_state.is_running
    location = st.text_input("Location", key = "location", disabled = locked)
    start_date = st.date_input("Start Date", min_value = datetime.date(2000, 1, 1),
                               key = "start_date", disabled = locked)
    end_date = st.date_input("End Date", min_value = datetime.date(2000, 1, 1),
                             key = "end_date", disabled = locked)
    pollutant = st.text_input("Pollutant", placeholder = "e.g. PM2.5, ethylene, VOCs",
                              key = "pollutant", disabled = locked)

    dates_ok = start_date <= end_date
    if not dates_ok:
        st.error("Start Date must be on or before End Date.")

    keys_ok = bool(st.session_state.deepseek_key and st.session_state.serper_key)
    fields_ok = bool(location and pollutant) and dates_ok

    if st.button("Search", disabled = locked or not fields_ok, key = "btn_single"):
        if not keys_ok:
            st.error("Please enter both API keys in the sidebar before searching.")
        else:
            st.session_state.is_running = True
            st.session_state._pending_single = True
            st.session_state.sources = {}
            st.session_state.synthesis = None
            st.rerun()

    if st.session_state._pending_single:
        st.session_state._pending_single = False
        search_pipeline.client = OpenAI(api_key = st.session_state.deepseek_key, base_url = "https://api.deepseek.com")
        loc = st.session_state.location
        start = st.session_state.start_date
        end = st.session_state.end_date
        poll = st.session_state.pollutant
        
        try:
            with st.status("Searching...", expanded = True) as status:
                st.session_state.sources = search_pipeline.run_search_pipeline(
                    loc, start, end, poll, st.session_state.search_model
                )
                status.update(label = f"Done — found {len(st.session_state.sources)} sources",
                              state = "complete", expanded = False)

            with st.spinner("Synthesizing findings..."):
                st.session_state.synthesis = search_pipeline.synthesize_findings(
                    loc, start, end, poll, st.session_state.sources
                )
                
            if isinstance(st.session_state.synthesis, dict):
                for src_key, new_date in st.session_state.synthesis.get("date_corrections", {}).items():
                    if src_key in st.session_state.sources:
                        st.session_state.sources[src_key]["date"] = new_date
        except Exception as e:
            st.error(f"Search error: {e}")
        finally:
            st.session_state.is_running = False

    if st.session_state.sources and st.session_state.synthesis is not None:
        _display_single_results(
            st.session_state.sources,
            st.session_state.synthesis,
            st.session_state.start_date,
            st.session_state.end_date
        )
        
        if st.button("🔄 New Search (clears current results)"):
            st.session_state.sources = {}
            st.session_state.synthesis = None
            st.rerun()

# Code for Tab #2 - Batch Search
with tab2:
    locked = st.session_state.is_running
    st.markdown(
        "Upload an Excel file with columns: **Location**, **Start Date**, **End Date**, **Pollutant**. "
        "Results will be exported to a downloadable Excel file with a Summary sheet and a Sources sheet."
        " Columns must be spelled exactly how it is shown"
    )

    uploaded = st.file_uploader(
        "Upload batch file (.xlsx or .xls)", type = ["xlsx", "xls"],
        disabled = locked, key = "batch_file"
    )

    valid_rows, file_errors = [], []
    if uploaded is not None:
        try:
            df = pd.read_excel(uploaded)
            valid_rows, file_errors = excel_export.validate_batch_df(df)
        except Exception as e:
            file_errors = [f"Could not read file: {e}"]

        if file_errors:
            for err in file_errors:
                st.error(err)
        if valid_rows:
            n_valid = len(valid_rows)
            st.success(f"{n_valid} valid {'search' if n_valid == 1 else 'searches'} found.")
            if n_valid > 20:
                est_hours = max(1, round(n_valid * 3 / 60, 0))
                st.warning(
                    f"Each search typically takes 1-5 minutes."
                    f"{n_valid} searches may take approximately {est_hours}+ hours."
                )
            with st.expander("Preview searches"):
                st.dataframe(pd.DataFrame(valid_rows), use_container_width = True)

    keys_ok_batch = bool(st.session_state.deepseek_key and st.session_state.serper_key)
    can_run_batch = bool(valid_rows) and keys_ok_batch and not locked

    if st.button("Start Batch Search", disabled = not can_run_batch, key = "btn_batch"):
        st.session_state.is_running = True
        st.session_state._pending_batch = True
        st.session_state._batch_rows = valid_rows
        st.session_state.batch_excel = None
        st.rerun()

    # Persistent download button — shows partial or final results between reruns
    if st.session_state.batch_excel is not None and not st.session_state._pending_batch:
        st.download_button(
            "Download Results (.xlsx)",
            data = st.session_state.batch_excel,
            file_name = "batch_results.xlsx",
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key = "dl_persistent"
        )

    if st.session_state._pending_batch:
        st.session_state._pending_batch = False
        
        rows = st.session_state._batch_rows
        n = len(rows)
        search_pipeline.client = OpenAI(api_key = st.session_state.deepseek_key, base_url = "https://api.deepseek.com")

        wb = openpyxl.Workbook()
        summary_ws = wb.active
        summary_ws.title = "Summary"
        sources_ws = wb.create_sheet("Sources")

        summary_ws.append(["Search #", "Location", "Start Date", "End Date", "Pollutant",
                            "Score", "Confidence", "Most Likely Cause", "Source Count", "Error"])
        sources_ws.append(["Search #", "Location", "Pollutant", "Source #",
                            "Title", "URL", "Type", "Event Type", "Entity", "Date", "Summary"])

        excel_export._style_header(summary_ws)
        excel_export._style_header(sources_ws)

        st.session_state.batch_excel = excel_export._wb_to_bytes(wb)
        
        if st.button("⛔ Stop and Download Current Results"):
            st.rerun()
        
        progress_bar = st.progress(0, text = "Starting batch search...")
        status_text = st.empty()

        try:
            for i, row in enumerate(rows, 1):
                loc = row["location"]
                start = row["start_date"]
                end = row["end_date"]
                poll = row["pollutant"]

                progress_bar.progress(i / n, text = f"Search {i} / {n} — {loc} · {poll}")
                status_text.info(f"Processing search **{i}/{n}**: {loc}, {poll} ({start} to {end})")

                err_msg = ""
                sources = {}
                synthesis = {}
                try:
                    with st.expander(f"Search {i} / {n}: {loc} — {poll}", expanded = False):
                        sources = search_pipeline.run_search_pipeline(loc, start, end, poll, st.session_state.search_model)
                        st.write("Synthesizing...")
                        synthesis = search_pipeline.synthesize_findings(loc, start, end, poll, sources)
                except Exception as e:
                    err_msg = str(e)
                    st.error(f"Search {i} error: {e}")

                if isinstance(synthesis, dict):
                    score = synthesis.get("score", "")
                    confidence = synthesis.get("confidence", "")
                    cause = synthesis.get("most_likely_cause", "")
                    for src_key, new_date in synthesis.get("date_corrections", {}).items():
                        if src_key in sources:
                            sources[src_key]["date"] = new_date
                else:
                    score, confidence, cause = "", "", (str(synthesis) if synthesis else "")

                summary_ws.append([
                    i, loc, str(start), str(end), poll,
                    score, confidence, cause, len(sources), err_msg
                ])
                
                for j, src in enumerate(sources.values(), 1):
                    sources_ws.append([
                        i, loc, poll, j,
                        src.get("title", ""), src.get("url", ""),
                        src.get("type", ""), src.get("event_type", ""), src.get("entity", ""),
                        src.get("date", ""), src.get("summary", "")
                    ])

                st.session_state.batch_excel = excel_export._wb_to_bytes(wb)

        except Exception as catastrophic_e:
            st.error(f"Batch search stopped unexpectedly: {catastrophic_e}")
        finally:
            excel_export._autosize_columns(summary_ws)
            excel_export._autosize_columns(sources_ws)
            st.session_state.batch_excel = excel_export._wb_to_bytes(wb)
            st.session_state.is_running = False

        progress_bar.progress(1.0, text = "Batch complete!")
        status_text.success(f"All {n} searches complete.")
        st.download_button(
            "Download Full Results (.xlsx)",
            data = st.session_state.batch_excel,
            file_name = "batch_results.xlsx",
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key = "dl_final"
        )