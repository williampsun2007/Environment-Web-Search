import streamlit as st
import json
from openai import OpenAI, AuthenticationError as OpenAIAuthError
import re
import requests
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pypdf import PdfReader
import io
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill

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

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": f'''
                Search the internet for information about events like wildfires, industrial accidents,
                or other disasters that could cause pollutant spikes.
                ''',
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def search_web(query):
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": st.session_state.serper_key,
                "Content-Type": "application/json"
            },
            json = {"q": query},
            timeout = 20
        )
        response.raise_for_status()
        results = response.json().get("organic", [])
    except requests.exceptions.HTTPError:
        if response.status_code in (401, 403):
            raise RuntimeError(f"Invalid Serper API key (HTTP {response.status_code}). Please check your key in the sidebar.")
        return []
    except (requests.exceptions.RequestException, ValueError):
        return []

    simplified = []
    for r in results:
        title = r.get("title", "")
        url = r.get("link", "")
        snippet = r.get("snippet", "")
        if title and url:  # Skip malformed entries missing essential fields
            simplified.append({"title": title, "url": url, "snippet": snippet})

    return simplified

def searching_llm(location, start_date, end_date, pollutant, existing_sources, searched_queries = None, model = "deepseek-reasoner"):
    if searched_queries is None:
        searched_queries = set()

    if start_date.year == end_date.year:
        search_year = str(start_date.year)
    elif end_date.year - start_date.year == 1:
        search_year = f"{start_date.year} OR {end_date.year}"
    else:
        search_year = f"{start_date.year}–{end_date.year}"

    already_searched_block = ""
    if searched_queries:
        already_searched_block = f"""
                ALREADY-SEARCHED QUERIES — DO NOT REPEAT THESE:
                The following exact queries were run in previous rounds. Skip them entirely and
                do not rerun them. Use different phrasings, sites, or angles instead:
                {json.dumps(sorted(searched_queries), indent = 16)}"""

    messages = [
        {
            "role": "user",
            "content": f'''
                A {pollutant} spike was detected near {location} between {start_date} and {end_date}.

                Here are the sources already found so far:
                {json.dumps(existing_sources)}

                Your job is to search the web for ANY ADDITIONAL events that could explain this {pollutant} spike, such as:
                - Wildfires or prescribed burns
                - Industrial accidents or chemical spills
                - Power plant emissions
                - Any other unusual event that could release {pollutant} or its precursors

                Do not search for or return sources already in the list above.
                {already_searched_block}

                YOU MUST FOLLOW THIS EXACT SEARCH STRATEGY IN ORDER:

                PHASE 1 — GOVERNMENT SOURCES (search these first, before anything else):
                Run site-targeted searches against official government domains. Use the location and
                YEAR(S) ONLY (not the full date) in each query — e.g. "Tulsa Oklahoma 2021", not
                "Tulsa, Oklahoma 2021-03-10". At minimum, attempt ALL of the following:
                  - site:airnow.gov {location} {search_year}  (EPA AirNow — most important for air quality)
                  - site:aqs.epa.gov {location} {search_year}  (EPA Air Quality System data)
                  - site:epa.gov {location} {pollutant} {search_year}
                  - site:epa.gov {location} {search_year}
                  - site:noaa.gov {location} {search_year}
                  - site:cdc.gov {location} {search_year}
                  - site:deq.[state-abbreviation].gov {location} {search_year}
                    (derive the 2-letter state abbreviation: Oklahoma → ok, Texas → tx, California → ca, etc.)
                  - site:csb.gov {location} {search_year}
                  - site:ntsb.gov {location} {search_year}
                  - site:nifc.gov {location} {search_year}  (wildfire incidents)

                NAMED-AGENCY FALLBACK (apply immediately after each Phase 1 query that returns 0 or very few results):
                Retry without the site: filter using the agency's full name. Examples:
                  - "Oklahoma Department of Environmental Quality" Tulsa 2021 air quality advisory
                  - "EPA Region 6" Tulsa Oklahoma 2021 pollution incident
                  - "[State] DEQ" OR "[State] Department of Environmental Quality" {location} {search_year}
                  - "EPA" {location} {pollutant} incident {search_year}
                This catches government documents hosted on press-release aggregators or archive pages
                that Google does not serve under the site: filter.

                PHASE 2 — ACADEMIC SOURCES (search these after government sources):
                  - site:pubmed.ncbi.nlm.nih.gov {location} {pollutant} {search_year}
                  - scholar.google.com {location} {pollutant} pollution {search_year}
                  - site:sciencedirect.com {location} {pollutant} {search_year}

                PHASE 3 — GENERAL FALLBACK (only after phases 1 and 2):
                Run broader queries for news and other sources only if government and academic
                searches returned insufficient results. Include {pollutant} in these queries where relevant.

                PERSISTENCE RULE:
                If any query returns few or no results, retry with alternative phrasings before
                moving on. Try angles such as: "air quality advisory", "incident report",
                "emergency response", "health alert", "emissions event".

                MINIMUM SEARCH REQUIREMENT:
                You MUST run at least 5 searches before returning any results, even if early
                searches look comprehensive. Do not return results after fewer than 5 searches.

                When done, return a JSON dictionary of ONLY the newly found sources
                (do NOT repeat any sources already in the list above), each with:
                - title
                - url
                - type: use one of these labels exactly — "Government Report", "Government Database",
                  "Academic Institution", "News Outlet", "Local News", "Wikipedia", "Social Media", "Other".
                  Use "Wikipedia" for Wikipedia.org pages (not "News Outlet").
                  Use "Academic Institution" for universities, research labs, hospital systems, and similar
                  non-governmental academic bodies (not "Government Report").
                  Use "Government Report" for a post published directly by an official government agency's
                  own account (e.g., a state environmental agency's Facebook or Twitter/X page announcing
                  an official advisory or statement) — NOT "Social Media". Reserve "Social Media" for posts
                  by private individuals, unofficial commentary, or discussion that the agency itself did
                  not publish.
                - summary (one sentence of what the source says)

                Do not write any explanatory text before or after the JSON.
                Do not acknowledge the task or describe what you are doing.
                Immediately begin searching using the search tool, and when done return ONLY the JSON.

                After you have finished all your searches, you MUST return a JSON dictionary of only the
                new sources found, with keys "Source_1", "Source_2", etc. Do not stop without returning
                this JSON. The JSON must be the last thing you output, with no text before or after it.
            '''
        }
    ]

    search_count = 0
    nudge_sent = False

    # Agentic loop: keep calling the model until it stops issuing tool calls and
    # returns the final JSON. Each iteration either executes search tool calls and
    # appends the results to the message history, or breaks out when the model
    # produces a plain text (JSON) response.
    while True:
        call_kwargs = {"model": model, "messages": messages}
        if not nudge_sent:
            call_kwargs["tools"] = tools
        try:
            response = client.chat.completions.create(**call_kwargs)
        except OpenAIAuthError:
            raise RuntimeError("Invalid DeepSeek API key. Please check your key in the sidebar.")
        except Exception as e:
            print(f"LLM API error in searching_llm: {e}")
            return {}

        text = response.choices[0].message

        if text.tool_calls:
            # content must be None (not omitted) when tool_calls are present per the API spec
            messages.append({"role": "assistant", "content": None, "tool_calls": text.tool_calls})

            for tool_call in text.tool_calls:
                query = json.loads(tool_call.function.arguments)["query"]
                searched_queries.add(query)
                print(f"Searching for: {query}")
                st.write(f"Searching: *{query}*")
                
                try: 
                    results = search_web(query)
                except RuntimeError:
                    raise RuntimeError("Run time error occured, check your API keys.")
                
                print(f"Got {len(results)} results")
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;→ {len(results)} results")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(results)
                })

            search_count += 1

            if search_count == 10:
                # Nudge the model to wrap up and stop offering tools — fires exactly once
                messages.append({"role": "user", "content": "You have reached the maximum number of searches. Please return the JSON now."})
                nudge_sent = True
            elif search_count > 13:
                # Hard exit if the model ignores the nudge
                return {}
        else:
            print(f"Raw response searching llm: '{response.choices[0].message.content}'")
            try:
                text = response.choices[0].message.content
                # Strip markdown code fences the model sometimes wraps around JSON
                text = re.sub(r"```[a-z]*", "", text).strip()
                # Extract the JSON object even when the model prepends explanatory prose
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
                return {}
            except json.JSONDecodeError:
                print("Invalid JSON in searching_llm, skipping")
                return {}


def sort_sources(location, start_date, end_date, pollutant, sources):
    prompt = f'''
        A {pollutant} spike was detected near {location} between {start_date} and {end_date}. Here are
        a list of current sources:

        {json.dumps(sources)}

        Your job is to only return a JSON object of these sources, with keys Source_1, Source_2, Source_3,
        etc., organized as follows:

        STEP 0 — Filter for relevance. Exclude any source that meets EITHER of these conditions:

          CONDITION A — Geographic irrelevance (both sub-conditions must be true to exclude):
            (a1) The event it describes occurred in a clearly different geographic area (different state
                 or region) from {location}
            (a2) The source does not document any direct effect (smoke, emissions, contamination)
                 reaching {location} or its surrounding area

          POLLUTANT RELEVANCE NOTE (applies to Condition A only — geographic relevance):
            The spike being investigated is specifically for {pollutant}.
            Sources that explicitly mention {pollutant} or known precursors/co-pollutants of {pollutant}
            are geographically relevant. Sources describing events that could not plausibly release {pollutant}
            (e.g., a wildfire when the pollutant is a heavy metal) may be excluded under Condition A
            unless they document direct effects reaching {location}.
            IMPORTANT: This note applies ONLY to Condition A (geographic scope). Mentioning {pollutant}
            does NOT exempt a source from Condition B (must document a specific incident, not routine
            administrative content) or Condition C (event must fall within the spike window).

          CONDITION B — Not a specific event: The source is routine monitoring data, an annual
            program report, a permit application or reclassification, or any other background /
            administrative content that does NOT document a specific incident, accident, fire,
            spill, or other discrete occurrence that could explain a pollutant spike.
            Exclude examples:
              - Air quality monitoring dashboards showing routine historical data (no incident)
              - Annual air quality program updates or ozone advance reports
              - Operating permit applications, renewals, or reclassifications
              - Permit attachments or annexes listing chemicals handled at a facility (e.g.,
                "Attachments to Draft RCRA Permit listing methanol as a hazardous waste" — this
                documents that a facility handles the chemical, NOT that a release occurred)
              - Hazardous waste management notices, proposed exclusions, or reclassifications
              - Title V air permit program reviews documenting routine regulatory oversight
              - State environmental agency reviews of incomplete or pending permit applications
              - Any document showing a facility handles {pollutant} without documenting a
                specific release, spill, or emission event
              - General background information about a facility or region
            Keep examples:
              - Incident reports for a specific fire, spill, explosion, or accident
              - News coverage of a discrete pollution event
              - Air quality health advisories tied to a specific event
              - Monitoring readings specifically documenting an elevation during an incident

          Excluded sources must be completely omitted from the output JSON — do not include
          them with a low rank.

          Additional Condition B exclude examples:
            - Planning announcements or preparatory notices for events that have not yet
              occurred (e.g., "Planning underway for prescribed burns", "Burn scheduled
              for next week")
            - Outlook or forecast documents describing anticipated future fire danger or
              air quality conditions, unless they document that the forecasted conditions
              actually occurred (e.g., a seasonal wildfire outlook that only projects risk,
              not a confirmed incident)
            - Smoke management plans, air quality management plans, state implementation
              plans, or other regulatory/procedural frameworks describing how agencies
              control or respond to emissions (e.g., "Oklahoma Smoke Management Plan",
              "State Smoke Management Plan", "Emission Reduction Plan") — these describe
              protocols, not specific events

          CONDITION C — Temporal irrelevance: Exclude a source if BOTH of the following are true:
            (c1) The source's publication date or the event it describes falls clearly outside
                 the investigation window ({start_date} to {end_date}) — for example, months
                 or years before or after the spike period.
            (c2) The source is NOT a retrospective investigation, incident report, or academic
                 study that was published after {end_date} but specifically documents the event
                 or conditions during {start_date}–{end_date}.
            Excluded examples:
              - An article from December 2021 about a wildfire when the spike is March 2021
                (describes a different, later event; cannot be causal)
              - A news story from 2019 about a facility when the spike is 2021
              - A routine emission test report from March 2018 at a facility when the spike is
                February 2019 — the test predates the window by nearly a year and is not a
                retrospective report about the Feb 2019 spike; exclude it under Condition C
            Keep examples:
              - A CSB investigation report published June 2021 about an accident that
                occurred in March 2021
              - An EPA post-incident summary published weeks after {end_date} that
                specifically covers the spike period

            IMPORTANT CLARIFICATION ON CONDITION C:
              A source describing a specific past event (explosion, spill, accident) at a
              facility near {location} does NOT escape Condition C simply because the facility
              is nearby or handles {pollutant}. The event itself — not the facility's general
              history or location — must fall within or directly document conditions during
              {start_date}–{end_date}. Ask: "Did this event happen during the query window?"
              If the answer is no, and the source does not explicitly document that event's
              effects persisting into {start_date}–{end_date}, exclude it.
              Example: A January 2018 explosion at a Henderson chlorine plant is NOT evidence
              for a May 2018 chlorine spike, even though the same facility handles chlorine.
              Exclude it under Condition C.

        STEP 1 — Assign each source an event_type (e.g. Wildfire, Industrial Accident, Power Plant Emissions,
        General Air Quality, etc.) based on what the source describes.

        STEP 1b — Assign each source an entity: the single specific facility, company, vessel,
        pipeline, or named/dated incident that is the actual origin of the pollutant release or
        exposure being described — regardless of the source's event_type.

          CRITICAL — cross-event_type consistency: If two or more sources describe the SAME
          underlying facility or incident (e.g., one source is an EPA/state shutdown order for a
          facility and another is an ATSDR/IDPH risk-assessment report about emissions from that
          SAME facility), they MUST share the IDENTICAL entity string, even though their event_type
          differs (e.g. "Industrial Accident" vs "General Air Quality"). Use the most complete,
          consistent canonical name (company + facility name), e.g. always
          "Sterigenics Willowbrook Facility" — never sometimes "Sterigenics" and sometimes
          "Willowbrook facility" or "the Willowbrook plant."

          PRESERVE EXISTING VALUES: If a source in the input JSON already has an entity value, keep
          it exactly as-is unless you have clear evidence it is wrong. When assigning entity to a
          newly-added source, first check whether any entity value already used elsewhere in the
          input JSON describes the same facility/incident — if so, reuse that exact string verbatim
          (same spelling/capitalization) instead of writing a new, similar-but-different string.

          DO NOT OVER-MERGE: Only share an entity string across sources that describe the same
          physical facility, company, vessel, or single specific incident. Do NOT merge two
          different facilities just because they are the same industry, owner, or nearby location —
          if in doubt, treat them as separate entities.

          QUALIFIER DRIFT (same owner/location, missing sub-plant designation): The DO NOT
          OVER-MERGE rule above is about different facilities/companies — it does NOT apply when
          the only difference between a candidate name and an already-used entity string is a
          dropped or added qualifier (e.g. "East"/"West", "Plant", "Unit 2") for the SAME owner and
          location. In that narrower case, default to MERGING: if a source doesn't itself specify
          which sub-plant/unit is responsible, reuse the existing (more specific) entity string
          already assigned to that owner/location rather than creating a new, less-qualified
          variant. Only keep them separate if some source explicitly distinguishes the sub-plants
          (e.g. one source is specifically about the West plant and another specifically about the
          East plant).

          COMBINED-ENTITY NAMING: When a single source implicates two or more facilities jointly
          (e.g. two nearby refineries both named as possible sources in the same report), join their
          names with " & " in ALPHABETICAL order (e.g. always "A Refinery & B Refinery", never
          "B Refinery & A Refinery") so the same combination always produces the identical string
          regardless of which facility the source mentions first.

          NO SINGLE RESPONSIBLE ENTITY: If a source does not describe a specific attributable
          facility, company, or incident (e.g., a general regional air-quality-index reading, a
          diffuse meteorological/wind pattern with no identifiable responsible party), set entity
          equal to that source's own event_type string (e.g. entity = "General Air Quality").
          Do not invent a facility name.

        STEP 2 — Assign each source a date field (e.g. "March 2021") if the month can be inferred from the
        source title or summary. Omit the date field entirely if unknown.

        STEP 3 — Order the sources by these three criteria in priority order:
          1. Group by event_type — all sources with the same event_type MUST be consecutive.
             NEVER interleave sources from two different event_types.
          2. Within each group, rank strictly from most to least credible using this exact tier list:
               Tier 1 (highest): Government reports / Government databases
               Tier 2: Academic Institution / peer-reviewed (universities, research labs, hospital systems; type label: "Academic Institution")
               Tier 3: Established national news outlets
               Tier 4: Local news outlets
               Tier 5: Blogs / Non-profits / Industry publications
               Tier 6: Wikipedia / encyclopedias  (Wikipedia is LOW credibility, below all news; type label: "Wikipedia")
               Tier 7 (lowest): Social media
             A lower-tier source must NEVER appear before a higher-tier source within the same group.
             NOTE: University offices (e.g., a university EHS page, health center, or research center) are
             Tier 2 Academic Institution — NOT Tier 1 Government.
             Wikipedia pages are always Tier 6 Wikipedia — NOT Tier 3 News Outlet.
             A post published directly by an official government agency's own account (e.g., a state
             environmental agency's Facebook or Twitter/X page announcing an official advisory) is Tier 1
             Government Report — NOT Tier 7 Social Media. Tier 7 Social Media is reserved for posts by
             private individuals, unofficial commentary, or discussion the agency itself did not publish.
          3. Within each credibility tier, sort by date oldest-to-newest if sources span different months

        At the end of the JSON, there should be another key, 'Continue', which is either true or false. If true, then
        another LLM will search further, update the sources list, and you will be called again to sort them. If false,
        then we will stop searching. For example, you may continue searching if we only have very little sources, and you
        may stop searching if you believe there are enough sources already, or continuing to look for any will be less meaningful.
        Stop searching if there are already 8+ sources, or if the top sources are already high credibility government/academic ones.

        Do not search for new sources.
        Your job is to only sort the current existing ones.

        Do not write any explanatory text before or after the JSON.
        Do not acknowledge the task or describe what you are doing.
        Return ONLY the JSON.

        Example JSON (always follow this exact format):
        {{
            "Source_1": {{
                "title": "California Air Resources Board - Incident Report: Dixie Fire Emissions",
                "url": "https://ww2.arb.ca.gov/...",
                "type": "Government Report",
                "event_type": "Wildfire",
                "entity": "Dixie Fire",
                "date": "July 2021",
                "summary": "Official emissions data recorded during the Dixie Fire, July-August 2021."
            }},
            "Source_2": {{
                "title": "EPA AirNow Fire and Smoke Map - Event Log",
                "url": "https://www.airnow.gov/...",
                "type": "Government Database",
                "event_type": "Wildfire",
                "entity": "Dixie Fire",
                "date": "August 2021",
                "summary": "Air quality index readings spiking in the region during the relevant period."
            }},
            "Source_3": {{
                "title": "Princeton University EHS - Air Quality Alert June 29-30 2023",
                "url": "https://ehs.princeton.edu/...",
                "type": "Academic Institution",
                "event_type": "Wildfire",
                "entity": "2023 Canadian Wildfires",
                "date": "June 2023",
                "summary": "University environmental health and safety office alert about wildfire smoke air quality."
            }},
            "Source_4": {{
                "title": "Reuters - Massive wildfire spreads across Northern California",
                "url": "https://www.reuters.com/...",
                "type": "News Outlet",
                "event_type": "Wildfire",
                "entity": "Dixie Fire",
                "date": "July 2021",
                "summary": "News coverage reporting the fire's spread and affected counties."
            }},
            "Source_5": {{
                "title": "2023 Canadian wildfires - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/2023_Canadian_wildfires",
                "type": "Wikipedia",
                "event_type": "Wildfire",
                "entity": "2023 Canadian Wildfires",
                "date": "June 2023",
                "summary": "Wikipedia article documenting the 2023 Canadian wildfire season and its effects on air quality across North America."
            }},
            "Source_6": {{
                "title": "Reddit post - r/California - Anyone else seeing crazy smoke?",
                "url": "https://www.reddit.com/...",
                "type": "Social Media",
                "event_type": "Wildfire",
                "entity": "Dixie Fire",
                "summary": "User reports of heavy smoke in the area around the same dates."
            }},
            "Source_7": {{
                "title": "California Air Resources Board - Facebook post: Air quality advisory issued",
                "url": "https://www.facebook.com/CaliforniaARB/posts/...",
                "type": "Government Report",
                "event_type": "Wildfire",
                "entity": "Dixie Fire",
                "date": "July 2021",
                "summary": "Official agency Facebook post announcing an air quality health advisory due to wildfire smoke — Government Report tier, not Social Media, because it is the agency's own official statement."
            }},
            "Source_8": {{
                "title": "CSB Investigation Report - Refinery Explosion",
                "url": "https://www.csb.gov/...",
                "type": "Government Report",
                "event_type": "Industrial Accident",
                "entity": "Example Refinery",
                "date": "March 2021",
                "summary": "Chemical Safety Board report on the refinery explosion and resulting emissions."
            }},
            "Source_9": {{
                "title": "Illinois EPA Seal Order - Acme Chemical Plant Ethylene Oxide Emissions",
                "url": "https://www.illinois.gov/...",
                "type": "Government Report",
                "event_type": "Industrial Accident",
                "entity": "Acme Chemical Plant",
                "date": "February 2019",
                "summary": "State agency shutdown/seal order restricting the facility due to public health hazards from emissions."
            }},
            "Source_10": {{
                "title": "ATSDR Health Consultation - Ethylene Oxide Concentrations Near Acme Chemical Plant",
                "url": "https://www.atsdr.cdc.gov/...",
                "type": "Government Report",
                "event_type": "General Air Quality",
                "entity": "Acme Chemical Plant",
                "date": "March 2019",
                "summary": "Federal health agency risk assessment of elevated emissions near the same facility as Source_9 — different event_type (monitoring/risk data vs. enforcement action) but the SAME entity, so it shares the identical entity string."
            }},
            "Source_11": {{
                "title": "State Air Quality Index Summary - Regional Ozone Levels",
                "url": "https://www.airnow.gov/...",
                "type": "Government Database",
                "event_type": "General Air Quality",
                "entity": "General Air Quality",
                "date": "June 2021",
                "summary": "Routine regional AQI reading with no single identifiable responsible facility or incident, so entity falls back to the event_type string."
            }},
            "Continue": true
        }}
    '''

    try:
        response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = [{
                "role": "user",
                "content": prompt
            }],
            response_format = {"type": "json_object"}
        )
    except OpenAIAuthError:
        raise RuntimeError("Invalid DeepSeek API key. Please check your key in the sidebar.")
    except Exception as e:
        print(f"LLM API error in sort_sources: {e}")
        return {}

    try:
        text = response.choices[0].message.content
        text = re.sub(r"```[a-z]*", "", text).strip()
        print(f"Raw response sort sources: {text}")
        return json.loads(text)
    except json.JSONDecodeError:
        print("Invalid JSON in sort sources, skipping")
        return {}

def dedupe_sources(sources):
    seen_urls = set()
    deduped = {}
    i = 1
    for v in sources.values():
        url = v.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped[f"Source_{i}"] = v
            i += 1
    return deduped

def _credibility_tier(source_type: str) -> int:
    t = source_type.lower()
    if "government" in t:                                    return 1
    if "academic" in t or "peer" in t or "journal" in t:     return 2
    if "local news" in t:                                    return 4
    if "news" in t:                                          return 3
    if "wikipedia" in t or "encyclopedia" in t:              return 6
    if "social" in t:                                        return 7
    return 5

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}

def _source_in_window(date_str, start_date, end_date, tolerance_days = 60):
    if not date_str:
        return None
    
    m = re.search(r'(\w+)\s+(\d{4})', date_str, re.IGNORECASE)
    if not m:
        return None
    mon = _MONTH_MAP.get(m.group(1).lower())
    if not mon:
        return None
    
    src_date = datetime.date(int(m.group(2)), mon, 1)
    window_start = start_date - datetime.timedelta(days = tolerance_days)
    window_end = end_date + datetime.timedelta(days = tolerance_days)
    return window_start <= src_date <= window_end

OUT_OF_WINDOW_DISCOUNT = 0.25

def compute_confidence_score(sources: dict, start_date = None, end_date = None) -> dict:
    tier_weights = {1: 10, 2: 8, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
    source_list = list(sources.values())
    if not source_list:
        return {"score": 0, "confidence": "Low", "quality": 0, "consensus": 0, "coverage": 0,
                "top_entity": "", "top_et": "", "group_scores": {}, "entity_scores": {}}

    event_weights, event_counts = {}, {}
    entity_weights, entity_counts = {}, {}
    entity_event_types = {}  # entity -> {event_type: discounted_count}
    entity_raw_counts = {}  # entity -> raw (undiscounted) source count, for display only
    entity_et_raw_counts = {}  # entity -> {event_type: raw (undiscounted) source count}
    total_weight = 0
    for src in source_list:
        et = src.get("event_type", "Unknown")
        entity = src.get("entity") or et
        weight = tier_weights.get(_credibility_tier(src.get("type", "")), 1)
        discount = 1
        if start_date and end_date and _source_in_window(src.get("date", ""), start_date, end_date) is False:
            discount = OUT_OF_WINDOW_DISCOUNT
        event_weights[et] = event_weights.get(et, 0) + weight * discount
        event_counts[et] = event_counts.get(et, 0) + discount
        entity_weights[entity] = entity_weights.get(entity, 0) + weight * discount
        entity_counts[entity] = entity_counts.get(entity, 0) + discount
        entity_event_types.setdefault(entity, {})
        entity_event_types[entity][et] = entity_event_types[entity].get(et, 0) + discount
        entity_raw_counts[entity] = entity_raw_counts.get(entity, 0) + 1
        entity_et_raw_counts.setdefault(entity, {})
        entity_et_raw_counts[entity][et] = entity_et_raw_counts[entity].get(et, 0) + 1
        total_weight += weight * discount

    total_count = len(source_list)
    top_entity = max(entity_weights, key = lambda e: min(entity_weights[e], 40) + round((entity_counts[e] / total_count) * 30))
    quality = round(min(entity_weights[top_entity], 40))
    consensus = round((entity_counts[top_entity] / total_count) * 30)
    coverage = round(min(total_weight, 30))
    score = quality + consensus + coverage
    confidence = "High" if score >= 70 else ("Medium" if score >= 40 else "Low")

    # Representative event_type for the winning entity — its own highest-weight event_type.
    top_et = max(entity_event_types[top_entity], key = lambda et: entity_event_types[top_entity][et])

    group_scores = {}
    for et in event_weights:
        g_quality = round(min(event_weights[et], 40))
        g_consensus = round((event_counts[et] / total_count) * 30)
        g_score = g_quality + g_consensus + coverage
        group_scores[et] = {
            "score": g_score,
            "confidence": "High" if g_score >= 70 else ("Medium" if g_score >= 40 else "Low")
        }

    entity_scores = {}
    for e in entity_weights:
        e_quality = round(min(entity_weights[e], 40))
        e_consensus = round((entity_counts[e] / total_count) * 30)
        e_score = e_quality + e_consensus + coverage
        entity_scores[e] = {
            "score": e_score,
            "confidence": "High" if e_score >= 70 else ("Medium" if e_score >= 40 else "Low"),
            "event_types": sorted(entity_event_types[e]),
            "source_count": entity_raw_counts[e],
        }

    # Attach each event_type group's dominant entity and any sibling event_types sharing it,
    # so the UI can flag cross-group corroboration without relying on LLM prose.
    et_dominant_entity = {}
    for et in event_weights:
        et_dominant_entity[et] = max(
            (e for e in entity_event_types if et in entity_event_types[e]),
            key = lambda e: entity_event_types[e][et]
        )
    for et, gs in group_scores.items():
        dom = et_dominant_entity[et]
        gs["entity"] = dom
        gs["sibling_event_types"] = [x for x in entity_scores[dom]["event_types"] if x != et]

        # Every distinct entity contributing sources to this event_type group, so the UI can
        # show real per-entity numbers instead of implying the group score is one cause's score.
        group_entities = sorted(
            (e for e in entity_event_types if et in entity_event_types[e]),
            key = lambda e: entity_scores[e]["score"], reverse = True
        )
        gs["entities"] = [
            {"entity": e, "score": entity_scores[e]["score"], "confidence": entity_scores[e]["confidence"],
             "source_count": entity_et_raw_counts[e][et]}
            for e in group_entities
        ]
        gs["is_single_entity"] = len(group_entities) <= 1

    return {"score": score, "confidence": confidence, "quality": quality, "consensus": consensus,
            "coverage": coverage, "top_entity": top_entity, "top_et": top_et,
            "group_scores": group_scores, "entity_scores": entity_scores}

def _canonicalize_entity(entity: str) -> str:
    if not entity or "&" not in entity:
        return entity
    parts = [p.strip() for p in entity.split("&") if p.strip()]
    return " & ".join(sorted(parts, key = str.lower))

def post_sort_sources(sources: dict) -> dict:
    group_order, groups = [], {}
    for src in sources.values():
        if src.get("entity"):
            src["entity"] = _canonicalize_entity(src["entity"])
        et = src.get("event_type", "Unknown")
        if et not in groups:
            group_order.append(et)
            groups[et] = []
        groups[et].append(src)

    result, i = {}, 1
    for et in group_order:
        for src in sorted(groups[et], key = lambda s: _credibility_tier(s.get("type", ""))):
            result[f"Source_{i}"] = src
            i += 1
    return result

def run_search_pipeline(loc, start, end, poll, model):
    sources = {}
    searched_queries = set()

    st.write(f"**Round 1** — Searching for {poll} events near {loc}")
    new_sources = searching_llm(loc, start, end, poll, sources, searched_queries, model)
    offset = len(sources)
    merged = dedupe_sources({**sources,
              **{f"Source_{offset + i + 1}": v for i, v in enumerate(new_sources.values())}})
    st.write(f"Ranking {len(merged)} sources...")
    result = sort_sources(loc, start, end, poll, merged)
    sorted_sources = {k: v for k, v in result.items() if k != "Continue"}
    sources = post_sort_sources(sorted_sources) if sorted_sources else merged

    count = 2
    while result.get("Continue", False) and count <= 5:
        st.write(f"**Round {count}** — Refining with {len(sources)} sources so far")
        new_sources = searching_llm(loc, start, end, poll, sources, searched_queries, model)
        offset = len(sources)
        merged = dedupe_sources({**sources,
                  **{f"Source_{offset + i + 1}": v for i, v in enumerate(new_sources.values())}})
        st.write(f"Ranking {len(merged)} sources...")
        result = sort_sources(loc, start, end, poll, merged)
        sorted_sources = {k: v for k, v in result.items() if k != "Continue"}
        sources = post_sort_sources(sorted_sources) if sorted_sources else merged
        count += 1

    return sources

class _TextExtractor(HTMLParser):
    _skip = {"script", "style", "head", "nav", "footer"}

    def __init__(self):
        super().__init__()
        self._buf, self._depth = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._skip and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            self._buf.append(data)

    def get_text(self):
        return re.sub(r"\s+", " ", " ".join(self._buf)).strip()

_PAGE_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_page_text(url):
    try:
        resp = requests.get(url, timeout = 20, headers = _PAGE_FETCH_HEADERS)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")

        looks_like_pdf = url.lower().split("?")[0].endswith(".pdf")
        if looks_like_pdf and "pdf" not in content_type.lower() and len(resp.content) < 1000:
            print(f"fetch_page_text blocked (likely bot-protection challenge page) for {url}: "
                  f"expected PDF but got {len(resp.content)} bytes of '{content_type}'")
            return {"text": "", "status": "blocked"}

        if "application/pdf" in content_type:
            reader = PdfReader(io.BytesIO(resp.content))
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            text = re.sub(r"\s+", " ", text).strip()[:3000]
            return {"text": text, "status": "ok" if text else "empty"}

        if "text/html" not in content_type:
            print(f"fetch_page_text skipped unsupported content-type '{content_type}' for {url}")
            return {"text": "", "status": "error"}

        extractor = _TextExtractor()
        extractor.feed(resp.text)
        text = extractor.get_text()[:3000]
        return {"text": text, "status": "ok" if text else "empty"}
    except Exception as e:
        print(f"fetch_page_text failed for {url}: {e}")
        return {"text": "", "status": "error"}

def synthesize_findings(location, start_date, end_date, pollutant, sources):
    score_data = compute_confidence_score(sources, start_date, end_date)

    top_entity = score_data.get("top_entity", "")
    top_et = score_data.get("top_et", "")

    group_score_lines = "\n".join(
        f"  - {et}: {gs['score']}/100 ({gs['confidence']})"
        + (f" — spans {len(gs['entities'])} distinct entities, not a single cause"
           if not gs.get("is_single_entity", True) else "")
        for et, gs in sorted(score_data["group_scores"].items(), key = lambda x: x[1]["score"],
        reverse = True)
    )

    entity_score_lines = "\n".join(
        f"  - {e}: {es['score']}/100 ({es['confidence']}) — {es['source_count']} source(s) "
        f"across event type(s): {', '.join(es['event_types'])}"
        for e, es in sorted(score_data.get("entity_scores", {}).items(), key = lambda x: x[1]["score"],
        reverse = True)
    )

    score_context = f"""The following confidence scores were computed deterministically from
        source credibility and consensus. An "entity" is the specific facility, company, or
        incident responsible for a source's described event; sources that share an entity are
        evidence for the SAME underlying cause even when their event_type label differs (e.g. an
        enforcement/shutdown source and a monitoring/risk-assessment source about the same
        facility) — they are NOT competing alternative causes.

        Overall: {score_data['score']}/100 → {score_data['confidence']}
        Per entity (descending by score):
        {entity_score_lines}
        Top-ranked entity: {top_entity} (representative event type: {top_et})

        Per event type (descending by score) — informational subgroup view only:
        {group_score_lines}"""

    _STATUS_NOTE = {
        "ok": "",
        "empty": "page text unavailable — page fetched successfully but contained no extractable text",
        "blocked": "page text unavailable — site blocked automated access (bot-protection challenge page)",
        "error": "page text unavailable — the page could not be fetched (network error or unsupported content type)",
    }

    enriched = {}
    with ThreadPoolExecutor(max_workers = 8) as executor:
        futures = {executor.submit(fetch_page_text, src.get("url", "")): (key, src)
                   for key, src in sources.items()}
        for future in as_completed(futures):
            key, src = futures[future]
            fetch_result = future.result()
            enriched[key] = {
                **src,
                "page_text": fetch_result["text"],
                "page_text_note": _STATUS_NOTE[fetch_result["status"]],
            }

    prompt = f"""
        A {pollutant} spike was detected near {location} between {start_date} and {end_date}.
        The following sources were found. Each includes its metadata and up to 3000 characters
        of text fetched directly from the source URL. If "page_text_note" is non-empty, the
        fetch failed for the stated reason and "page_text" should be treated as unavailable
        rather than as evidence the source itself lacks relevant content.

        {json.dumps(enriched, indent = 2)}

        {score_context}

        Analyze these sources for an environmental analyst and return ONLY a JSON object with
        these exact keys (no prose before or after the JSON):

        {{
          "most_likely_cause": "One sentence identifying {top_entity} as the most likely cause of
           the {pollutant} spike (combined confidence score {score_data['score']}/100 across
           {score_data.get('entity_scores', {}).get(top_entity, {}).get('source_count', 0)} source(s)
           spanning event type(s): {', '.join(score_data.get('entity_scores', {}).get(top_entity, {}).get('event_types', []))}),
           citing specific source titles. If {top_entity} equals its own event_type value (meaning no
           single identifiable responsible facility/company/incident was found — e.g. a general
           regional air-quality pattern), phrase the sentence around that general condition instead
           of implying a specific facility exists. Explicitly state whether this cause would
           plausibly release {pollutant} or its precursors. If the fetched source text strongly
           contradicts this ranking, note the discrepancy.",
          "key_evidence": ["Bullet citing source 1 — include any direct mention of {pollutant} if found", "Bullet citing source 2", "Bullet citing source 3"],
          "gaps_and_alternatives": ["Gap or alternative explanation 1", "Gap 2"],
          "confidence_rationale": "One sentence explaining why the evidence warrants an overall
           score of {score_data['score']}/100 ({score_data['confidence']}) — what specifically drives
           the score up or down (e.g. source credibility, degree of consensus across entities, total
           volume of evidence).",
          "group_confidence": {{
            "<event_type exactly as it appears in the sources>": {{
              "rationale": "One sentence explaining why this event type received its computed
               score (visible in the score context above) — what makes its sources stronger or weaker than
               other event types. If this event type's sources share an entity with another event type
               shown in the per-entity table above, say so explicitly and note that they are combined
               evidence for the same entity, not a competing cause, citing the combined entity score.
               If the score context marks this event type as spanning multiple distinct entities, say so
               explicitly and make clear the score reflects aggregate volume across several unrelated
               facilities/incidents, not confidence in any single one of them as the cause."
            }}
          }}
        }}

        For group_confidence, include one entry per distinct event_type value present in the sources above.
        Use the exact event_type string as the key. Be concise and specific. Ground claims in the source text above; do not invent facts.
    """
    try:
        response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = [{"role": "user", "content": prompt}],
            response_format = {"type": "json_object"}
        )
    except OpenAIAuthError:
        raise RuntimeError("Invalid DeepSeek API key. Please check your key in the sidebar.")
    except Exception as e:
        print(f"LLM API error in synthesize_findings: {e}")
        return {}

    try:
        text = response.choices[0].message.content
        text = re.sub(r"```[a-z]*", "", text).strip()
        result = json.loads(text)
        result.update(score_data)
        return result
    except json.JSONDecodeError:
        print("Invalid JSON in synthesis_findings()")
        return text

# Helpers for the batch feature
def validate_batch_df(df):
    # Return (rows: list[dict], errors: list[str]) from a raw DataFrame.
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"location", "start date", "end date", "pollutant"}
    missing = required - set(df.columns)
    if missing:
        return [], [f"Missing required columns: {', '.join(sorted(missing))}"]

    rows, errors = [], []
    for idx, row in df.iterrows():
        row_num = idx + 2  # Excel row number (1-indexed header + offset)
        raw_vals = [str(row.get(c, "")).strip() for c in ("location", "start date", "end date", "pollutant")]
        if not any(v for v in raw_vals if v not in ("nan", "")):
            continue  # silently skip fully blank rows
        loc = str(row["location"]).strip() if pd.notna(row["location"]) else ""
        poll = str(row["pollutant"]).strip() if pd.notna(row["pollutant"]) else ""
        if loc in ("nan", ""):
            loc = ""
        if poll in ("nan", ""):
            poll = ""
        if not loc or not poll:
            errors.append(f"Row {row_num}: missing {'location' if not loc else 'pollutant'}")
            continue
        try:
            start = pd.to_datetime(row["start date"]).date()
        except Exception:
            errors.append(f"Row {row_num}: invalid start date '{row['start date']}'")
            continue
        try:
            end = pd.to_datetime(row["end date"]).date()
        except Exception:
            errors.append(f"Row {row_num}: invalid end date '{row['end date']}'")
            continue
        if start > end:
            errors.append(f"Row {row_num}: start date {start} is after end date {end}")
            continue
        rows.append({"location": loc, "start_date": start, "end_date": end, "pollutant": poll})
    return rows, errors

def _style_header(ws):
    hdr_font = Font(bold = True, color = "FFFFFF")
    hdr_fill = PatternFill("solid", fgColor = "4472C4")
    for cell in ws[1]:
        if cell.value is not None:
            cell.font = hdr_font
            cell.fill = hdr_fill
    ws.freeze_panes = "A2"

def _autosize_columns(ws):
    for col in ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default = 0)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

def _wb_to_bytes(wb):
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

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
                date_warning = " **(⚠ outside query window)**" if _source_in_window(date_str, start_date, end_date) is False else ""
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

    keys_ok = bool(st.session_state.deepseek_key and st.session_state.serper_key)
    fields_ok = bool(location and pollutant)

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
        client = OpenAI(api_key = st.session_state.deepseek_key, base_url = "https://api.deepseek.com")
        loc = st.session_state.location
        start = st.session_state.start_date
        end = st.session_state.end_date
        poll = st.session_state.pollutant
        
        try:
            with st.status("Searching...", expanded = True) as status:
                st.session_state.sources = run_search_pipeline(
                    loc, start, end, poll, st.session_state.search_model
                )
                status.update(label = f"Done — found {len(st.session_state.sources)} sources",
                              state = "complete", expanded = False)

            with st.spinner("Synthesizing findings..."):
                st.session_state.synthesis = synthesize_findings(
                    loc, start, end, poll, st.session_state.sources
                )
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
            valid_rows, file_errors = validate_batch_df(df)
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
        client = OpenAI(api_key = st.session_state.deepseek_key, base_url = "https://api.deepseek.com")

        wb = openpyxl.Workbook()
        summary_ws = wb.active
        summary_ws.title = "Summary"
        sources_ws = wb.create_sheet("Sources")

        summary_ws.append(["Search #", "Location", "Start Date", "End Date", "Pollutant",
                            "Score", "Confidence", "Most Likely Cause", "Source Count", "Error"])
        sources_ws.append(["Search #", "Location", "Pollutant", "Source #",
                            "Title", "URL", "Type", "Event Type", "Entity", "Date", "Summary"])

        _style_header(summary_ws)
        _style_header(sources_ws)

        st.session_state.batch_excel = _wb_to_bytes(wb)
        
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
                        sources = run_search_pipeline(loc, start, end, poll, st.session_state.search_model)
                        st.write("Synthesizing...")
                        synthesis = synthesize_findings(loc, start, end, poll, sources)
                except Exception as e:
                    err_msg = str(e)
                    st.error(f"Search {i} error: {e}")

                if isinstance(synthesis, dict):
                    score = synthesis.get("score", "")
                    confidence = synthesis.get("confidence", "")
                    cause = synthesis.get("most_likely_cause", "")
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

                st.session_state.batch_excel = _wb_to_bytes(wb)

        except Exception as catastrophic_e:
            st.error(f"Batch search stopped unexpectedly: {catastrophic_e}")
        finally:
            _autosize_columns(summary_ws)
            _autosize_columns(sources_ws)
            st.session_state.batch_excel = _wb_to_bytes(wb)
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