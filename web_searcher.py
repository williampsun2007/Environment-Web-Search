import streamlit as st
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import requests
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser

load_dotenv()

client = OpenAI(api_key=os.getenv("api_key"), base_url = "https://api.deepseek.com")

if "location" not in st.session_state:
    st.session_state.location = ""
if "start_date" not in st.session_state:
    st.session_state.start_date = ""
if "end_date" not in st.session_state:
    st.session_state.end_date = ""
if "pollutant" not in st.session_state:
    st.session_state.pollutant = ""
if "sources" not in st.session_state:
    st.session_state.sources = {}
    
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
                "X-API-KEY": os.getenv("search_api_key"),
                "Content-Type": "application/json"
            },
            json={"q": query},
            timeout=20
        )
        response.raise_for_status()
        results = response.json().get("organic", [])
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
    
def searching_llm(location, start_date, end_date, pollutant, existing_sources, searched_queries = None):
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
                {json.dumps(sorted(searched_queries), indent = 16)}
                """

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
                - type (e.g. Government Report, News Outlet, Social Media, etc.)
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
        call_kwargs = {"model": "deepseek-reasoner", "messages": messages}
        if not nudge_sent:
            call_kwargs["tools"] = tools
        try:
            response = client.chat.completions.create(**call_kwargs)
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
                results = search_web(query)
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

          POLLUTANT RELEVANCE NOTE: The spike being investigated is specifically for {pollutant}.
            Sources that explicitly mention {pollutant} or known precursors/co-pollutants of {pollutant}
            are highly relevant. Sources describing events that could not plausibly release {pollutant}
            (e.g., a wildfire when the pollutant is a heavy metal) may be excluded under Condition A
            unless they document direct effects reaching {location}.

          CONDITION B — Not a specific event: The source is routine monitoring data, an annual
            program report, a permit application or reclassification, or any other background /
            administrative content that does NOT document a specific incident, accident, fire,
            spill, or other discrete occurrence that could explain a pollutant spike.
            Exclude examples:
              - Air quality monitoring dashboards showing routine historical data (no incident)
              - Annual air quality program updates or ozone advance reports
              - Operating permit applications, renewals, or reclassifications
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
            Keep examples:
              - A CSB investigation report published June 2021 about an accident that
                occurred in March 2021
              - An EPA post-incident summary published weeks after {end_date} that
                specifically covers the spike period

        STEP 1 — Assign each source an event_type (e.g. Wildfire, Industrial Accident, Power Plant Emissions,
        General Air Quality, etc.) based on what the source describes.

        STEP 2 — Assign each source a date field (e.g. "March 2021") if the month can be inferred from the
        source title or summary. Omit the date field entirely if unknown.

        STEP 3 — Order the sources by these three criteria in priority order:
          1. Group by event_type — all sources with the same event_type MUST be consecutive.
             NEVER interleave sources from two different event_types.
          2. Within each group, rank strictly from most to least credible using this exact tier list:
               Tier 1 (highest): Government reports / Government databases
               Tier 2: Academic / peer-reviewed
               Tier 3: Established national news outlets
               Tier 4: Local news outlets
               Tier 5: Blogs / Non-profits / Industry publications
               Tier 6: Wikipedia / encyclopedias  (Wikipedia is LOW credibility, below all news)
               Tier 7 (lowest): Social media
             A lower-tier source must NEVER appear before a higher-tier source within the same group.
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
                "date": "July 2021",
                "summary": "Official emissions data recorded during the Dixie Fire, July-August 2021."
            }},
            "Source_2": {{
                "title": "EPA AirNow Fire and Smoke Map - Event Log",
                "url": "https://www.airnow.gov/...",
                "type": "Government Database",
                "event_type": "Wildfire",
                "date": "August 2021",
                "summary": "Air quality index readings spiking in the region during the relevant period."
            }},
            "Source_3": {{
                "title": "Reuters - Massive wildfire spreads across Northern California",
                "url": "https://www.reuters.com/...",
                "type": "News Outlet",
                "event_type": "Wildfire",
                "date": "July 2021",
                "summary": "News coverage reporting the fire's spread and affected counties."
            }},
            "Source_4": {{
                "title": "CSB Investigation Report - Refinery Explosion",
                "url": "https://www.csb.gov/...",
                "type": "Government Report",
                "event_type": "Industrial Accident",
                "date": "March 2021",
                "summary": "Chemical Safety Board report on the refinery explosion and resulting emissions."
            }},
            "Source_5": {{
                "title": "Reddit post - r/California - Anyone else seeing crazy smoke?",
                "url": "https://www.reddit.com/...",
                "type": "Social Media",
                "event_type": "Wildfire",
                "summary": "User reports of heavy smoke in the area around the same dates."
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
            response_format={"type": "json_object"}
        )
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

def post_sort_sources(sources: dict) -> dict:
    group_order, groups = [], {}
    for src in sources.values():
        et = src.get("event_type", "Unknown")
        if et not in groups:
            group_order.append(et)
            groups[et] = []
        groups[et].append(src)

    result, i = {}, 1
    for et in group_order:
        for src in sorted(groups[et], key=lambda s: _credibility_tier(s.get("type", ""))):
            result[f"Source_{i}"] = src
            i += 1
    return result

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


def fetch_page_text(url):
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return ""
        extractor = _TextExtractor()
        extractor.feed(resp.text)
        return extractor.get_text()[:3000]
    except Exception:
        return ""

def synthesize_findings(location, start_date, end_date, pollutant, sources):
    enriched = {}
    with ThreadPoolExecutor(max_workers = 8) as executor:
        futures = {executor.submit(fetch_page_text, src.get("url", "")): (key, src)
                   for key, src in sources.items()}
        for future in as_completed(futures):
            key, src = futures[future]
            enriched[key] = {**src, "page_text": future.result()}

    prompt = f"""
        A {pollutant} spike was detected near {location} between {start_date} and {end_date}.
        The following sources were found. Each includes its metadata and up to 3000 characters
        of text fetched directly from the source URL (may be empty if the page was inaccessible).

        {json.dumps(enriched, indent = 2)}

        Analyze these sources for an environmental analyst and return ONLY a JSON object with
        these exact keys (no prose before or after the JSON):

        {{
          "most_likely_cause": "One sentence naming the most likely cause of the {pollutant} spike, citing specific source titles. Explicitly state whether the identified event would plausibly release {pollutant} or its precursors.",
          "key_evidence": ["Bullet citing source 1 — include any direct mention of {pollutant} if found", "Bullet citing source 2", "Bullet citing source 3"],
          "gaps_and_alternatives": ["Gap or alternative explanation 1", "Gap 2"],
          "confidence": "High or Medium or Low",
          "confidence_rationale": "One sentence explaining the confidence rating based on source quality and how well the evidence connects to {pollutant} specifically."
        }}

        Be concise and specific. Ground claims in the source text above; do not invent facts.
    """
    try:
        response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = [{"role": "user", "content": prompt}],
            response_format = {"type": "json_object"}
        )
    except Exception as e:
        print(f"LLM API error in synthesize_findings: {e}")
        return {}

    try:
        text = response.choices[0].message.content
        text = re.sub(r"```[a-z]*", "", text).strip()
        return json.loads(text)
    except json.JSONDecodeError:
        print("Invalid JSON in synthesis_findings()")
        return text

st.title("Pollution Event Web Searcher")
st.session_state.location = st.text_input("Location")
st.session_state.start_date = st.date_input("Start Date", min_value = datetime.date(2000, 1, 1))
st.session_state.end_date = st.date_input("End Date", min_value = datetime.date(2000, 1, 1))
st.session_state.pollutant = st.text_input("Pollutant", placeholder = "e.g. PM2.5, ethylene, VOCs")

if st.session_state.location and st.session_state.start_date and \
st.session_state.end_date and st.session_state.pollutant:
    if st.button("Search"):
        st.session_state.sources = {}

        with st.status("Searching...", expanded = True) as status:
            searched_queries = set()
            st.write(f"**Round 1** — Searching for {st.session_state.pollutant} events near {st.session_state.location}")
            new_sources = searching_llm(st.session_state.location, st.session_state.start_date,
                                st.session_state.end_date, st.session_state.pollutant, st.session_state.sources,
                                searched_queries)
            offset = len(st.session_state.sources)
            merged = dedupe_sources({**st.session_state.sources,
                      **{f"Source_{offset + i + 1}": v for i, v in enumerate(new_sources.values())}})
            st.write(f"Ranking {len(merged)} sources...")
            result = sort_sources(st.session_state.location, st.session_state.start_date,
                            st.session_state.end_date, st.session_state.pollutant, merged)
            sorted_sources = {k: v for k, v in result.items() if k != "Continue"}
            st.session_state.sources = post_sort_sources(sorted_sources) if sorted_sources else merged
            print(f"Trial 1: {st.session_state.sources}")

            # Alternate search → sort up to 5 rounds; sort_sources decides when enough sources exist
            count = 2
            while result.get("Continue", False) and count <= 5:
                st.write(f"**Round {count}** — Refining with {len(st.session_state.sources)} sources so far")
                new_sources = searching_llm(st.session_state.location, st.session_state.start_date,
                                    st.session_state.end_date, st.session_state.pollutant, st.session_state.sources,
                                    searched_queries)
                offset = len(st.session_state.sources)
                merged = dedupe_sources({**st.session_state.sources,
                          **{f"Source_{offset + i + 1}": v for i, v in enumerate(new_sources.values())}})
                st.write(f"Ranking {len(merged)} sources...")
                result = sort_sources(st.session_state.location, st.session_state.start_date,
                                    st.session_state.end_date, st.session_state.pollutant, merged)
                sorted_sources = {k: v for k, v in result.items() if k != "Continue"}
                st.session_state.sources = post_sort_sources(sorted_sources) if sorted_sources else merged
                print(f"Trial {count}: {st.session_state.sources}")
                count += 1

            status.update(label=f"Done — found {len(st.session_state.sources)} sources", state="complete", expanded=False)

        with st.spinner("Synthesizing findings..."):
            synthesis = synthesize_findings(
                st.session_state.location,
                st.session_state.start_date,
                st.session_state.end_date,
                st.session_state.pollutant,
                st.session_state.sources
            )

        st.subheader("Summary")
        if isinstance(synthesis, dict):
            st.info(f"**Most Likely Cause:** {synthesis.get('most_likely_cause', '')}")

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
            color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(confidence.upper(), "gray")
            st.markdown(
                f'<span style="color:{color};font-weight:bold">Confidence: {confidence}</span>'
                f' — {synthesis.get("confidence_rationale", "")}',
                unsafe_allow_html=True
            )
        else:
            st.info(synthesis)

        groups = {}
        for source in st.session_state.sources.values():
            et = source.get("event_type", "Unknown Event")
            if et not in groups:
                groups[et] = []
            groups[et].append(source)

        for event_type, group_sources in groups.items():
            st.header(event_type)
            for i, source in enumerate(group_sources, 1):
                st.subheader(f"Source {i}: {source['title']}")
                st.write(f"**Type:** {source['type']}")
                if source.get("date"):
                    st.write(f"**Date:** {source['date']}")
                st.write(f"**Summary:** {source['summary']}")
                st.write(f"**URL:** {source['url']}")
                st.divider()
