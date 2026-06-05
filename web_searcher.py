import streamlit as st
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import requests

load_dotenv()

client = OpenAI(api_key=os.getenv("api_key"), base_url = "https://api.deepseek.com")

if "location" not in st.session_state:
    st.session_state.location = ""
if "start_date" not in st.session_state:
    st.session_state.start_date = ""
if "end_date" not in st.session_state:
    st.session_state.end_date = ""
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
    
def searching_llm(location, start_date, end_date, existing_sources):
    messages = [
        {
            "role": "user",
            "content": f'''
                A pollutant spike was detected near {location} between {start_date} and {end_date}.
        
                Here are the sources already found so far:
                {json.dumps(existing_sources)}
        
                Your job is to search the web for ANY ADDITIONAL events that could explain this spike, such as:
                - Wildfires or prescribed burns
                - Industrial accidents or chemical spills
                - Power plant emissions
                - Any other unusual event that could release pollutants

                Do not search for or return sources already in the list above.
                Use the search tool to find relevant sources. Search multiple times with different
                queries if needed.
                
                You MUST use the search tool at least once before returning any sources.
                Do not rely on your own knowledge. Always search the web first.

                When done, return a JSON dictionary of ALL sources (existing + newly found), each with:
                - title
                - url
                - type (e.g. Government Report, News Outlet, Social Media, etc.)
                - summary (one sentence of what the source says)

                Do not write any explanatory text before or after the JSON.
                Do not acknowledge the task or describe what you are doing.
                Immediately begin searching using the search tool, and when done return ONLY the JSON.
                
                After you have finished all your searches, you MUST return a JSON dictionary of all sources found,
                with keys "Source_1", "Source_2", etc. Do not stop without returning this JSON.
                The JSON must be the last thing you output, with no text before or after it.
            '''
        }
    ]
    
    search_count = 0

    # Agentic loop: keep calling the model until it stops issuing tool calls and
    # returns the final JSON. Each iteration either executes search tool calls and
    # appends the results to the message history, or breaks out when the model
    # produces a plain text (JSON) response.
    while True:
        response = client.chat.completions.create(
            model = "deepseek-reasoner",
            messages = messages,
            tools = tools,
            response_format={"type": "json_object"}
        )

        text = response.choices[0].message

        if text.tool_calls:
            # content must be None (not omitted) when tool_calls are present per the API spec
            messages.append({"role": "assistant", "content": None, "tool_calls": text.tool_calls})

            for tool_call in text.tool_calls:
                query = json.loads(tool_call.function.arguments)["query"]
                print(f"Searching for: {query}")
                results = search_web(query)
                print(f"Got {len(results)} results")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(results)
                })

            search_count += 1

            # Safety cap: nudge the model to wrap up instead of looping forever
            if search_count >= 10:
                messages.append({"role": "user", "content": "You have reached the maximum number of searches. Please return the JSON now."})
        else:
            print(f"Raw response searching llm: '{response.choices[0].message.content}'")
            try:
                text = response.choices[0].message.content
                # Strip markdown code fences the model sometimes wraps around JSON
                text = re.sub(r"```[a-z]*", "", text).strip()
                return json.loads(text)
            except json.JSONDecodeError:
                print("Invalid JSON in searching_llm, skipping")
                return {}
        
    
def sort_sources(location, start_date, end_date, sources):
    prompt = f'''
        A pollutant spike was detected near {location} between {start_date} and {end_date}. Here are
        a list of current sources:
        
        {json.dumps(sources)}
        
        Your job is to only return a JSON object of these sources, with keys Source_1, Source_2, Source_3,
        etc., but ranked from most to least credible. 
        
        For example, the hierarchy of credibility could be:
        Government reports > Academic/peer-reviewed > Established news > Local news > Blogs/Non-profits > Wikipedia > Social media
        
        At the end of the JSON, there should be another key, 'Continue', which is either true or false. If true, then
        another LLM will search further, update the sources list, and you will be called again to sort them. If false,
        then we will stop searching. For example, you may continue searching if we only have very little sources, and you
        may stop searching if you believe there are enough sources already, or continuing to look for any will be less meaningful.
        Stop searching if there are already 8+ sources, or if the top sources are already high credibility government/academic ones.
        
        Do not search for new sources.
        Your job is to only sort the current existing ones.
        
        Do not write any explanatory text before or after the JSON.
        Do not acknowledge the task or describe what you are doing.
        Immediately begin searching using the search tool, and when done return ONLY the JSON.
        
        Example JSON (always follow this exact format):
        {{
            "Source_1": {{
                "title": "California Air Resources Board - Incident Report: Dixie Fire Emissions",
                "url": "https://ww2.arb.ca.gov/...",
                "type": "Government Report",
                "summary": "Official emissions data recorded during the Dixie Fire, July-August 2021."
            }},
            "Source_2": {{
                "title": "EPA AirNow Fire and Smoke Map - Event Log",
                "url": "https://www.airnow.gov/...",
                "type": "Government Database",
                "summary": "Air quality index readings spiking in the region during the relevant period."
            }},
            "Source_3": {{
                "title": "Reuters - Massive wildfire spreads across Northern California",
                "url": "https://www.reuters.com/...",
                "type": "News Outlet",
                "summary": "News coverage reporting the fire's spread and affected counties."
            }},
            "Source_4": {{
                "title": "Reddit post - r/California - Anyone else seeing crazy smoke?",
                "url": "https://www.reddit.com/...",
                "type": "Social Media",
                "summary": "User reports of heavy smoke in the area around the same dates."
            }},
            "Continue": true
        }}
    '''
    
    response = client.chat.completions.create(
        model = "deepseek-reasoner",
        messages = [{
            "role": "user",
            "content": prompt
        }],
        response_format={"type": "json_object"}
    )
    
    try:
        text = response.choices[0].message.content
        text = re.sub(r"```[a-z]*", "", text).strip()
        print(f"Raw response sort sources: {text}")
        return json.loads(text)
    except json.JSONDecodeError:
        print("Invalid JSON in sort sources, skipping")
        return {}
    
st.title("Pollution Event Web Searcher")
st.session_state.location = st.text_input("Location")
st.session_state.start_date = st.date_input("Start Date")
st.session_state.end_date = st.date_input("End Date")

if st.session_state.location and st.session_state.start_date and \
st.session_state.end_date:
    if st.button("Search"):
        placeholder = st.empty()
        placeholder.write("Searching...")
        
        st.session_state.sources = {}
    
        sources = searching_llm(st.session_state.location, st.session_state.start_date,
                            st.session_state.end_date, st.session_state.sources)
        result = sort_sources(st.session_state.location, st.session_state.start_date,
                        st.session_state.end_date, sources)
        # sort_sources includes a "Continue" key; strip it before storing/displaying
        st.session_state.sources = {k: v for k, v in result.items() if k != "Continue"}
        print(f"Trial 1: {st.session_state.sources}")

        # Alternate search → sort up to 5 rounds; sort_sources decides when enough sources exist
        count = 2
        while result.get("Continue", False) and count <= 5:
            sources = searching_llm(st.session_state.location, st.session_state.start_date,
                                st.session_state.end_date, st.session_state.sources)
            result = sort_sources(st.session_state.location, st.session_state.start_date,
                                st.session_state.end_date, sources)
            st.session_state.sources = {k: v for k, v in result.items() if k != "Continue"}
            print(f"Trial {count}: {st.session_state.sources}")
            count += 1
            
        placeholder.write("Done!")
        
        for key, source in st.session_state.sources.items():
            st.subheader(f"{key}: {source['title']}")
            st.write(f"**Type:** {source['type']}")
            st.write(f"**Summary:** {source['summary']}")
            st.write(f"**URL:** {source['url']}")
            st.divider()
