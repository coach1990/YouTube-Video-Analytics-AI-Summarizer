import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import traceback
from urllib.parse import urlparse, parse_qs
import isodate # For parsing ISO 8601 duration
from datetime import datetime # For formatting datetime strings
import re # For parsing SRT/VTT content (though less needed with the new library, keep parse_srt_to_text for general cleaning)

# --- Add Imports for Transcript Library ---
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
# -----------------------------------------

# --- Add Google Generative AI Import ---
# need to install this library: pip install google-generativeai
import google.generativeai as genai
# ------------------------------------

# --- Configuration & Constants ---
# IMPORTANT: Configure your API keys using Streamlit secrets: https://docs.streamlit.io/develop/concepts/secrets
# Create a file .streamlit/secrets.toml with:
# YOUTUBE_API_KEY="YOUR_ACTUAL_YOUTUBE_API_KEY"
# GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"

YOUTUBE_API_KEY = 'YOUR_GOOGLE_CONSOLE_YOUTUBE_API_KEY'

GEMINI_API_KEY = 'YOUR_GOOGLE_AI_STUDIO_API'


if not YOUTUBE_API_KEY:
    st.error("🚨 YouTube API Key is not configured in Streamlit secrets (`.streamlit/secrets.toml`). Cannot fetch video details.")
    # Don't stop, allow transcript fetching if possible, but disable detail fetching
    youtube = None # Set youtube client to None if key is missing
else:
    # --- Initialize YouTube API Client ---
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        st.error(f"🚨 Failed to initialize YouTube API client: {e}")
        youtube = None


# --- Helper Functions (Existing) ---
def extract_video_id(video_url_or_id):
    if not video_url_or_id: return None
    # Check if it's already a valid 11-character ID
    if len(video_url_or_id) == 11 and not any(c in video_url_or_id for c in " /&?="):
         return video_url_or_id
    try:
        parsed_url = urlparse(video_url_or_id)
        # Handle standard watch URL
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'): # Use standard hostnames
            if parsed_url.path == '/watch':
                p = parse_qs(parsed_url.query); return p.get('v', [None])[0]
            # Handle embed/v/shorts paths
            elif parsed_url.path.startswith(('/embed/', '/v/', '/shorts/')):
                 return parsed_url.path.split('/')[2]
        # Handle youtu.be style links (often redirects, less reliable)
        # Commenting out less reliable patterns
        # if parsed_url.hostname == 'youtu.be': return parsed_url.path[1:]
    except Exception:
        pass # Fall through if URL parsing fails
        
    # If it doesn't look like a URL or a valid ID, return None
    return None


def format_duration(iso_duration):
    if not iso_duration: return "N/A"
    try: return str(isodate.parse_duration(iso_duration))
    except Exception: return "N/A (Error Parsing Duration)"

def format_datetime_str(iso_datetime_str):
    if not iso_datetime_str: return "N/A"
    try:
        # Handle potential time zone information or lack thereof
        dt_obj = datetime.fromisoformat(iso_datetime_str.replace('Z', '+00:00')) if iso_datetime_str.endswith('Z') else datetime.fromisoformat(iso_datetime_str)
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z") # Assuming 'Z' means UTC
    except Exception: return "N/A (Error Parsing Date)"

# --- Keep parser for safety, although the library gives clean text ---
def parse_srt_to_text(srt_content):
    """
    Parses SRT or VTT content and extracts plain text.
    Removes timestamps, sequence numbers, and common formatting tags.
    (Less crucial if using youtube-transcript-api's output directly)
    """
    if not srt_content:
        return ""
    # This parser logic is primarily for raw SRT/VTT files, which
    # youtube-transcript-api saves us from having to download directly.
    # We might keep it for robustness or if the library output needs extra cleaning.
    # For now, let's assume the library's output is clean list of text segments.
    # If transcript_text is already just joined text, this function isn't needed.
    pass # Placeholder - we likely won't use this anymore with the new library output

# --- NEW Transcript Fetching using youtube-transcript-api ---
def list_transcript_languages(video_id):
    """Lists available transcript languages for a video using youtube-transcript-api."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Return a list of dictionaries with 'language', 'lang_code', 'is_generated'
        return [{'language': t.language, 'lang_code': t.language_code, 'is_generated': t.is_generated} for t in transcript_list]
    except NoTranscriptFound:
        return [] # No transcripts available
    except TranscriptsDisabled:
        st.info("ℹ️ Transcripts are disabled for this video.")
        return [] # Transcripts disabled
    except Exception as e:
        st.warning(f"Error listing transcripts: {str(e)}")
        return [] # Other errors

def get_transcript_text(video_id, language_code):
    """Fetches and joins transcript text for a given language code."""
    try:
        # The library fetches and returns a list of dictionaries {'text': '...', 'start': ..., 'duration': ...}
        transcript_segments = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
        
        # Join the 'text' components to get the full plain text transcript
        transcript_text = " ".join([segment['text'] for segment in transcript_segments])
        return transcript_text
    except Exception as e:
        st.error(f"❌ Error fetching or processing transcript: {str(e)}")
        st.code(traceback.format_exc())
        return None

# --- API Fetching Functions (Existing, using 'youtube' client) ---
# These functions still use the google-api-python-client and require YOUTUBE_API_KEY
# They are only called if the 'youtube' client was successfully initialized
# (i.e., if YOUTUBE_API_KEY was found in secrets)

def fetch_video_category_name(category_id, yt_service):
    if not yt_service or not category_id: return "N/A"
    try:
        request = yt_service.videoCategories().list(part="snippet", id=category_id)
        response = request.execute()
        return response["items"][0]["snippet"]["title"] if response.get("items") else "N/A (Unknown ID)"
    except Exception: return "N/A (Error)"

def fetch_video_details(video_id, yt_service):
    if not yt_service: return None # Don't proceed if youtube client failed to initialize
    try:
        parts = ["snippet", "contentDetails", "statistics", "status", "liveStreamingDetails", "topicDetails", "recordingDetails"]
        request = yt_service.videos().list(part=",".join(parts), id=video_id)
        response = request.execute()
        if not response.get("items"): st.error(f"❌ Video not found or API error for ID: {video_id}"); return None
        
        data = response["items"][0]
        snippet, content, stats, status_info, live, topic, recording = (
            data.get(p, {}) for p in ["snippet", "contentDetails", "statistics", "status", "liveStreamingDetails", "topicDetails", "recordingDetails"]
        )
        
        cat_name = fetch_video_category_name(snippet.get("categoryId"), yt_service)
        # Safely get integer values, defaulting to 0 if key is missing
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
        favorites = int(stats.get("favoriteCount", 0)) if "favoriteCount" in stats else 0


        details = {
            "Basic Info": {"ID": video_id, "Title": snippet.get("title"), "Published": format_datetime_str(snippet.get("publishedAt")), "Category": cat_name, "Tags": ", ".join(snippet.get("tags",[])) or "N/A", "Description": snippet.get("description")},
            "Language": {"Default Lang": snippet.get("defaultLanguage"), "Audio Lang": snippet.get("defaultAudioLanguage")},
            "Channel": {"ID": snippet.get("channelId"), "Title": snippet.get("channelTitle")},
            "Thumbnails": {k: v.get("url") for k, v in snippet.get("thumbnails", {}).items()},
            "Content": {"Duration": format_duration(content.get("duration")), "Dimension": content.get("dimension"), "Definition": content.get("definition"), "Caption (API Status)": content.get("caption"), "Licensed": content.get("licensedContent"), "Rating": str(content.get("contentRating", {})), "Region Restriction": str(content.get("regionRestriction"))},
            "Statistics": {"Views": views, "Likes": likes, "Favorites": favorites, "Comments": stats.get("commentCount", "N/A"), "Like Ratio": f"{(likes/views)*100:.2f}%" if views else "N/A", "Comment Ratio": f"{(comments/views)*100:.2f}%" if views and "commentCount" in stats else "N/A"},
            "Status": {"Upload": status_info.get("uploadStatus"), "Privacy": status_info.get("privacyStatus"), "License": status_info.get("license"), "Embeddable": status_info.get("embeddable"), "Public Stats": status_info.get("publicStatsViewable"), "For Kids": status_info.get("madeForKids")},
            "Live": {"Status": snippet.get("liveBroadcastContent"), "Scheduled Start": format_datetime_str(live.get("scheduledStartTime")), "Actual Start": format_datetime_str(live.get("actualStartTime")), "Actual End": format_datetime_str(live.get("actualEndTime")), "Viewers": live.get("concurrentViewers")},
            "Topics": {"IDs": ", ".join(topic.get("relevantTopicIds",[])), "Categories": topic.get("topicCategories",[])},
            "Recording": {"Date": format_datetime_str(recording.get("recordingDate")), "Location Desc": recording.get("locationDescription"), "Lat": recording.get("location",{}).get("latitude"), "Lon": recording.get("location",{}).get("longitude")}
        }
        return details
    except HttpError as e: st.error(f"❌ API Error fetching details: {e.resp.status} {e._get_reason()}"); st.json(e.content); return None
    except Exception as e: st.error(f"❌ Unexpected error fetching details: {e}"); st.code(traceback.format_exc()); return None

def fetch_comments_sample(video_id, yt_service, max_results=5):
    if not yt_service: return ["YouTube API key not available to fetch comments."]
    comments_list = []
    try:
        # Check if comments are allowed
        video_details = yt_service.videos().list(part="status", id=video_id).execute()
        if not video_details.get('items') or not video_details['items'][0]['status'].get('commentAllowed', True):
             return ["Comments are disabled for this video."]

        request = yt_service.commentThreads().list(part="snippet",videoId=video_id,textFormat="plainText",maxResults=max_results,order="relevance")
        response = request.execute()
        for item in response.get("items", []): comments_list.append(f"**{item['snippet']['topLevelComment']['snippet']['authorDisplayName']}:** {item['snippet']['topLevelComment']['snippet']['textDisplay']}")
        return comments_list if comments_list else ["No comments found (though comments are allowed)."]
    except HttpError as e:
         if e.resp.status == 403: # Likely API quota or comments disabled at video/channel level
             return ["Error fetching comments: API Quota exceeded or comments potentially disabled."]
         else:
             return [f"Error fetching comments: {e.resp.status} {e._get_reason()}"]
    except Exception: return ["Error fetching comments."]


def fetch_related_videos(video_id, yt_service, max_results=5):
    if not yt_service: return ["YouTube API key not available to fetch related videos."]
    related_videos_list = []
    try:
        # Use the search().list method with relatedToVideoId
        request = yt_service.search().list(part="snippet", relatedToVideoId=video_id, type="video", maxResults=max_results)
        response = request.execute()
        for item in response.get("items", []):
            # Construct a clickable link
            video_link = f"https://www.youtube.com/watch?v={item['id']['videoId']}" # Use standard YouTube URL
            related_videos_list.append(f"- [{item['snippet']['title']}]({video_link}) by *{item['snippet']['channelTitle']}*")
        return related_videos_list if related_videos_list else ["No related videos found."]
    except HttpError as e:
        st.warning(f"API Error fetching related videos: {e.resp.status} {e._get_reason()}")
        return ["Error fetching related videos due to API error."]
    except Exception: return ["Error fetching related videos."]


# --- AI Summarization Function ---
def summarize_transcript(transcript_text, api_key):
    """Summarizes the transcript using a Generative AI model (like Gemini(I used Gemini), chatGPT, co-Pilot)."""
    if not api_key:
        return "Error: Gemini API Key is not configured in Streamlit secrets (`.streamlit/secrets.toml`)."
    if not transcript_text:
        return "Error: No transcript text available to summarize."

    try:
        genai.configure(api_key=api_key)

        # --- Model Availability and Selection ---
        model_name_to_use = None
        
        # Define a prioritized list of model names suitable for text tasks (without 'models/' prefix)
        # Order matters: try the most recommended/capable ones first
        prioritized_model_names = [
            'gemini-1.5-flash-latest', # As suggested by the error message as a good alternative
            'gemini-1.5-pro-latest',
            'gemini-1.0-pro-latest',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-1.0-pro',
            'gemini-pro', # Keep the original preferred name as a possibility
            'gemini-pro-vision', # Sometimes vision models can process text too, as a fallback
        ]

        available_models_names = [] # Store just the names like 'gemini-1.5-flash'
        try:
            st.text("Listing available Gemini models for text generation...")
            # List models and filter for those supporting generateContent (for text generation)
            all_available_models = list(genai.list_models())

            available_models_names_with_prefix = [
                m.name for m in all_available_models
                if 'generateContent' in m.supported_generation_methods
            ]
            # Convert to just the model names without 'models/' for easier matching
            available_models_names = [name.split('/')[-1] for name in available_models_names_with_prefix]


            # Log available models for debugging - show simple names
            st.sidebar.json(available_models_names) # Optional: show list in sidebar
            st.text(f"Found {len(available_models_names)} models supporting generateContent.")

        except Exception as e:
            st.error(f"Failed to list available Gemini models. Please check your API key and network connection: {e}")
            return f"Error checking model availability: {e}"

        if not available_models_names:
             return "Error: No Gemini models supporting generateContent found for your API key. Check Google AI Studio for model access."

        # Iterate through the prioritized list to find the first available model
        for preferred_name in prioritized_model_names:
            if preferred_name in available_models_names:
                 model_name_to_use = preferred_name
                 st.info(f"Using model: '{model_name_to_use}' for summarization.")
                 break # Found a suitable model, stop searching

        if not model_name_to_use:
             # If none of the prioritized models were found
             available_simple_names = [name.split('/')[-1] for name in available_models_names_with_prefix]
             return f"Error: None of the preferred models ({', '.join(prioritized_model_names)}) available. Available models supporting generateContent: {', '.join(available_simple_names)}"

        # --- End Model Availability and Selection ---

        # Initialize the model with the determined name (add 'models/' prefix for initialization)
        # Use the full name obtained from the list
        full_model_name = f'models/{model_name_to_use}'


        model = genai.GenerativeModel(full_model_name)

        # ... rest of summarization logic (prompt, chunking, generate_content call) ...
        prompt = f"Please summarize the following video transcript concisely:\n\n{transcript_text}"

        # Handle Long Transcripts (Basic Chunking Example)
        # Adjust max_chunk_size based on the specific model's context window (1.5 models are very large)
        # might need a more sophisticated chunking strategy that respects sentence boundaries.
        # A common recommendation is to stay well within the model's context window.
        # For gemini-1.5-flash or pro, the context window is huge (1M tokens), so 30k chars should be fine for most transcripts.
        max_chunk_size = 30000 # chunk size in characters (roughly)

        chunks = [transcript_text[i:i+max_chunk_size] for i in range(0, len(transcript_text), max_chunk_size)]

        summary = "" # Initialize summary variable

        if len(chunks) > 1:
             st.info(f"Transcript is long ({len(transcript_text)} chars), splitting into {len(chunks)} chunks for summarization.")
             summaries = []
             for i, chunk in enumerate(chunks):
                  # Add a progress indicator (optional, but helpful for long jobs)
                  st.text(f"Summarizing chunk {i+1}/{len(chunks)}...")
                  chunk_prompt = f"Summarize this part of a video transcript:\n\n{chunk}"
                  chunk_response = model.generate_content(chunk_prompt)
                  # Ensure response has parts and text attribute before joining
                  chunk_summary = "".join([part.text for part in chunk_response.parts if hasattr(part, 'text')])
                  summaries.append(chunk_summary)

             if summaries:
                 st.text("Combining chunk summaries...")
                 # Summarize the summaries if there are multiple
                 final_summary_prompt = "Here are summaries of different parts of a video transcript. Combine them into a single, coherent summary:\n\n" + "\n\n---\n\n".join(summaries)
                 final_response = model.generate_content(final_summary_prompt)
                 summary = "".join([part.text for part in final_response.parts if hasattr(part, 'text')])
             else:
                 st.error("Could not generate summaries for the individual chunks.")


        elif len(chunks) == 1 and chunks[0]: # Handle the case where there's exactly one non-empty chunk
             st.text("Summarizing the transcript...")
             response = model.generate_content(prompt)
             summary = "".join([part.text for part in response.parts if hasattr(part, 'text')])
        else: # Handle empty transcript case (should be caught earlier, but safety)
             return "Error: Transcript is empty."


        if summary:
             return summary
        else:
             # If summary is empty string, it might still be a soft error from the API
             st.error("AI model returned no text for the summary.")
             return "Error: AI model returned no text for the summary."

    except Exception as e:
        # Catch potential API errors, token limit errors, etc.
        return f"Error during summarization: {str(e)}\nTraceback: {traceback.format_exc()}"
# --- Streamlit UI ---
st.set_page_config(page_title="YouTube Video Analytics & Tools", layout="wide")
st.title("🎬 YouTube Video Analytics & Tools")
st.markdown("Enter a YouTube Video URL or ID to fetch metadata, available transcripts, and summarize them.")

video_url_or_id = st.text_input("Enter YouTube Video URL or ID:", key="video_input", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ or dQw4w9WgXcQ")

# Initialize session state for transcript and summary
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'available_tracks' not in st.session_state: # Store available tracks from the new library
    st.session_state.available_tracks = []
if 'selected_language_code' not in st.session_state: # Store selected language code
     st.session_state.selected_language_code = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'last_processed_video_id' not in st.session_state: # Track last ID to clear state
    st.session_state.last_processed_video_id = None


if video_url_or_id:
    video_id = extract_video_id(video_url_or_id)

    # Clear state if a new video ID is entered
    if 'current_video_id' not in st.session_state or st.session_state.current_video_id != video_id:
         st.session_state.transcript = None
         st.session_state.available_tracks = []
         st.session_state.selected_language_code = None
         st.session_state.summary = None
         st.session_state.current_video_id = video_id # Update current ID

    if not video_id:
        st.error("❌ Invalid YouTube URL or Video ID.")
        # Also clear state if input becomes invalid
        st.session_state.transcript = None
        st.session_state.available_tracks = []
        st.session_state.selected_language_code = None
        st.session_state.summary = None
    else:
        # Display video embed
        st.video(f"https://www.youtube.com/watch?v={video_id}") # Use standard embed URL

        # --- Video Details Section (Only if YouTube API is available) ---
        if youtube: # Check if the YouTube client was successfully initialized
            with st.spinner(f"🔍 Fetching analytics for video ID: {video_id}..."):
                all_details = fetch_video_details(video_id, youtube)
            if all_details:
                st.success(f"✨ Analytics loaded for '{all_details.get('Basic Info', {}).get('Title', 'N/A')}'")
                for section, data in all_details.items():
                    if data and (not isinstance(data, dict) or any(v for k,v in data.items() if k != "Description") or (isinstance(data, dict) and data.get("Description"))):
                        with st.expander(f"📊 {section} Details", expanded=(section in ["Basic Info", "Statistics"])):
                            if isinstance(data, dict):
                                for k, v in data.items():
                                    if k == "Description" and v:
                                        st.markdown(f"**{k}:**")
                                        st.text_area("", v, height=150, disabled=True, label_visibility="collapsed")
                                    elif k == "Thumbnails" and isinstance(v, dict):
                                        st.markdown(f"**{k}:**")
                                        cols = st.columns(min(len(v),4))
                                        idx=0
                                        for ts, tu in v.items(): 
                                            if tu and idx < len(cols):
                                                 cols[idx].image(tu, caption=ts, use_column_width=True); idx+=1
                                    elif k == "Categories" and isinstance(v, list) and v != ["N/A"]:
                                        st.markdown(f"**{k}:**")
                                        for link_url in v:
                                             cat_name = link_url.split('/')[-1].replace('_', ' ').title()
                                             st.markdown(f"- [{cat_name}]({link_url})")
                                    elif v is not None and v != "":
                                         st.markdown(f"**{k}**: {v}")
                            else: st.markdown(str(data))
            else:
                st.warning("Could not retrieve details for the video using YouTube API.")
        else:
             st.warning("Skipping YouTube API detail fetching because API key is not configured.")


        st.markdown("---")
        # --- Transcript and Summarizer Section (Uses youtube-transcript-api) ---
        st.subheader("📜 Video Transcript & Summarizer")
        
        # List available languages (only fetch once per video ID)
        if not st.session_state.available_tracks or st.session_state.last_processed_video_id != video_id:
             st.session_state.available_tracks = list_transcript_languages(video_id)
             st.session_state.last_processed_video_id = video_id # Mark this ID as processed for tracks


        if not st.session_state.available_tracks:
            st.info("No caption tracks found for this video or transcripts are disabled.")
            st.session_state.transcript = None # Ensure state is clear
            st.session_state.summary = None
        else:
            # Create options for the selectbox {Display Name: Language Code}
            track_options = {
                 f"{t['language']} ({t['lang_code']}) {'(Generated)' if t['is_generated'] else ''}": t['lang_code']
                 for t in st.session_state.available_tracks
            }

            # Determine the default index
            default_index = 0
            if st.session_state.selected_language_code and st.session_state.selected_language_code in track_options.values():
                 default_index = list(track_options.values()).index(st.session_state.selected_language_code)
            # Optionally prioritize non-generated ('standard') English if available
            elif any(t['lang_code'] == 'en' and not t['is_generated'] for t in st.session_state.available_tracks):
                 try:
                      # Find the index of the first non-generated English track
                      default_index = next(i for i, t in enumerate(st.session_state.available_tracks) if t['lang_code'] == 'en' and not t['is_generated'])
                 except StopIteration: # Should not happen if 'any' was True
                      pass
            # Optionally prioritize any English (generated or not)
            elif any(t['lang_code'] == 'en' for t in st.session_state.available_tracks):
                 try:
                      # Find the index of the first English track
                      default_index = next(i for i, t in enumerate(st.session_state.available_tracks) if t['lang_code'] == 'en')
                 except StopIteration:
                      pass
            # Otherwise, default to the first track in the list


            selected_track_display = st.selectbox(
                "Select available transcript language:",
                options=list(track_options.keys()), # Display names
                index=default_index,
                key="track_selector"
            )
            
            # Get the language code for the selected display name
            st.session_state.selected_language_code = track_options[selected_track_display]

            # Button to fetch transcript
            if st.button("📄 Fetch Transcript", key="fetch_transcript_btn"):
                st.session_state.transcript = None # Reset previous transcript
                st.session_state.summary = None # Reset previous summary
                if st.session_state.selected_language_code:
                    with st.spinner(f"Fetching and parsing transcript for language '{st.session_state.selected_language_code}'..."):
                        transcript_text = get_transcript_text(video_id, st.session_state.selected_language_code)
                        if transcript_text:
                            st.session_state.transcript = transcript_text
                            st.success("Transcript fetched successfully!")
                        else:
                            st.error("Could not retrieve or parse the transcript for the selected language.")
                else:
                    st.warning("Please select a caption track first.")

            # Display the fetched transcript if available in session state
            if st.session_state.transcript:
                st.markdown("#### Full Video Transcript:")
                st.text_area("Transcript", st.session_state.transcript, height=300, key="transcript_display")

                # --- Summarization Button and Logic ---
                if st.button("💡 Summarize Transcript (AI)", key="summarize_btn"):
                    if not GEMINI_API_KEY:
                        st.error("🚨 Gemini API Key is not configured in Streamlit secrets (`.streamlit/secrets.toml`). Cannot perform summarization.")
                    elif st.session_state.transcript:
                        st.session_state.summary = None # Reset previous summary
                        with st.spinner("Generating summary with AI..."):
                            # Call the summarization function
                            summary = summarize_transcript(st.session_state.transcript, GEMINI_API_KEY)
                            st.session_state.summary = summary # Store result

                        if st.session_state.summary and not st.session_state.summary.startswith("Error"):
                            st.success("Summary generated!")
                        else:
                             st.error(f"Failed to generate summary: {st.session_state.summary}")
                    else:
                         st.warning("No transcript available to summarize. Please fetch transcript first.")


                # Display the generated summary if available in session state
                if st.session_state.summary:
                     st.markdown("#### ✨ AI Summary:")
                     st.write(st.session_state.summary)
                # --- End Summarization Logic ---


        st.markdown("---")
        # --- Other Existing Sections (Comments, Related Videos) ---
        # Use expanders for comments and related videos, only if youtube client is available
        if youtube:
            with st.expander("💬 Comments Sample (max 5)"):
                # Fetch comments only when expander is potentially viewed or video changes
                if 'comments_sample' not in st.session_state or st.session_state.last_processed_video_id != video_id:
                     st.session_state.comments_sample = fetch_comments_sample(video_id, youtube)
                
                for c in st.session_state.comments_sample:
                     st.markdown(c)

            with st.expander("🔗 Related Videos (max 5)"):
                 # Fetch related videos only when expander is potentially viewed or video changes
                 if 'related_videos' not in st.session_state or st.session_state.last_processed_video_id != video_id:
                      st.session_state.related_videos = fetch_related_videos(video_id, youtube)
                
                 for r in st.session_state.related_videos:
                      st.markdown(r)
        else:
             st.info("Comments and Related Videos sections skipped because YouTube API key is not configured.")


st.markdown("---")
st.caption(f"YouTube Analytics & Tools | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
