# YouTube-Video-Analytics-AI-Summarizer
A Streamlit web application for fetching YouTube video analytics, extracting available transcripts, and generating AI summaries using Google Gemini.
This project is licensed under the GNU General Public License v3.0 License - see the (LICENSE) file for details.

## Project Description
Please watch ref_video.mp4
This project is a web application built with Streamlit that allows users to:

1.  Fetch comprehensive metadata and analytics for any public YouTube video using the YouTube Data API v3.
2.  List available transcript languages for the video.
3.  Extract and display the plain text transcript for a selected language using the `youtube-transcript-api` library.
4.  Generate a concise AI-powered summary of the video transcript using the Google Gemini API.

It's a useful tool for quickly getting insights into YouTube videos and summarizing lengthy content.

## Features

* Embed and display the YouTube video.
* Show detailed video analytics (views, likes, duration, category, tags, etc.).
* List available caption/transcript tracks (including auto-generated ones).
* Fetch and display the full video transcript.
* Summarize the fetched transcript using a Generative AI model (Google Gemini).
* Display a sample of recent comments (requires YouTube API key).
* Show related videos (requires YouTube API key).

## Technologies Used

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web application UI.
* **Google API Client Library:** To interact with the YouTube Data API v3 for video details.
* **youtube-transcript-api:** To reliably fetch available video transcripts without requiring OAuth.
* **Google Generative AI SDK (`google-generativeai`):** To interact with the Google Gemini API for text summarization.
* **`isodate`:** For parsing ISO 8601 durations.
* **`urllib.parse` & `re`:** For URL parsing and text cleaning.

## Setup and Installation

Follow these steps to get the project running on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```
    (Replace `<your-repo-url>` and `<your-repo-folder>` with your GitHub repository details).

2.  **Create a virtual environment:** It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    # On Windows
    python -m venv .venv

    # On macOS/Linux
    python3 -m venv .venv
    ```

3.  **Activate the virtual environment:**
    ```bash
    # On Windows
    .venv\Scripts\activate

    # On macOS/Linux
    source .venv/bin/activate
    ```
    Your terminal prompt should change to indicate the virtual environment is active (e.g., `(.venv) your-folder>`).

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file, you can create one by running `pip freeze > requirements.txt` after manually installing the packages or by manually listing them in the file. The main packages needed are:
    * `streamlit`
    * `google-api-python-client`
    * `isodate`
    * `google-generativeai`
    * `youtube-transcript-api`

## Configuration (API Keys)

This application requires API keys for both YouTube Data API and Google Gemini API. It's recommended to store these securely using Streamlit's secrets management.

1.  **Get a YouTube Data API v3 Key:**
    * Go to the [Google Cloud Console](http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu0).
    * Create a new project.
    * Enable the "YouTube Data API v3" for that project.
    * Go to "Credentials" and create an "API Key".
    * Restrict the API key to the "YouTube Data API v3" for security.

2.  **Get a Google Gemini API Key:**
    * Go to [Google AI Studio](http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu1).
    * Sign in and click "Get API key".
    * Create an API key.

3.  **Configure Streamlit Secrets:**
    * In the root directory of your project (where your script and the `.venv` folder are), create a folder named `.streamlit`.
    * Inside the `.streamlit` folder, create a file named `secrets.toml`.
    * Add your API keys to `secrets.toml` in the following format:

    ```toml
    YOUTUBE_API_KEY="YOUR_ACTUAL_YOUTUBE_API_KEY"
    GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
    ```
    Replace the placeholder values with the actual keys you obtained. Keep the quotation marks.

## How to Run

With your virtual environment activated and API keys configured, run the Streamlit application from your terminal:

```bash
streamlit run youtube_metadata_viewer.py
