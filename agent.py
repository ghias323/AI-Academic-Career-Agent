
import os
import re
import json
import hashlib
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import google.auth
import gspread
from google import genai
from tavily import TavilyClient


# ============================================================
# 1. CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PROFILE = """
Candidate profile:
- Age: 38
- Nationality: Pakistani
- Domicile: Punjab, Pakistan
- Qualification: MPhil Computer Science
- Experience: More than 14 years of teaching
- Current position: Contract Lecturer in Computer Science
  at COMSATS University Islamabad, Sahiwal Campus.

Teaching and academic areas:
- Programming Fundamentals (C/C++/Java)
- Object-Oriented Programming
- Data Structures and Algorithms
- Mobile Application Development
- Database Management Systems
- Artificial Intelligence
- Compiler Construction
- Automata and Computation
- Web Programming

Job preferences:
- Computer Science Lecturer
- Lecturer in Computing / Software Engineering
- Assistant Professor Computer Science, where eligible
- Relevant academic and teaching positions worldwide

Scholarship preferences:
- Fully funded PhD scholarships only
- Computer Science, Artificial Intelligence, Deep Learning,
  NLP, Generative AI, Healthcare AI, Explainable AI,
  Process Model Verification and Model Checking
- IELTS waiver possibilities are welcome.
- Age limits and restrictions must be checked carefully.
"""

JOB_QUERIES = [
    'Computer Science Lecturer faculty vacancies university 2026 apply deadline',
    'Assistant Professor Computer Science academic jobs international 2026',
    'Lecturer Computing Software Engineering university jobs worldwide 2026',
    'Computer Science lecturer jobs Pakistan university 2026 official',
]

SCHOLARSHIP_QUERIES = [
    'fully funded PhD Computer Science scholarships 2026 international students',
    'fully funded PhD Artificial Intelligence deep learning scholarship 2026',
    'funded PhD NLP generative AI healthcare AI international students 2026',
    'university PhD Computer Science scholarship tuition stipend 2026',
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

tavily = TavilyClient(api_key=TAVILY_API_KEY)
gemini = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# 2. GOOGLE SHEETS CONNECTION
# ============================================================

def connect_sheets():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets"
    ]

    credentials, _ = google.auth.default(scopes=scopes)
    client = gspread.authorize(credentials)

    spreadsheet = client.open_by_key(SPREADSHEET_ID)

    jobs_sheet = spreadsheet.worksheet("Academic Jobs")
    scholarships_sheet = spreadsheet.worksheet("Scholarships")

    logging.info("Connected to Google Sheets: %s", spreadsheet.title)

    return jobs_sheet, scholarships_sheet


# ============================================================
# 3. SEARCH WEB USING TAVILY
# ============================================================

def search_web(query):
    try:
        logging.info("Searching: %s", query)

        response = tavily.search(
            query=query,
            search_depth="advanced",
            topic="general",
            max_results=6,
            include_raw_content="markdown"
        )

        return response.get("results", [])

    except Exception as error:
        logging.exception("Search failed: %s", error)
        return []


def collect_search_results(queries):
    collected = {}

    for query in queries:
        results = search_web(query)

        for result in results:
            url = result.get("url", "").strip()

            if not url or url in collected:
                continue

            collected[url] = {
                "title": result.get("title", ""),
                "url": url,
                "content": result.get("raw_content")
                           or result.get("content", ""),
            }

    logging.info("Unique search results collected: %s", len(collected))

    return list(collected.values())


# ============================================================
# 4. GEMINI ANALYSIS
# ============================================================

def extract_json(text):
    """Extract JSON even if the model wraps it in Markdown."""
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def analyse_result(item, category):
    if category == "job":
        fields = """
        Return these fields:
        job_title, organization, country, city, department,
        position_type, required_qualification, experience_required,
        age_limit, salary, deadline, application_url,
        official_source_url, eligibility_status, age_relaxation,
        verification_status, notes.
        """
        task = """
        Assess whether this is a real Computer Science or closely
        related academic teaching/faculty vacancy. Compare the
        requirements with the candidate profile. Do not assume
        that age relaxation is available. Flag it for verification
        if the advertisement or applicable rules are unclear.
        """
    else:
        fields = """
        Return these fields:
        scholarship_name, university, country, degree_level,
        research_area, funding_type, tuition_coverage,
        monthly_stipend, ielts_requirement, age_limit,
        eligibility_status, deadline, application_url,
        official_source_url, verification_status, notes.
        """
        task = """
        Include only PhD opportunities that appear fully funded.
        Check tuition coverage and stipend information. If full
        funding is not clearly established, do not classify it as
        fully funded. Assess whether the candidate's Computer
        Science background and research interests are relevant.
        """

    prompt = f"""
You are an academic recruitment and scholarship research analyst.

Candidate profile:
{PROFILE}

Task:
{task}

Analyse the following search result.

Title: {item['title']}
URL: {item['url']}

Available webpage content:
{item['content'][:12000]}

Instructions:
- Extract only information supported by the supplied content.
- Never invent deadlines, salaries, age limits, requirements or links.
- Use "Not stated" when a detail is unavailable.
- Use "Insufficient Information" when eligibility cannot be assessed.
- Eligibility must be one of:
  Eligible, Potentially Eligible, Not Eligible, Insufficient Information.
- Verification status must be one of:
  Verified, Needs Review, Expired, Unknown.
- A search-result snippet alone is not sufficient to mark a vacancy
  or scholarship as Verified.
- Use the supplied source URL if no separate official URL is provided.
- If the page is not an official university, government or scholarship
  provider page, mark it Needs Review.
- Do not claim that an opportunity is open unless the page supports it.
- Return one JSON object, not Markdown.

{fields}
"""

    try:
        response = gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        )

        data = extract_json(response.text)

        if not isinstance(data, dict):
            raise ValueError("Gemini did not return a JSON object")

        return data

    except Exception as error:
        logging.exception("Gemini analysis failed: %s", error)
        return None


# ============================================================
# 5. BASIC SOURCE AND RECORD VALIDATION
# ============================================================

OFFICIAL_DOMAIN_HINTS = (
    ".edu", ".ac.", ".gov", ".edu.pk", ".gov.pk"
)


def source_looks_official(url):
    try:
        host = urlparse(url).netloc.lower()
        return any(hint in host for hint in OFFICIAL_DOMAIN_HINTS)
    except Exception:
        return False


def clean(value):
    if value is None:
        return "Not stated"

    value = str(value).strip()

    return value if value else "Not stated"


def make_record_id(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def prepare_record(data, source_url, category):
    if not data:
        return None

    data["official_source_url"] = clean(
        data.get("official_source_url", source_url)
    )

    # Keep the actual search-result URL as the evidence link.
    evidence_url = source_url

    if not source_looks_official(evidence_url):
        data["verification_status"] = "Needs Review"
    elif data.get("verification_status") == "Verified":
        # The agent has not independently checked every application
        # requirement. Keep a human-review safeguard.
        data["verification_status"] = "Needs Review"

    data["source_url"] = evidence_url
    data["record_id"] = make_record_id(evidence_url)
    data["date_found"] = datetime.now(timezone.utc).isoformat()
    data["last_checked"] = data["date_found"]

    return data


# ============================================================
# 6. GOOGLE SHEETS DUPLICATE DETECTION
# ============================================================

def existing_record_ids(worksheet):
    try:
        records = worksheet.get_all_records()
        return {
            str(row.get("Job ID", "")).strip()
            for row in records
            if row.get("Job ID")
        }
    except Exception:
        logging.exception("Could not read existing job IDs")
        return set()


def existing_scholarship_ids(worksheet):
    try:
        records = worksheet.get_all_records()
        return {
            str(row.get("Scholarship ID", "")).strip()
            for row in records
            if row.get("Scholarship ID")
        }
    except Exception:
        logging.exception("Could not read existing scholarship IDs")
        return set()


# ============================================================
# 7. SAVE ACADEMIC JOBS
# ============================================================

def save_job(worksheet, data, known_ids):
    if not data:
        return

    record_id = data["record_id"]

    if record_id in known_ids:
        logging.info("Duplicate job skipped: %s", data.get("job_title"))
        return

    row = [
        record_id,
        clean(data.get("job_title")),
        clean(data.get("organization")),
        clean(data.get("country")),
        clean(data.get("city")),
        clean(data.get("department")),
        clean(data.get("position_type")),
        clean(data.get("required_qualification")),
        clean(data.get("experience_required")),
        clean(data.get("age_limit")),
        clean(data.get("salary")),
        clean(data.get("deadline")),
        clean(data.get("application_url")),
        data["source_url"],
        clean(data.get("eligibility_status")),
        clean(data.get("age_relaxation")),
        clean(data.get("verification_status")),
        data["date_found"],
        data["last_checked"],
        clean(data.get("notes")),
    ]

    worksheet.append_row(row, value_input_option="USER_ENTERED")
    known_ids.add(record_id)

    logging.info("New academic job saved: %s", data.get("job_title"))


# ============================================================
# 8. SAVE FULLY FUNDED SCHOLARSHIPS
# ============================================================

def save_scholarship(worksheet, data, known_ids):
    if not data:
        return

    record_id = data["record_id"]

    if record_id in known_ids:
        logging.info(
            "Duplicate scholarship skipped: %s",
            data.get("scholarship_name")
        )
        return

    funding = clean(data.get("funding_type")).lower()

    # Conservative filter: do not save an opportunity unless the
    # model identifies its funding as fully funded.
    if "fully funded" not in funding:
        logging.info(
            "Scholarship skipped: full funding not established: %s",
            data.get("scholarship_name")
        )
        return

    row = [
        record_id,
        clean(data.get("scholarship_name")),
        clean(data.get("university")),
        clean(data.get("country")),
        clean(data.get("degree_level")),
        clean(data.get("research_area")),
        clean(data.get("funding_type")),
        clean(data.get("tuition_coverage")),
        clean(data.get("monthly_stipend")),
        clean(data.get("ielts_requirement")),
        clean(data.get("age_limit")),
        clean(data.get("eligibility_status")),
        clean(data.get("deadline")),
        clean(data.get("application_url")),
        data["source_url"],
        clean(data.get("verification_status")),
        data["date_found"],
        data["last_checked"],
        clean(data.get("notes")),
    ]

    worksheet.append_row(row, value_input_option="USER_ENTERED")
    known_ids.add(record_id)

    logging.info(
        "New scholarship saved: %s",
        data.get("scholarship_name")
    )


# ============================================================
# 9. MAIN AGENT
# ============================================================

def run_agent():
    logging.info("=" * 60)
    logging.info("AI ACADEMIC CAREER AGENT STARTED")
    logging.info("=" * 60)

    jobs_sheet, scholarships_sheet = connect_sheets()

    known_job_ids = existing_record_ids(jobs_sheet)
    known_scholarship_ids = existing_scholarship_ids(
        scholarships_sheet
    )

    # ------------------ ACADEMIC JOBS ------------------

    job_results = collect_search_results(JOB_QUERIES)

    for item in job_results:
        analysis = analyse_result(item, "job")

        if analysis:
            record = prepare_record(
                analysis,
                item["url"],
                "job"
            )
            save_job(jobs_sheet, record, known_job_ids)

    # ------------------ SCHOLARSHIPS ------------------

    scholarship_results = collect_search_results(
        SCHOLARSHIP_QUERIES
    )

    for item in scholarship_results:
        analysis = analyse_result(item, "scholarship")

        if analysis:
            record = prepare_record(
                analysis,
                item["url"],
                "scholarship"
            )
            save_scholarship(
                scholarships_sheet,
                record,
                known_scholarship_ids
            )

    logging.info("=" * 60)
    logging.info("AI ACADEMIC CAREER AGENT COMPLETED")
    logging.info("Jobs currently tracked: %s", len(known_job_ids))
    logging.info(
        "Scholarships currently tracked: %s",
        len(known_scholarship_ids)
    )
    logging.info("=" * 60)


if __name__ == "__main__":
    run_agent()
