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
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Four records per Gemini request: fewer API calls than one-by-one analysis.
BATCH_SIZE = 4
MAX_RESULTS_PER_QUERY = 6
MAX_CONTENT_CHARS = 3500

PROFILE = """
Candidate profile:
- Age: 38
- Nationality: Pakistani; Punjab domicile, Pakistan
- Qualification: MPhil Computer Science
- Experience: More than 14 years of teaching
- Current position: Contract Lecturer in Computer Science at
  COMSATS University Islamabad, Sahiwal Campus.
- Teaching: Programming Fundamentals (C/C++/Java), OOP, DSA,
  Mobile Application Development, DBMS, AI, Compiler Construction,
  Automata and Computation, Web Programming.
- Research interests: AI, Deep Learning, NLP, Generative AI,
  Healthcare AI, Explainable AI, Process Model Verification,
  Model Checking.
- Job interests: Computer Science Lecturer, Lecturer Computing,
  Lecturer Software Engineering, and Assistant Professor CS
  only where the stated requirements are met.
- Scholarship interest: Fully funded PhD opportunities only.
"""

JOB_QUERIES = [
    "Computer Science Lecturer faculty vacancies university 2026 apply deadline",
    "Assistant Professor Computer Science academic jobs international 2026",
    "Lecturer Computing Software Engineering university jobs worldwide 2026",
    "Computer Science lecturer jobs Pakistan university 2026 official",
]

SCHOLARSHIP_QUERIES = [
    "fully funded PhD Computer Science scholarships 2026 international students",
    "fully funded PhD Artificial Intelligence deep learning scholarship 2026",
    "funded PhD NLP generative AI healthcare AI international students 2026",
    "university PhD Computer Science scholarship tuition stipend 2026",
]

ELIGIBILITY_VALUES = {
    "eligible",
    "potentially eligible",
    "not eligible",
    "insufficient information",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

tavily = TavilyClient(api_key=TAVILY_API_KEY)
gemini = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# GOOGLE SHEETS
# ============================================================

def connect_sheets():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials, _ = google.auth.default(scopes=scopes)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    jobs = spreadsheet.worksheet("Academic Jobs")
    scholarships = spreadsheet.worksheet("Scholarships")
    logging.info("Connected to Google Sheets: %s", spreadsheet.title)
    return jobs, scholarships


def make_record_id(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def read_existing_ids(worksheet, column_name):
    # Fail safely if Sheets cannot be read; do not risk duplicate insertion.
    records = worksheet.get_all_records()
    return {
        str(row.get(column_name, "")).strip()
        for row in records
        if row.get(column_name)
    }


# ============================================================
# TAVILY SEARCH
# ============================================================

def search_web(query):
    try:
        logging.info("Searching: %s", query)
        result = tavily.search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=MAX_RESULTS_PER_QUERY,
            include_raw_content=False,
        )
        return result.get("results", [])
    except Exception as error:
        logging.error("Tavily search failed for '%s': %s", query, error)
        return []


def collect_search_results(queries, known_ids, label):
    collected = {}
    already_tracked = 0

    for query in queries:
        for result in search_web(query):
            url = (result.get("url") or "").strip()
            if not url:
                continue

            record_id = make_record_id(url)
            if record_id in known_ids:
                already_tracked += 1
                continue

            if url not in collected:
                collected[url] = {
                    "title": result.get("title", ""),
                    "url": url,
                    "content": result.get("content", "") or "",
                    "record_id": record_id,
                }

    results = list(collected.values())
    logging.info(
        "%s: %d new unique results; %d already tracked results skipped",
        label, len(results), already_tracked
    )
    return results


# ============================================================
# GEMINI BATCH ANALYSIS
# ============================================================

def extract_json(response_text):
    text = (response_text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def build_batch_prompt(items, category):
    if category == "job":
        fields = """
job_title, organization, country, city, department, position_type,
required_qualification, experience_required, age_limit, salary,
deadline, application_url, official_source_url, eligibility_status,
eligibility_reason, missing_requirements, recommended_action,
age_relaxation, position_assessments, verification_status, notes
"""
        category_instructions = """
Assess whether each result is a genuine Computer Science or related
academic teaching/faculty vacancy. If one advertisement contains
multiple ranks (for example Assistant Professor and Teaching Assistant
Professor), assess each rank separately in position_assessments.
Do not apply one rank's PhD requirement automatically to another rank.
"""
    else:
        fields = """
scholarship_name, university, country, degree_level, research_area,
funding_type, tuition_coverage, monthly_stipend, ielts_requirement,
age_limit, deadline, application_url, official_source_url,
eligibility_status, eligibility_reason, missing_requirements,
recommended_action, verification_status, notes
"""
        category_instructions = """
Only consider PhD opportunities. The user's preference is fully funded:
the evidence must support full tuition coverage AND a stipend. Do not
assume a scholarship is fully funded from its title alone. If either
tuition or stipend is not established by the supplied content, state
what is missing and do not describe it as confirmed fully funded.
"""

    compact_items = []
    for index, item in enumerate(items):
        compact_items.append({
            "index": index,
            "title": item["title"],
            "url": item["url"],
            "content": item["content"][:MAX_CONTENT_CHARS],
        })

    return f"""
You are a careful academic recruitment and PhD scholarship analyst.

CANDIDATE PROFILE:
{PROFILE}

CATEGORY: {category}

MANDATORY ELIGIBILITY CLASSIFICATION:
Use exactly one of these four values:
1. Eligible
2. Potentially Eligible
3. Not Eligible
4. Insufficient Information

Definitions:
- Eligible: all clearly stated mandatory requirements are met by the
  candidate's known profile.
- Potentially Eligible: the candidate appears suitable, but a specific
  matter such as degree equivalence, age relaxation, recognition of
  experience, or an unclear requirement must be confirmed.
- Not Eligible: the source clearly states a mandatory requirement that
  the candidate does not meet.
- Insufficient Information: the source does not provide enough evidence
  to decide. Do not guess.

Candidate-specific rules:
- The candidate is 38 years old. Do NOT reject solely because age is
  not stated. If an age limit is below 38, check whether the source
  states an applicable relaxation or exemption. If unclear, use
  Potentially Eligible only when the rest of the requirements appear
  met, and explain that age relaxation must be confirmed. If the
  advertisement clearly excludes the candidate and no applicable
  relaxation exists, use Not Eligible.
- The candidate holds an MPhil in Computer Science, not a PhD.
  If a particular position explicitly requires a completed PhD, that
  position is Not Eligible. Assess other ranks separately.
- More than 14 years of teaching experience should be considered where
  relevant, but do not assume it replaces a mandatory qualification.
- Do not invent age relaxation, equivalence, deadlines, funding,
  requirements, salary, or application links.
- Use "Not stated" when information is absent.
- Explain every classification in eligibility_reason.
- Put unresolved criteria in missing_requirements.
- Give a practical recommended_action, such as Apply, Contact HR,
  Verify age relaxation, Check the full advertisement, or Do not apply.
- {category_instructions}
- Verification status must be one of: Needs Review, Verified, Expired,
  Unknown. A search-result snippet alone is never enough for Verified.
- Use the supplied result URL as the evidence URL.
- Return JSON only, no Markdown, in this structure:
  {{"results":[{{"index":0,"data":{{...}}}}]}}
- Return one result for every input index, including irrelevant results.
- Each data object must contain these fields:
{fields}

INPUT SEARCH RESULTS:
{json.dumps(compact_items, ensure_ascii=False)}
"""


def analyse_batch(items, category):
    if not items:
        return {}

    logging.info(
        "Gemini batch started: category=%s, records=%d",
        category, len(items)
    )

    try:
        response = gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=build_batch_prompt(items, category),
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )
        payload = extract_json(response.text)
        output = {}

        for result in payload.get("results", []):
            try:
                index = int(result["index"])
                data = result["data"]
                if 0 <= index < len(items) and isinstance(data, dict):
                    output[index] = data
            except (KeyError, TypeError, ValueError):
                logging.warning("Ignoring malformed Gemini batch item")

        logging.info(
            "Gemini batch completed: %d of %d records returned",
            len(output), len(items)
        )
        return output

    except Exception as error:
        # A failed batch is logged; the next batch/category can still run.
        logging.error(
            "Gemini batch failed (%s, %d records): %s",
            category, len(items), error
        )
        return {}


def analyse_in_batches(items, category):
    all_results = {}
    total_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_no, start in enumerate(
        range(0, len(items), BATCH_SIZE), start=1
    ):
        batch = items[start:start + BATCH_SIZE]
        logging.info(
            "%s analysis batch %d/%d",
            category, batch_no, total_batches
        )
        result = analyse_batch(batch, category)
        for local_index, data in result.items():
            all_results[start + local_index] = data

    logging.info(
        "%s analysis complete: %d/%d results analysed",
        category, len(all_results), len(items)
    )
    return all_results


# ============================================================
# VALIDATION AND ELIGIBILITY MESSAGE
# ============================================================

OFFICIAL_DOMAIN_HINTS = (".edu", ".ac.", ".gov", ".edu.pk", ".gov.pk")


def source_looks_official(url):
    try:
        host = urlparse(url).netloc.lower()
        return any(hint in host for hint in OFFICIAL_DOMAIN_HINTS)
    except Exception:
        return False


def clean(value):
    if value is None:
        return "Not stated"
    if isinstance(value, list):
        value = "; ".join(str(item) for item in value)
    value = str(value).strip()
    return value if value else "Not stated"


def normalize_eligibility(value):
    value = clean(value).lower()
    mapping = {
        "eligible": "Eligible",
        "potentially eligible": "Potentially Eligible",
        "not eligible": "Not Eligible",
        "insufficient information": "Insufficient Information",
    }
    return mapping.get(value, "Insufficient Information")


def prepare_record(data, source_url):
    if not isinstance(data, dict):
        return None

    data["eligibility_status"] = normalize_eligibility(
        data.get("eligibility_status")
    )
    data["official_source_url"] = clean(
        data.get("official_source_url", source_url)
    )
    data["source_url"] = source_url

    # Search results are leads, not independently verified applications.
    data["verification_status"] = (
        "Needs Review" if not source_looks_official(source_url)
        else "Needs Review"
    )

    data["record_id"] = make_record_id(source_url)
    now = datetime.now(timezone.utc).isoformat()
    data["date_found"] = now
    data["last_checked"] = now
    return data


def eligibility_notes(data):
    parts = [
        "Eligibility reason: " + clean(data.get("eligibility_reason")),
        "Missing requirements / checks: " +
        clean(data.get("missing_requirements")),
        "Recommended action: " + clean(data.get("recommended_action")),
    ]

    positions = data.get("position_assessments")
    if positions:
        parts.append("Separate position assessments: " + clean(positions))

    extra = clean(data.get("notes"))
    if extra != "Not stated":
        parts.append("Additional notes: " + extra)

    return " | ".join(parts)


# ============================================================
# SAVE ACADEMIC JOBS - EXISTING 20-COLUMN LAYOUT
# ============================================================

def save_job(worksheet, data, known_ids):
    if not data:
        return False

    record_id = data["record_id"]
    if record_id in known_ids:
        logging.info("Duplicate job skipped: %s", data.get("job_title"))
        return False

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
        data["eligibility_status"],
        clean(data.get("age_relaxation")),
        clean(data.get("verification_status")),
        data["date_found"],
        data["last_checked"],
        eligibility_notes(data),
    ]

    worksheet.append_row(row, value_input_option="USER_ENTERED")
    known_ids.add(record_id)
    logging.info("New academic job saved: %s", data.get("job_title"))
    return True


# ============================================================
# SAVE FULLY FUNDED PHD SCHOLARSHIPS - EXISTING 19-COLUMN LAYOUT
# ============================================================

def tuition_is_fully_covered(value):
    text = clean(value).lower()
    if text in ("not stated", "unknown", "partial", "partially funded"):
        return False
    indicators = (
        "full tuition", "tuition fully", "100% tuition",
        "tuition waiver", "full fee waiver", "all tuition fees",
        "entire tuition", "tuition fees covered",
    )
    return any(term in text for term in indicators)


def stipend_is_confirmed(value):
    text = clean(value).lower()
    if text in ("not stated", "unknown", "none", "no stipend", ""):
        return False
    # A concrete amount or explicit stipend wording is required.
    return bool(
        re.search(r"\d", text)
        or any(term in text for term in (
            "monthly stipend", "annual stipend", "living allowance",
            "maintenance allowance", "stipend provided",
        ))
    )


def funding_is_confirmed(data):
    return (
        tuition_is_fully_covered(data.get("tuition_coverage"))
        and stipend_is_confirmed(data.get("monthly_stipend"))
    )


def save_scholarship(worksheet, data, known_ids):
    if not data:
        return False

    record_id = data["record_id"]
    if record_id in known_ids:
        logging.info(
            "Duplicate scholarship skipped: %s",
            data.get("scholarship_name")
        )
        return False

    degree = clean(data.get("degree_level")).lower()
    if "phd" not in degree and "doctor" not in degree:
        logging.info(
            "Scholarship skipped (not clearly PhD): %s",
            data.get("scholarship_name")
        )
        return False

    if not funding_is_confirmed(data):
        logging.info(
            "Scholarship held back: full tuition and stipend not both "
            "confirmed by supplied evidence: %s",
            data.get("scholarship_name")
        )
        return False

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
        data["eligibility_status"],
        clean(data.get("deadline")),
        clean(data.get("application_url")),
        data["source_url"],
        clean(data.get("verification_status")),
        data["date_found"],
        data["last_checked"],
        eligibility_notes(data),
    ]

    worksheet.append_row(row, value_input_option="USER_ENTERED")
    known_ids.add(record_id)
    logging.info(
        "New fully funded PhD scholarship saved: %s",
        data.get("scholarship_name")
    )
    return True


# ============================================================
# PROCESS CATEGORY
# ============================================================

def process_category(results, category, worksheet, known_ids):
    analyses = analyse_in_batches(results, category)
    saved = 0

    for index, item in enumerate(results):
        data = analyses.get(index)
        if not data:
            logging.warning(
                "No analysis returned for %s: %s",
                category, item.get("title")
            )
            continue

        record = prepare_record(data, item["url"])
        if category == "job":
            saved += int(save_job(worksheet, record, known_ids))
        else:
            saved += int(save_scholarship(worksheet, record, known_ids))

    logging.info(
        "%s summary: discovered=%d, analysed=%d, newly saved=%d",
        category, len(results), len(analyses), saved
    )
    return saved


# ============================================================
# MAIN
# ============================================================

def run_agent():
    logging.info("=" * 60)
    logging.info("AI ACADEMIC CAREER AGENT STARTED")
    logging.info(
        "Model=%s | batch_size=%d | search_depth=basic",
        GEMINI_MODEL, BATCH_SIZE
    )
    logging.info("=" * 60)

    jobs_sheet, scholarships_sheet = connect_sheets()

    known_job_ids = read_existing_ids(jobs_sheet, "Job ID")
    known_scholarship_ids = read_existing_ids(
        scholarships_sheet, "Scholarship ID"
    )

    # Discover both categories first. Existing URLs are skipped before
    # any Gemini call, saving time and API usage.
    job_results = collect_search_results(
        JOB_QUERIES, known_job_ids, "Academic jobs"
    )
    scholarship_results = collect_search_results(
        SCHOLARSHIP_QUERIES, known_scholarship_ids, "Scholarships"
    )

    new_jobs = process_category(
        job_results, "job", jobs_sheet, known_job_ids
    )
    new_scholarships = process_category(
        scholarship_results, "scholarship",
        scholarships_sheet, known_scholarship_ids
    )

    logging.info("=" * 60)
    logging.info("AI ACADEMIC CAREER AGENT COMPLETED")
    logging.info("New academic jobs saved this run: %d", new_jobs)
    logging.info("New scholarships saved this run: %d", new_scholarships)
    logging.info("Total academic jobs tracked: %d", len(known_job_ids))
    logging.info("Total scholarships tracked: %d", len(known_scholarship_ids))
    logging.info("=" * 60)


if __name__ == "__main__":
    run_agent()
