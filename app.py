import os
import re
import logging

import numpy as np
from flask import Flask, jsonify, request
from openai import OpenAI

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Credentials ───────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL    = 'text-embedding-3-small'
PASS_THRESHOLD = 60

ICP_ANCHOR = """
A mid-market consumer packaged goods company with 100 to 5000 employees selling
everyday-use products in household care, personal care, food, or beverage categories.
The brand sells direct to consumers through retail, e-commerce, or grocery channels
and cares deeply about brand perception, consumer sentiment, and social listening.
Alternatively, a PR, communications, or advertising agency with 100 to 5000 employees
that explicitly serves CPG brand clients and manages consumer-facing campaigns,
brand reputation, or media strategy on their behalf.
""".strip()

# Thresholds calibrated for text-embedding-3-small cosine similarity range.
# Tune cutoffs here if scoring feels too strict or too generous after live testing.
ICP_BANDS = [
    (0.58, 60),   # core ICP — Edelman-tier and above
    (0.50, 45),   # strong fit — Grove, Weber Shandwick
    (0.40, 30),   # probable fit — Olipop, Liquid Death, Dentsu
    (0.33, 15),   # marginal — Palantir, Caterpillar
    (0.00,  0),   # no fit — Stripe-tier
]

GENERIC_PREFIXES = re.compile(
    r'^(info|press|hello|contact|team|support|sales|marketing|media|pr|admin|general)@',
    re.IGNORECASE
)

# Ordered tiers — first match wins, so more senior patterns must come first
SENIORITY_TIERS = [
    (re.compile(r'\b(cmo|cbo|chief\s+marketing|chief\s+brand)\b', re.IGNORECASE), 15, 'C-suite'),
    (re.compile(r'\bvp\b|\bvice\s+president\b',                   re.IGNORECASE), 12, 'VP-level'),
    (re.compile(r'\bdirector\b',                                   re.IGNORECASE),  8, 'Director-level'),
    (re.compile(r'\bsenior\s+manager\b|\bsr\.?\s+manager\b',      re.IGNORECASE),  4, 'Senior Manager'),
]

# ── Embedding Utilities ───────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    """Embed text and return a unit-normalized vector."""
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)    # normalize so dot product == cosine sim


# Pre-compute the anchor embedding once at startup — reused for every lead scored.
# This saves one API call per lead and keeps latency low on batch runs.
logger.info('Computing ICP anchor embedding...')
ANCHOR_VEC = embed(ICP_ANCHOR)
logger.info('Anchor ready.')

# ── Criterion 1: ICP Fit (60 pts) ─────────────────────────────────────────────

def score_icp_fit(company_name, industry, employee_count, description):
    profile = (
        f"Company: {company_name}. "
        f"Industry: {industry}. "
        f"Employees: {employee_count}. "
        f"Description: {description}"
    )
    lead_vec   = embed(profile)
    similarity = float(np.dot(lead_vec, ANCHOR_VEC))

    # Walk bands top-to-bottom, first threshold met wins
    points = 0
    for threshold, pts in ICP_BANDS:
        if similarity >= threshold:
            points = pts
            break

    return points, f"ICP fit: {points}/60 (cosine similarity: {similarity:.4f})"

# ── Criterion 2: Verified Email (25 pts) ──────────────────────────────────────

def score_email(contact_email):
    """
    25 pts  personal address (not a generic inbox)
    10 pts  generic inbox (info@, press@, etc.)
     0 pts  missing, malformed, or no @
    """
    email = contact_email.strip().lower()
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        return 0, 'Email: 0/25 (missing or malformed)'
    if GENERIC_PREFIXES.match(email):
        return 10, f'Email: 10/25 (generic inbox: {email})'
    return 25, f'Email: 25/25 (personal address: {email})'

# ── Criterion 3: Contact Seniority (15 pts) ───────────────────────────────────

def score_seniority(contact_title):
    """
    Match contact_title against seniority tiers in descending order.
    Titles that match nothing score 0.
    """
    for pattern, pts, label in SENIORITY_TIERS:
        if pattern.search(contact_title):
            return pts, f'Seniority: {pts}/15 ({label}: "{contact_title}")'
    return 0, f'Seniority: 0/15 (below threshold: "{contact_title}")'

# ── Qualification Logic ───────────────────────────────────────────────────────

def qualify_lead(body):
    icp_pts,    icp_note    = score_icp_fit(
        body['company_name'],
        body['industry'],
        body['employee_count'],
        body['description'],
    )
    email_pts,  email_note  = score_email(body['contact_email'])
    senior_pts, senior_note = score_seniority(body['contact_title'])

    total  = icp_pts + email_pts + senior_pts
    passed = total >= PASS_THRESHOLD
    notes  = (
        f"{icp_note} | "
        f"{email_note} | "
        f"{senior_note} | "
        f"Total: {total}/100 — {'PASS' if passed else 'FAIL'}"
    )
    return total, notes, passed

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/v1/qualify', methods=['POST'])
def qualify():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'Request body must be JSON'}), 400

    required = ['company_name', 'industry', 'employee_count',
                'description', 'contact_title', 'contact_email']
    missing = [f for f in required if not body.get(f) and body.get(f) != 0]
    if missing:
        return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

    if not isinstance(body['employee_count'], int):
        return jsonify({'error': 'employee_count must be an integer'}), 400

    try:
        total, notes, passed = qualify_lead(body)

        logger.info(f"[{'PASS' if passed else 'FAIL'}] {body['company_name']} — {total}/100")

        return jsonify({
            'company_name':    body['company_name'],
            'qualifier_score': total,
            'qualifier_notes': notes,
            'passed':          passed,
        }), 200

    except Exception as e:
        logger.exception('Unhandled error in /v1/qualify')
        return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
