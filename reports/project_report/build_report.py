#!/usr/bin/env python3
"""Build the CivicLens AI TE Lab project report as a PDF.

The layout reproduces the department's TE Lab report template: a crimson cover
page with the institute logo, a running header and footer, an Abstract page, a
Table of Contents and fifteen numbered chapters that each start on a new page,
with crimson data tables and a closing "END OF REPORT" line.

Every figure quoted in the report comes from this repository: the dataset in
data/raw/complaints_dataset.csv, the committed models in models/ and
reports/evaluation_report.md. The values were re-verified independently by
re-deriving the train/test split, retraining both selected models and running
the test suite. If the models or dataset change, update the numbers in the
content section below and re-run this script.

Usage:
    pip install reportlab
    python reports/project_report/build_report.py [--out PATH]

The Table of Contents page numbers are computed automatically (two-pass build).
The cover-page details (team, batch, guide) live in the COVER dictionary.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DEFAULT_OUT = HERE / "CivicLens_AI_Project_Report.pdf"

# --------------------------------------------------------------------------
# Cover-page details
# --------------------------------------------------------------------------
COVER = {
    "institute": "Ramrao Adik Institute of Technology",
    "department": "Department of Computer Science and Engineering",
    "report_type": "TE LAB PROJECT REPORT",
    "title_lines": [
        "CivicLens AI:",
        "NLP System for Citizen Complaint",
        "Triage, Prioritization and Routing",
    ],
    "students": [
        "Sanchit Patil (24AD1091)",
        "Shinoy Negi (24AD1126)",
        "Shravan Shewade (24AD1128)",
    ],
    "batch": "Batch: B2",
    "guide": "Dr. Ekta Sarda",
}
SHORT_TITLE = "CivicLens AI"
REPORT_LABEL = "TE LAB Project Report"
FOOTER_LEFT = "Ramrao Adik Institute of Technology"
PDF_TITLE = "CivicLens AI - TE Lab Project Report"

# --------------------------------------------------------------------------
# Design tokens (measured from the template)
# --------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
CRIMSON = colors.Color(0.65098, 0.098039, 0.207843)  # #A61935
INK = colors.HexColor("#222222")
MUTED = colors.HexColor("#555555")
RULE = colors.Color(0.6, 0.6, 0.6)
BORDER = colors.Color(0.69, 0.69, 0.69)
ZEBRA = colors.Color(0.96078, 0.96078, 0.96078)

MARGIN_X = 1.8 * cm
MARGIN_Y = 2.2 * cm
INDENT = 0.2 * cm
FONT, BOLD = "Helvetica", "Helvetica-Bold"


def _style(name: str, **kw) -> ParagraphStyle:
    base = dict(fontName=FONT, fontSize=10.2, leading=16, textColor=INK, alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)


BODY = _style("Body", alignment=TA_JUSTIFY, leftIndent=INDENT, rightIndent=INDENT, spaceAfter=8)
CHAP = _style("Chap", fontName=BOLD, fontSize=9, leading=11, textColor=MUTED, leftIndent=INDENT, spaceAfter=3)
H1 = _style("H1", fontName=BOLD, fontSize=20, leading=25, textColor=CRIMSON, leftIndent=INDENT, spaceAfter=8)
H2 = _style("H2", fontName=BOLD, fontSize=13, leading=16, leftIndent=INDENT, spaceBefore=10, spaceAfter=7, keepWithNext=1)
BUL = _style("Bullet", fontSize=10, leading=15, leftIndent=INDENT + 17, firstLineIndent=-8, spaceAfter=6)
NOTE = _style("Note", fontSize=8.7, leading=12.5, textColor=MUTED, leftIndent=INDENT, rightIndent=INDENT)
CELL = _style("Cell", fontSize=9, leading=11.5)
CELL_H = _style("CellH", fontName=BOLD, fontSize=9, leading=11.5, textColor=colors.white)
END = _style("End", fontSize=12, leading=15, textColor=CRIMSON, alignment=TA_CENTER)

_AMP = re.compile(r"&(?!(?:amp|lt|gt|quot|#\d+);)")


def mk(text: str) -> str:
    """Escape bare ampersands so plain text can be used inside Paragraph markup."""
    return _AMP.sub("&amp;", text)


# --------------------------------------------------------------------------
# Small flowable helpers
# --------------------------------------------------------------------------
def body(text: str) -> Paragraph:
    return Paragraph(mk(text), BODY)


def h2(text: str) -> Paragraph:
    return Paragraph(mk(text), H2)


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph("• " + mk(t), BUL) for t in items]


def note(label: str, text: str) -> Paragraph:
    return Paragraph(f"<b>{label}</b> {mk(text)}", NOTE)


def table(rows: list[list[str]], widths: list[float]) -> Table:
    data = [[Paragraph(mk(c), CELL_H) for c in rows[0]]]
    data += [[Paragraph(mk(c), CELL) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), CRIMSON),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
                ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return t


class PageMark(Flowable):
    """Zero-size marker that records the page it lands on (used for the TOC)."""

    def __init__(self, key: str, registry: dict[str, int]):
        super().__init__()
        self.key, self.registry = key, registry
        self.width = self.height = 0

    def wrap(self, aw, ah):
        return 0, 0

    def draw(self):
        self.registry[self.key] = self.canv.getPageNumber()


class TocLine(Flowable):
    """One table-of-contents entry: label, dot leader and right-aligned page number."""

    HEIGHT = 24
    RIGHT = 268.0  # right edge of the page-number column, measured from the text indent

    def __init__(self, label: str, page: int | str):
        super().__init__()
        self.label, self.page = label, str(page)

    def wrap(self, aw, ah):
        return aw, self.HEIGHT

    def draw(self):
        c, size = self.canv, 10.5
        x0, base = INDENT, self.HEIGHT - size
        right = x0 + self.RIGHT
        c.setFillColor(INK)
        c.setFont(FONT, size)
        c.drawString(x0, base, self.label)
        c.drawRightString(right, base, self.page)
        used = stringWidth(self.label + " ", FONT, size)
        tail = stringWidth(" " + self.page, FONT, size)
        dots = int((self.RIGHT - used - tail) / stringWidth(".", FONT, size))
        c.drawString(x0 + used, base, "." * max(dots, 3))


# --------------------------------------------------------------------------
# Page decorations
# --------------------------------------------------------------------------
def draw_cover(canv, doc):
    canv.saveState()
    canv.setFillColor(CRIMSON)
    canv.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    canv.drawImage(str(ASSETS / "logo_cover.png"), 181.4173, PAGE_H - 197.3386, width=232.4409, height=83.6220)
    canv.setFillColor(colors.white)

    def line(y_top: float, text: str, bold: bool = False, size: float = 11.5):
        canv.setFont(BOLD if bold else FONT, size)
        canv.drawCentredString(PAGE_W / 2, PAGE_H - y_top, text)

    line(248.5, COVER["institute"])
    line(265.5, COVER["department"])
    line(302.4, COVER["report_type"])
    for i, text in enumerate(COVER["title_lines"]):
        assert stringWidth(text, BOLD, 24) < PAGE_W - 2 * 60, f"cover title line too wide: {text}"
        line(357.4 + 31 * i, text, bold=True, size=24)
    line(488.7, "Submitted by", bold=True)
    y = 519.9
    for name in COVER["students"]:
        line(y, name)
        y += 17
    y += 19.8  # gap between the last name and the batch line, as in the template
    line(y, COVER["batch"])
    y += 62.4
    line(y, "Guided by", bold=True)
    line(y + 17, COVER["guide"])
    canv.restoreState()


def draw_body_page(canv, doc):
    canv.saveState()
    left, right = MARGIN_X, PAGE_W - MARGIN_X
    canv.setLineWidth(0.55)
    canv.setStrokeColor(RULE)
    # header
    canv.setFont(BOLD, 8)
    canv.setFillColor(CRIMSON)
    canv.drawString(left, PAGE_H - 36.9, SHORT_TITLE)
    canv.setFont(FONT, 8)
    canv.setFillColor(MUTED)
    canv.drawRightString(right, PAGE_H - 36.9, REPORT_LABEL)
    canv.line(left, PAGE_H - 48.2, right, PAGE_H - 48.2)
    # footer: rule stops at the logo, page number sits to its right
    logo_w, logo_h = 97.4, 39.7
    logo_x = 531.5 - logo_w
    canv.line(left, PAGE_H - 802.2, logo_x, PAGE_H - 802.2)
    canv.drawString(left, PAGE_H - 819.2, FOOTER_LEFT)
    canv.drawImage(str(ASSETS / "logo_footer.png"), logo_x, PAGE_H - 831.97, width=logo_w, height=logo_h)
    canv.drawRightString(right, PAGE_H - 819.2, str(doc.page))
    canv.restoreState()


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------
CHAPTERS = [
    "Introduction",
    "Literature Survey",
    "Limitations of Existing Systems",
    "Problem Statement and Objectives",
    "Dataset and Data Preparation",
    "Project Flow and System Architecture",
    "Technologies and Methodologies",
    "System Design and Implementation",
    "Application Features and User Workflow",
    "Result and Analysis",
    "Testing and Validation",
    "Security and Privacy Considerations",
    "Limitations and Future Scope",
    "Conclusion",
    "References",
]


def chapter(no: int, marks: dict[str, int]) -> list:
    title = CHAPTERS[no - 1].upper()
    return [
        PageBreak(),
        PageMark(f"ch{no}", marks),
        Paragraph(f"CHAPTER {no}", CHAP),
        Paragraph(mk(title), H1),
    ]


def story(toc_pages: dict[str, int], marks: dict[str, int]) -> list:
    s: list = [Spacer(1, 1), NextPageTemplate("Body")]

    # ---- Abstract ------------------------------------------------------
    s += [PageBreak(), PageMark("abstract", marks), Paragraph("ABSTRACT", H1)]
    s += [
        body(
            "CivicLens AI is an NLP-based complaint triage system developed to demonstrate how machine learning "
            "and transparent rule-based analysis can automate the handling of citizen complaints in a municipal "
            "grievance system. From the raw complaint text alone, the system classifies the complaint into one of "
            "ten civic categories, estimates its urgency as Low, Medium, High or Critical, checks whether similar "
            "complaints have already been reported, and routes it to the responsible department with a "
            "service-level deadline."
        ),
        body(
            "The project is implemented as a Python package with a FastAPI REST service, SQLite persistence and a "
            "React frontend that contains a citizen complaint form and a staff triage dashboard. The category "
            "classifier uses TF-IDF features with a calibrated Linear SVM, and the urgency classifier combines "
            "TF-IDF with 11 engineered lexicon and text-style features and Logistic Regression. Both were trained on a synthetic "
            "dataset of 3,000 complaints with deliberately injected label noise and typing errors. On a held-out "
            "test set of 600 complaints, the category classifier reports 98.33% accuracy and the urgency "
            "classifier reports 94.50% accuracy with an ordinal mean absolute error of 0.055."
        ),
        body(
            "The project demonstrates how a lightweight and explainable pipeline, in which every priority score "
            "can be traced to the keywords, model confidences and repeat reports that produced it, can be built "
            "without depending on an external AI API for its core NLP features."
        ),
        table(
            [
                ["Project aspect", "Summary"],
                ["Domain", "Natural Language Processing / Civic Technology"],
                ["Application", "FastAPI REST service with a React staff dashboard"],
                ["Dataset", "Synthetic complaint dataset (3,000 complaints, 10 categories)"],
                ["Main ML models", "TF-IDF + Linear SVM (category), TF-IDF + Logistic Regression (urgency)"],
                ["Supporting NLP", "Lexicon-based urgency scoring and TF-IDF duplicate detection"],
                ["Routing logic", "Department routing, SLA deadlines and critical escalation"],
            ],
            [155.9, 334.5],
        ),
    ]

    # ---- Table of contents --------------------------------------------
    s += [PageBreak(), PageMark("toc", marks), Paragraph("TABLE OF CONTENTS", H1)]
    for i, name in enumerate(CHAPTERS, 1):
        s.append(TocLine(f"{i}. {name}", toc_pages.get(f"ch{i}", 0)))

    # ---- 1 Introduction -------------------------------------------------
    s += chapter(1, marks)
    s += [
        body(
            "Municipal grievance systems receive a continuous stream of citizen complaints about potholes, water "
            "outages, garbage collection, street lighting and public safety. Many of them rely on citizens "
            "choosing a category from a drop-down list and on a human clerk deciding which complaints are urgent. "
            "In practice, citizens often pick the wrong category, urgent problems wait in the same queue as trivial "
            "ones, and the same problem is reported many times as separate, unrelated tickets."
        ),
        body(
            "CivicLens AI was developed as an NLP-based triage system that addresses these problems from the raw "
            "complaint text alone. The system tokenizes and stems the text, predicts the civic category using "
            "TF-IDF features and a Linear SVM, and estimates urgency with a second classifier that also uses "
            "features from a transparent keyword lexicon. A duplicate detector compares the complaint with earlier "
            "reports, and a routing engine combines all signals into a priority score, a responsible department "
            "and a service-level deadline."
        ),
        body(
            "The project goes beyond plain text classification by making every decision explainable. Each routed "
            "ticket carries an explanation that lists the keywords, model confidences and repeat reports behind "
            "its priority score, so that a reviewer can audit the outcome instead of trusting a black-box model."
        ),
        h2("1.1 Project scope"),
        *bullets(
            [
                "Automatic classification of complaints into ten civic categories.",
                "Four-level urgency classification: Low, Medium, High and Critical.",
                "Transparent lexicon-based scoring of urgency and frustration.",
                "Repeat-complaint detection using TF-IDF cosine similarity.",
                "Department routing, SLA deadlines and escalation of critical complaints.",
                "FastAPI REST service with SQLite persistence.",
                "React interface with a citizen complaint form and a staff triage dashboard.",
            ]
        ),
        h2("1.2 Significance"),
        body(
            "The project provides an academic implementation in which the major components can be inspected and "
            "tested independently. It demonstrates text preprocessing, feature extraction, supervised learning, "
            "rule-based scoring, model evaluation and full-stack integration in a single civic-technology "
            "application."
        ),
    ]

    # ---- 2 Literature survey -------------------------------------------
    s += chapter(2, marks)
    s += [
        body(
            "The project draws from established work in text categorization, term weighting, sentiment lexicons "
            "and information retrieval. These areas provide the theoretical basis for the modules implemented in "
            "CivicLens AI."
        ),
        table(
            [
                ["Work", "Contribution relevant to the project"],
                ["Salton & Buckley (1988)", "TF-IDF term weighting used to represent text as numerical features."],
                ["Porter (1980)", "Suffix-stripping algorithm used to reduce complaint words to their stems."],
                ["McCallum & Nigam (1998)", "Comparison of event models for Naive Bayes text classification."],
                ["Joachims (1998)", "Support Vector Machines for high-dimensional text categorization."],
                ["Sebastiani (2002)", "Survey of machine-learning methods for automated text categorization."],
                ["Hutto & Gilbert (2014)", "VADER: a transparent, lexicon-based approach to sentiment analysis."],
                ["Manning et al. (2008)", "Vector space model and cosine similarity for matching documents."],
                ["Gutiérrez et al. (2016)", "Ordinal regression methods and evaluation measures for ordered classes."],
            ],
            [136.1, 354.3],
        ),
        h2("2.1 Relation to CivicLens AI"),
        body(
            "The implementation uses TF-IDF as the common text representation for category classification, "
            "urgency classification and duplicate detection. Multinomial Naive Bayes, Logistic Regression and a "
            "Linear SVM are compared for category classification, following the text-categorization literature. "
            "Urgency and frustration scoring follows the lexicon-based approach of VADER, using a smaller "
            "domain-specific word list whose every match can be inspected. Because urgency levels are ordered, "
            "the evaluation also reports an ordinal mean absolute error in addition to accuracy."
        ),
    ]

    # ---- 3 Limitations of existing systems -----------------------------
    s += chapter(3, marks)
    s += [
        body(
            "Several practical limitations motivate the combined design of CivicLens AI. These limitations are "
            "considered from the perspective of the functionality targeted by the project."
        ),
        *bullets(
            [
                "Manual category selection from a drop-down list often sends complaints to the wrong department.",
                "Urgency is judged by a human clerk, so dangerous complaints may wait in the same queue as trivial "
                "requests.",
                "The same problem reported by several citizens becomes several unrelated tickets, and repeat "
                "reports are not used as evidence of importance.",
                "Black-box classifiers do not explain why a complaint received a given priority, which is "
                "undesirable for a civic system that must be auditable.",
                "Keyword-only rules are brittle against typing mistakes, merged words and varied phrasing.",
                "Cloud-based NLP services need internet connectivity, API credentials and external processing of "
                "citizen data.",
            ]
        ),
        body(
            "CivicLens AI responds by combining supervised classification, a transparent lexicon, duplicate "
            "detection and deterministic routing in a single local and explainable pipeline."
        ),
    ]

    # ---- 4 Problem statement and objectives ----------------------------
    s += chapter(4, marks)
    s += [
        body(
            "<b>Problem Statement:</b> To build an NLP-based system that reads a free-text citizen complaint, "
            "identifies its civic category and urgency, detects repeat reports, and routes it to the responsible "
            "department with a service-level deadline while keeping every decision explainable."
        ),
        h2("4.1 Objectives"),
        *bullets(
            [
                "Generate a reproducible, labelled complaint dataset with realistic label noise and typing errors.",
                "Preprocess complaint text using tokenization, stop-word removal and stemming.",
                "Compare Multinomial Naive Bayes, Logistic Regression and Linear SVM for category classification.",
                "Compare Logistic Regression, Linear SVM and Random Forest for urgency classification.",
                "Combine TF-IDF features with engineered lexicon features for urgency prediction.",
                "Detect repeat complaints using TF-IDF cosine similarity.",
                "Blend classifier, lexicon and repeat-report signals into a 0-100 priority score.",
                "Route tickets to departments with SLA deadlines and escalate Critical complaints.",
                "Provide a REST API and a web interface for citizens and civic staff.",
            ]
        ),
        h2("4.2 Expected outcome"),
        body(
            "The expected outcome is a working demonstration of an NLP-based complaint triage system in which "
            "classification, prioritization, duplicate detection and routing are available through one API and "
            "one web application."
        ),
    ]

    # ---- 5 Dataset and data preparation --------------------------------
    s += chapter(5, marks)
    s += [
        body(
            "The project uses a synthetic dataset generated by <b>civiclens/data/generate_dataset.py</b> and "
            "stored as <b>data/raw/complaints_dataset.csv</b>. Each record contains a complaint ID, the complaint "
            "text, its category and urgency labels, the ward, the street and a submission date. No real municipal "
            "dataset was available for redistribution, so complaints are composed from hand-written templates that "
            "are tiered by urgency, combined with wards, streets, durations, salutations and sign-offs using a "
            "fixed random seed."
        ),
        table(
            [
                ["Dataset property", "Verified value"],
                ["Total complaints", "3,000 (all texts unique)"],
                ["Categories", "10 (289 to 308 complaints per category)"],
                ["Urgency: Low / Medium / High / Critical", "813 / 1,016 / 790 / 381 (27.1% / 33.9% / 26.3% / 12.7%)"],
                ["Average complaint length", "33.8 words (minimum 9, maximum 74)"],
                ["Training / test split", "2,400 / 600 (80% / 20%), stratified by category"],
                ["Random state", "42"],
                ["Injected label noise", "Urgency shifted by one level in 8%; category swapped within related pairs in 3%"],
                ["Injected text noise", "One or two dropped characters or merged words in about half of the complaints"],
                ["TF-IDF vocabulary", "3,363 unigram and bigram terms (category model)"],
            ],
            [198.5, 291.9],
        ),
        h2("5.1 Text preprocessing"),
        body(
            "Each complaint is lowercased and split into alphabetic tokens with a regular expression, which "
            "discards digits and punctuation. English stop-words from scikit-learn's bundled list and "
            "single-letter tokens are removed, and the remaining tokens are reduced to stems with the Porter "
            "stemmer. No NLTK corpora need to be downloaded, so the pipeline runs fully offline. The tokens are "
            "then converted to TF-IDF features using unigrams and bigrams, a minimum document frequency of two "
            "and sublinear term frequency."
        ),
        h2("5.2 Dataset observation"),
        body(
            "Every complaint text is unique, but the texts are assembled from a finite set of templates. As a "
            "result, 93.2% of the test complaints have a training complaint with a TF-IDF cosine similarity of "
            "at least 0.55, and 49.5% have one of at least 0.70. Label noise and typing errors were injected on "
            "purpose so that the metrics do not reach a trivial 100%."
        ),
        note(
            "Important:",
            "The dataset is synthetic. The reported metrics show how the pipeline behaves on template-based "
            "complaints and should not be interpreted as accuracy on real citizen-submitted text.",
        ),
    ]

    # ---- 6 Project flow and system architecture ------------------------
    s += chapter(6, marks)
    s += [
        table(
            [
                ["Stage", "Operation", "Output"],
                ["1", "Citizen submits the complaint text and an optional location", "Raw complaint"],
                ["2", "Tokenization, stop-word removal and Porter stemming", "Stemmed tokens"],
                ["3", "TF-IDF vectorization (unigrams and bigrams)", "Sparse feature vector"],
                ["4", "Category classification (calibrated Linear SVM)", "Category + confidence"],
                ["5", "Urgency classification (TF-IDF + 11 engineered features)", "Urgency class + confidence"],
                ["6", "Lexicon analysis of urgency and frustration terms", "Lexicon score + matched terms"],
                ["7", "Duplicate search inside the predicted category", "Similar complaint IDs"],
                ["8", "Routing: priority blend, department, SLA and escalation", "Routed ticket"],
                ["9", "Persistence in SQLite and delivery through the API", "Stored ticket / dashboard view"],
            ],
            [42.5, 292.0, 155.9],
        ),
        h2("6.1 Architectural layers"),
        body(
            "<b>Presentation layer:</b> React single-page application with a citizen complaint form and a staff "
            "triage dashboard."
        ),
        body(
            "<b>Application layer:</b> FastAPI routes, Pydantic request validation, CORS configuration and a lock "
            "around the shared analyzer."
        ),
        body(
            "<b>ML/NLP layer:</b> Preprocessing, TF-IDF classifiers, lexicon scoring, duplicate index and routing "
            "engine, orchestrated by the ComplaintAnalyzer class."
        ),
        body(
            "<b>Data layer:</b> SQLite database accessed through SQLAlchemy, storing every routed ticket together "
            "with its explanation."
        ),
        h2("6.2 Processing principle"),
        body(
            "The design combines three independent signals instead of trusting the classifier alone: the "
            "predicted urgency class, the transparent lexicon score and the number of similar complaints already "
            "reported. The priority score is 0.7 × classifier score + 0.3 × lexicon score, where the "
            "classifier score is 25, 50, 75 or 100 for Low, Medium, High and Critical and the lexicon score is "
            "scaled to 0-100. If two or more similar complaints exist, 12 points are added (capped at 100). Scores "
            "of at least 37.5, 62.5 and 87.5 map to Medium, High and Critical, with SLA deadlines of 72, 24 and 4 "
            "hours (168 hours for Low). Critical tickets are also escalated to the Disaster and Emergency "
            "Response Cell."
        ),
    ]

    # ---- 7 Technologies and methodologies ------------------------------
    s += chapter(7, marks)
    s += [
        table(
            [
                ["Component", "Technology / methodology"],
                ["Programming language", "Python 3 (backend and machine learning), JavaScript (frontend)"],
                ["Machine learning", "scikit-learn [10]"],
                ["Feature extraction", "TF-IDF with unigrams, bigrams and sublinear term frequency"],
                ["Category classification", "Multinomial Naive Bayes, Logistic Regression, Linear SVM"],
                ["Urgency classification", "Logistic Regression, Linear SVM, Random Forest [9]"],
                ["NLP preprocessing", "Regex tokenizer, scikit-learn stop-words, NLTK Porter stemmer"],
                ["Rule-based analysis", "Hand-authored urgency, dampener, frustration and calm lexicons"],
                ["Duplicate detection", "TF-IDF cosine similarity within the predicted category"],
                ["Web framework", "FastAPI, Uvicorn and Pydantic"],
                ["Database", "SQLite with SQLAlchemy 2.0"],
                ["Frontend", "React 19 and Vite"],
            ],
            [155.9, 334.5],
        ),
        h2("7.1 Category classification methodology"),
        body(
            "Complaint text is converted to TF-IDF features. Multinomial Naive Bayes, Logistic Regression and a "
            "Linear SVM are compared with 5-fold stratified cross-validation on macro-F1, and the best model is "
            "kept. The Linear SVM won; as it has no native probability output, it is wrapped in a calibrated "
            "classifier (sigmoid calibration [11], 3 folds) so that every prediction carries a confidence score."
        ),
        h2("7.2 Urgency classification methodology"),
        body(
            "The urgency classifier joins TF-IDF features with 11 engineered features: nine from the lexicon "
            "analysis (urgency and sentiment scores, counts of urgent, dampener and frustration terms, repeat "
            "flag, duration mentions, exclamation and capitalized-word counts) and two text-length measures. "
            "They are standardized and combined with the TF-IDF matrix in a FeatureUnion. Logistic Regression, "
            "Linear SVM and a 300-tree Random Forest were compared by cross-validation, and Logistic Regression "
            "was selected."
        ),
        h2("7.3 Lexicon and duplicate-detection methodology"),
        body(
            "The lexicon holds 37 critical terms, 22 elevated terms, 16 low-urgency dampeners, 20 frustration "
            "terms and 8 calm terms with small hand-set weights. Matches are summed into a raw urgency score and "
            "squashed to the range 0-1 with a logistic function. The duplicate detector keeps a TF-IDF matrix of "
            "all earlier complaints (3,000 at start-up) and returns up to five complaints of the same predicted "
            "category whose cosine similarity is at least 0.55; every new complaint is then added to the index."
        ),
    ]

    # ---- 8 System design and implementation ----------------------------
    s += chapter(8, marks)
    s += [
        table(
            [
                ["Source module", "Role in the system"],
                ["config.py", "Paths, ten categories, urgency levels, SLA hours and department routing table."],
                ["generate_dataset.py", "Synthetic dataset generator with label noise and typing-error injection."],
                ["preprocessing.py", "Tokenization, stop-word removal and Porter stemming."],
                ["lexicons.py / features.py", "Rule-based urgency and frustration scoring; 11 engineered features."],
                ["category_model.py / urgency_model.py", "Category and urgency classifiers: model selection, training, prediction."],
                ["duplicates.py", "TF-IDF similarity index for repeat-complaint detection."],
                ["router.py", "Priority blending, department routing, SLA deadline and escalation."],
                ["pipeline.py", "ComplaintAnalyzer: the path from raw text to a routed ticket."],
                ["main.py (api)", "FastAPI endpoints, validation, persistence and dashboard statistics."],
            ],
            [150.0, 340.4],
        ),
        h2("8.1 Database design (complaints table)"),
        table(
            [
                ["Field(s)", "Purpose"],
                ["complaint_id, text, location", "Primary key, complaint text and optional location."],
                ["category, category_confidence", "Predicted category and its confidence."],
                ["ml_predicted_urgency, lexicon_urgency_score", "Classifier's urgency class and normalized lexicon score."],
                ["priority_score, final_urgency_level", "Blended 0-100 score and the resulting urgency level."],
                ["department, escalated_department", "Responsible department and, for Critical tickets, the emergency cell."],
                ["sla_hours, sla_deadline", "Promised response time and its deadline."],
                ["duplicate_count, similar_complaint_ids, explanation", "Repeat-report evidence and the audit explanation."],
                ["status, submitted_at", "Workflow status (Open, In Progress, Resolved) and timestamp."],
            ],
            [215.0, 275.4],
        ),
        h2("8.2 Persistence and concurrency"),
        body(
            "The application uses SQLAlchemy with SQLite by default; moving to PostgreSQL or MySQL only requires "
            "changing the database URL. The trained analyzer is loaded once at start-up, and a lock protects its "
            "in-memory duplicate index because that index is updated on every submission."
        ),
    ]

    # ---- 9 Application features and user workflow ----------------------
    s += chapter(9, marks)
    s += [
        h2("9.1 Complaint submission"),
        body(
            "On the Report an issue tab, a citizen describes the problem in their own words (10 to 4,000 "
            "characters) and may enter a location. The system returns a ticket showing the ticket ID, urgency "
            "level, priority score out of 100, category with confidence, routed department, SLA deadline, number "
            "of similar reports and the keywords that triggered the urgency flag."
        ),
        h2("9.2 Staff triage dashboard"),
        body(
            "The Staff dashboard tab shows summary tiles (total, open, in progress, resolved and average "
            "priority), breakdowns by category, urgency and department, and a complaints table sorted by "
            "priority score. Staff can filter by status, urgency, category and department."
        ),
        h2("9.3 Status workflow"),
        body(
            "Every ticket starts as Open. Staff change it to In Progress or Resolved from the dashboard, which "
            "calls the API and updates the stored record."
        ),
        h2("9.4 REST API"),
        table(
            [
                ["Method and path", "Description"],
                ["GET /health", "Health check; reports whether the trained models are loaded."],
                ["POST /complaints", "Submit a complaint and receive the routed ticket."],
                ["GET /complaints", "List complaints with category, urgency, department and status filters."],
                ["GET /complaints/{id}", "Fetch a single complaint."],
                ["PATCH /complaints/{id}/status", "Set the status to Open, In Progress or Resolved."],
                ["GET /stats/dashboard", "Aggregate counts for the staff dashboard."],
            ],
            [170.0, 320.4],
        ),
        h2("9.5 Command-line demo and explainability"),
        body(
            "The script scripts/demo_cli.py analyzes complaints without the API or database. Every ticket "
            "carries an explanation listing the matched urgent, dampener and frustration terms, the classifier "
            "confidences, the blended score and any repeat-report boost."
        ),
    ]

    # ---- 10 Result and analysis ----------------------------------------
    s += chapter(10, marks)
    s += [
        body(
            "The models were evaluated on a held-out test set of 600 complaints (a 20% stratified split) that "
            "was not used for training or model selection. Model selection used 5-fold stratified "
            "cross-validation on the 2,400 training complaints only. The test-set results of the two selected "
            "models are shown below."
        ),
        table(
            [
                ["Task", "Selected model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
                ["Category (10 classes)", "Linear SVM (calibrated)", "98.33%", "0.984", "0.983", "0.983"],
                ["Urgency (4 classes)", "Logistic Regression", "94.50%", "0.951", "0.939", "0.944"],
            ],
            [110.0, 122.0, 58.0, 68.0, 66.0, 64.4],
        ),
        h2("10.1 Model selection"),
        table(
            [
                ["Task", "Candidate model", "Mean macro-F1 (5-fold CV)", "Std"],
                ["Category", "Multinomial Naive Bayes", "0.9796", "0.0059"],
                ["Category", "Logistic Regression", "0.9804", "0.0054"],
                ["Category", "Linear SVM (selected)", "0.9808", "0.0055"],
                ["Urgency", "Logistic Regression (selected)", "0.9377", "0.0127"],
                ["Urgency", "Linear SVM", "0.9371", "0.0114"],
                ["Urgency", "Random Forest", "0.9367", "0.0123"],
            ],
            [90.0, 170.0, 150.0, 80.4],
        ),
        Spacer(1, 10),
        body(
            "The differences between candidates are smaller than the fold-to-fold standard deviation, so the "
            "candidate models perform practically equally on this dataset; the selected model is the one with "
            "the highest mean."
        ),
        h2("10.2 Interpretation"),
        body(
            "The category classifier misclassified 10 of 600 test complaints. Six categories were classified "
            "perfectly, and every error occurred inside two related pairs: Roads &amp; Infrastructure with "
            "Street Lighting, and Sanitation &amp; Waste Management with Health &amp; Sanitation. These are the "
            "pairs that the dataset generator swaps when it injects category label noise."
        ),
        body(
            "The urgency classifier misclassified 33 of 600 complaints. Its ordinal mean absolute error is "
            "0.055 and every error was to an adjacent level, so no Critical complaint was predicted as Low or "
            "Medium. Critical complaints reached a precision of 0.970 and a recall of 0.914: 64 of 70 were "
            "found and the other six were classified as High."
        ),
        h2("10.3 Independent reproduction"),
        body(
            "As an additional check, the dataset was re-read, the 80/20 split with random state 42 was "
            "regenerated (identical to the stored split) and both selected models were retrained from scratch "
            "with scikit-learn 1.9.1 without saving any files. This reproduced 98.33% category accuracy and "
            "94.50% urgency accuracy with an ordinal MAE of 0.055, matching the committed evaluation report."
        ),
        body(
            "The complete routing decision was also evaluated. After blending the urgency classifier with the "
            "lexicon score (without the repeat-report boost), the final urgency level matched the label for "
            "93.83% of the test complaints, slightly below the classifier alone. The blend changed 4 of 600 "
            "predictions (two Critical to High and two Low to Medium)."
        ),
        note(
            "Evaluation caution:",
            "the dataset is synthetic and template-based, and 93.2% of the test complaints have a training "
            "complaint with a cosine similarity of at least 0.55, so accuracy on real citizen complaints is "
            "likely to be lower. The TF-IDF vectorizers are fitted inside the training pipeline, so no test "
            "data leaks into the models; only the duplicate index is deliberately built from the full corpus.",
        ),
        h2("10.4 Sample pipeline outputs"),
        table(
            [
                ["Complaint (summary)", "Category (confidence)", "Level (score)", "Routed to", "SLA"],
                [
                    "Massive pothole, two-wheeler already crashed, \"fix immediately\"",
                    "Roads & Infrastructure (0.91)",
                    "Critical (98.6)",
                    "Public Works Department (PWD); escalated to Disaster & Emergency Response Cell",
                    "4 h",
                ],
                [
                    "Overgrown park grass, \"no rush\"",
                    "Parks & Public Property (0.96)",
                    "Low (36.2)",
                    "Parks & Horticulture Department",
                    "168 h",
                ],
                [
                    "No water for two days, elderly residents struggling, \"emergency\"",
                    "Water Supply (0.95)",
                    "Critical (97.1)",
                    "Municipal Water Board; escalated to Disaster & Emergency Response Cell",
                    "4 h",
                ],
                [
                    "One streetlight out for a few days",
                    "Street Lighting (0.95)",
                    "Medium (50.0)",
                    "Electrical Maintenance Cell (PWD)",
                    "72 h",
                ],
            ],
            [122.0, 92.0, 62.0, 172.4, 42.0],
        ),
        Spacer(1, 10),
        body(
            "These are the built-in samples of scripts/demo_cli.py. In the second sample, four similar "
            "complaints in the seeded corpus triggered the +12 repeat-report boost. Analysing one complaint "
            "end to end (models, lexicon, duplicate search and routing) took about 9 ms on the development "
            "machine."
        ),
    ]

    # ---- 11 Testing and validation -------------------------------------
    s += chapter(11, marks)
    s += [
        h2("11.1 Testing support in the project"),
        *bullets(
            [
                "The pytest suite contains 34 automated tests, and all of them pass: 9 API, 7 lexicon, 4 pipeline, "
                "7 preprocessing and 7 routing tests.",
                "API tests use FastAPI's TestClient with an in-memory SQLite database, so the real database file "
                "is never touched.",
                "Routing tests verify the department mapping, SLA deadlines, critical escalation, the repeat-report "
                "boost and the 100-point cap.",
                "Pipeline tests check that a complete ticket is produced, that empty text is rejected and that a "
                "repeated complaint is detected as a duplicate.",
                "The evaluation script reports accuracy, precision, recall, F1, confusion matrices and ordinal MAE "
                "for the held-out test set.",
            ]
        ),
        h2("11.2 Functional validation areas"),
        table(
            [
                ["Area", "Expected behavior"],
                ["Preprocessing", "Text is lowercased, URLs and punctuation are removed, stop-words dropped and tokens stemmed."],
                ["Lexicon scoring", "Emergency text scores above 0.8 and minor text below 0.4; repeats and durations are detected."],
                ["Classification", "A complaint receives a category with confidence and an urgency class."],
                ["Duplicate detection", "A resubmitted complaint is matched to its earlier copy."],
                ["Routing", "Each category maps to its department; Critical tickets are escalated with a 4-hour SLA."],
                ["REST API", "Valid complaints return 201, short text or an invalid status returns 422, unknown IDs return 404."],
                ["Dashboard statistics", "Counts reflect the complaints that have been submitted."],
            ],
            [130.0, 360.4],
        ),
        h2("11.3 Validation limitation"),
        body(
            "The frontend has no automated tests, and the system was validated only on synthetic data. The "
            "blend weights, score thresholds and department table are design choices that were not compared "
            "with real municipal outcomes."
        ),
    ]

    # ---- 12 Security and privacy ---------------------------------------
    s += chapter(12, marks)
    s += [
        body(
            "Because the application handles citizen-submitted text and location details, security and privacy "
            "are important aspects of the implementation."
        ),
        h2("12.1 Existing measures"),
        *bullets(
            [
                "NLP processing is fully local; no external AI API or cloud service receives complaint text.",
                "The pipeline runs offline after installation because no NLTK corpora need to be downloaded.",
                "Pydantic schemas validate input: complaint text must be 10 to 4,000 characters and the location "
                "at most 200 characters.",
                "Status updates are restricted to Open, In Progress and Resolved.",
                "SQLAlchemy builds parameterized queries for all filters, which prevents SQL injection.",
                "CORS is limited to the local frontend origins (localhost:5173 and 127.0.0.1:5173).",
                "A lock protects the shared duplicate index against concurrent requests.",
            ]
        ),
        h2("12.2 Deployment considerations"),
        *bullets(
            [
                "The API has no authentication or role separation, so the staff endpoints must be protected "
                "before a real deployment.",
                "The CORS origins, the API address used by the frontend and the database URL should move to "
                "environment configuration.",
                "Complaint text may contain personal information; access control, retention rules and encryption "
                "at rest should be added.",
                "SQLite should be replaced by PostgreSQL or MySQL for concurrent multi-user use.",
                "Priority scores are decision-support indicators, and a human should review Critical or "
                "sensitive tickets.",
            ]
        ),
    ]

    # ---- 13 Limitations and future scope -------------------------------
    s += chapter(13, marks)
    s += [
        h2("13.1 Current limitations"),
        *bullets(
            [
                "The dataset is synthetic and template-based, so the reported accuracy is not evidence of "
                "performance on real citizen complaints.",
                "The pipeline is English-only; code-mixed and regional-language complaints are not supported.",
                "Urgency is treated as four-class classification rather than ordinal regression.",
                "The blend weights, thresholds and repeat boost are hand-set, and the blend slightly lowered "
                "urgency accuracy on the test set (93.83% against 94.50%).",
                "The duplicate index is seeded with the synthetic corpus and lives in memory, so complaints "
                "submitted later are not remembered by the index after a restart.",
                "Each complaint receives exactly one category, even when it describes more than one issue.",
                "The API has no authentication and the frontend has no automated tests.",
            ]
        ),
        h2("13.2 Future scope"),
        *bullets(
            [
                "Validate and retrain the models on real municipal complaint data, including short, messy and "
                "code-mixed text.",
                "Add support for regional languages such as Hindi and Marathi.",
                "Fine-tune a transformer such as DistilBERT or IndicBERT while keeping the lexicon explanation layer.",
                "Model urgency with ordinal regression or an ordinal loss.",
                "Tune the blend weights and thresholds using staff feedback.",
                "Use active learning so that staff corrections to category and urgency feed periodic retraining.",
                "Store the duplicate index in the database and merge similar reports into a single ticket.",
                "Add staff authentication, role-based access, map-based views and frontend tests.",
            ]
        ),
    ]

    # ---- 14 Conclusion --------------------------------------------------
    s += chapter(14, marks)
    s += [
        body(
            "CivicLens AI demonstrates a practical application of Natural Language Processing and machine "
            "learning to municipal complaint handling. The system integrates a FastAPI service, SQLite "
            "persistence, a React interface, supervised category and urgency classification, lexicon-based "
            "scoring, duplicate detection and a deterministic routing engine that turns raw complaint text into "
            "a prioritized, routed and explainable ticket."
        ),
        body(
            "The project uses different techniques for different sub-problems. TF-IDF with a calibrated Linear "
            "SVM identifies the category, TF-IDF combined with lexicon features and Logistic Regression "
            "estimates urgency, cosine similarity finds repeat reports, and a transparent scoring formula turns "
            "these signals into a priority score, a department and an SLA deadline."
        ),
        body(
            "On the held-out test set of 600 synthetic complaints, the category classifier reached 98.33% "
            "accuracy and the urgency classifier 94.50% accuracy with an ordinal MAE of 0.055, and both results "
            "were reproduced independently. Because the dataset is synthetic and template-based, these figures "
            "demonstrate the pipeline rather than real-world performance, and validation on real complaints is "
            "the most important next step."
        ),
        body(
            "Overall, the project provides a clear demonstration of how NLP and machine learning can be "
            "integrated into a useful civic application in which every priority decision can be traced back to "
            "its evidence."
        ),
    ]

    # ---- 15 References --------------------------------------------------
    s += chapter(15, marks)
    refs = [
        "[1] G. Salton, C. Buckley, \"Term-weighting approaches in automatic text retrieval,\" Information "
        "Processing & Management, 1988.",
        "[2] M. F. Porter, \"An algorithm for suffix stripping,\" Program, 1980.",
        "[3] A. McCallum, K. Nigam, \"A Comparison of Event Models for Naive Bayes Text Classification,\" "
        "AAAI-98 Workshop on Learning for Text Categorization, 1998.",
        "[4] T. Joachims, \"Text Categorization with Support Vector Machines: Learning with Many Relevant "
        "Features,\" European Conference on Machine Learning (ECML), 1998.",
        "[5] F. Sebastiani, \"Machine Learning in Automated Text Categorization,\" ACM Computing Surveys, 2002.",
        "[6] C. J. Hutto, E. Gilbert, \"VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social "
        "Media Text,\" Proceedings of ICWSM, 2014.",
        "[7] C. D. Manning, P. Raghavan, H. Schütze, Introduction to Information Retrieval, Cambridge University "
        "Press, 2008.",
        "[8] P. A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro, C. Hervás-Martínez, "
        "\"Ordinal Regression Methods: Survey and Experimental Study,\" IEEE Transactions on Knowledge and Data "
        "Engineering, 2016.",
        "[9] L. Breiman, \"Random Forests,\" Machine Learning, 2001.",
        "[10] F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of Machine Learning "
        "Research, 2011.",
        "[11] J. Platt, \"Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized "
        "Likelihood Methods,\" Advances in Large Margin Classifiers, MIT Press, 1999.",
        "[12] CivicLens AI source project, including the NLP pipeline, routing engine, FastAPI service, React "
        "frontend, trained models, automated tests and evaluation report.",
        "[13] Synthetic complaint dataset generated by civiclens/data/generate_dataset.py (3,000 complaints, "
        "random seed 42).",
    ]
    s += [body(r) for r in refs]
    s += [Spacer(1, 60), Paragraph("— END OF REPORT —", END)]
    return s


# --------------------------------------------------------------------------
# Build (two passes so the table of contents carries real page numbers)
# --------------------------------------------------------------------------
def _build_once(out: Path, toc_pages: dict[str, int]) -> dict[str, int]:
    marks: dict[str, int] = {}
    doc = BaseDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_Y,
        bottomMargin=MARGIN_Y,
        title=PDF_TITLE,
        author=", ".join(n.split(" (")[0] for n in COVER["students"]),
        subject=f"{COVER['institute']} - {COVER['department']}",
        creator="CivicLens AI report generator (ReportLab)",
    )
    cover_frame = Frame(
        MARGIN_X, MARGIN_Y, 100, 100, id="cover", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0
    )
    body_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates(
        [
            PageTemplate(id="Cover", frames=[cover_frame], onPage=draw_cover),
            PageTemplate(id="Body", frames=[body_frame], onPage=draw_body_page),
        ]
    )
    doc.build(story(toc_pages, marks))
    return marks


def build(out: Path) -> dict[str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        first = _build_once(Path(tmp) / "pass1.pdf", {})
    final = _build_once(out, first)
    assert final == first, "pagination changed between passes"
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output PDF path")
    args = parser.parse_args()
    pages = build(args.out)
    print(f"Wrote {args.out}")
    print("Chapter start pages:", {k: v for k, v in pages.items() if k.startswith("ch")})
