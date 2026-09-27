"""Synthetic citizen-complaint dataset generator.

CivicLens AI is trained on a *synthetic* corpus rather than a scraped or
licensed municipal 311 dataset, since no such dataset was available to
redistribute for this project. To keep the labels meaningful (not just
random noise), each complaint is assembled from hand-written template
sentences that were themselves tiered by urgency: "Critical" templates use
language a real emergency complaint would use (injury, immediate danger,
health hazards), "Low" templates use the soft, patient language of a minor
suggestion. Category, urgency, ward, street, duration and sign-off are then
composed combinatorially with a fixed random seed, so the generation is
reproducible but produces thousands of distinct complaint texts.

This mirrors how a human annotator would have labelled a real dataset
(reading the text and judging urgency from its content) rather than leaking
the label through metadata the model never sees, so a classifier trained on
this data is learning genuine textual urgency cues, not shortcuts.

Run as a script to (re)write data/raw/complaints_dataset.csv:
    python -m civiclens.data.generate_dataset
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from civiclens.config import CATEGORIES, DATASET_PATH, RANDOM_SEED, ZONES

N_PER_CATEGORY = 300

STREET_NAMES = [
    "MG Road", "Station Road", "Church Street", "Lake View Road", "Hilltop Avenue",
    "Market Lane", "Mill Road", "Riverside Drive", "Park Street", "Gandhi Nagar Road",
    "Nehru Colony Road", "Industrial Estate Road", "Rose Garden Street", "Elm Street",
    "Maple Avenue", "5th Cross Road", "Temple Street", "Harbor Road", "Old Bridge Road",
    "Sunset Boulevard",
]

DURATIONS_TIER = {
    "Critical": ["since this morning", "for the past two days", "right now", "as of last night"],
    "High": ["for over a week", "for the past 10 days", "for two weeks now", "since last Monday"],
    "Medium": ["for a few days", "for about a week", "for the last few days", "on and off for a week"],
    "Low": ["for a little while", "recently", "for a few weeks", "lately"],
}

SALUTATIONS = ["Dear Sir/Madam,", "To whom it may concern,", "Hi,", "Hello,", ""]

SIGNOFFS_TIER = {
    "Critical": [
        "Please treat this as an emergency and act immediately.",
        "This needs urgent attention before someone gets hurt.",
        "Requesting immediate action, this cannot wait.",
    ],
    "High": [
        "Please resolve this as soon as possible.",
        "Kindly prioritize this complaint.",
        "Hoping for a quick resolution, this has gone on too long.",
    ],
    "Medium": [
        "Please look into this when you get a chance.",
        "Kindly arrange for this to be fixed soon.",
        "Would appreciate an update on this.",
    ],
    "Low": [
        "No rush, just wanted to flag it.",
        "Thanks in advance.",
        "Whenever convenient, thank you.",
        "",
    ],
}

REPEAT_PHRASES = [
    "This is the second time I'm reporting this issue.",
    "I had already reported this last month with no response.",
    "Several of my neighbours have also complained about this.",
]

# --------------------------------------------------------------------------
# Realism noise. Real complaint triage data is never perfectly clean: two
# annotators disagree on how urgent something is, citizens mistype things on
# their phones, and some categories are genuinely ambiguous. Without this,
# every label maps deterministically onto a fixed vocabulary and a
# classifier trivially hits ~100% accuracy, which is a red flag for a real
# evaluation. These rates are injected *after* text generation and are
# reported openly in the evaluation report rather than hidden.
URGENCY_LABEL_NOISE_RATE = 0.08
CATEGORY_LABEL_NOISE_RATE = 0.03
TEXT_CORRUPTION_RATE_WEIGHTS = [0, 0, 0, 1, 1, 2]  # ~50% none, ~33% one typo, ~17% two

# Only swap within genuinely confusable neighbours (mirrors real annotation
# disagreement), never to a wholly unrelated category.
CONFUSABLE_CATEGORIES = {
    "Sanitation & Waste Management": "Health & Sanitation",
    "Health & Sanitation": "Sanitation & Waste Management",
    "Roads & Infrastructure": "Street Lighting",
    "Street Lighting": "Roads & Infrastructure",
    "Noise & Pollution": "Health & Sanitation",
    "Public Safety & Law Enforcement": "Other / General Administration",
}


def _typo(word: str, rng: random.Random) -> str:
    """Drop one interior character, e.g. a fast-typed 'streeet' -> 'stret'."""
    if len(word) < 4:
        return word
    i = rng.randint(1, len(word) - 2)
    return word[:i] + word[i + 1:]


def _corrupt_text(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    if len(words) < 6:
        return text
    n_edits = rng.choice(TEXT_CORRUPTION_RATE_WEIGHTS)
    for _ in range(n_edits):
        if rng.random() < 0.5:
            idx = rng.randrange(len(words))
            words[idx] = _typo(words[idx], rng)
        elif len(words) > 2:
            idx = rng.randrange(len(words) - 1)
            words[idx] = words[idx] + words[idx + 1]
            del words[idx + 1]
    return " ".join(words)

# --------------------------------------------------------------------------
# Category -> urgency tier -> template sentences.
# {street}, {ward}, {duration} are filled in at generation time.
# --------------------------------------------------------------------------
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "Roads & Infrastructure": {
        "Critical": [
            "There is a massive pothole on {street} in {ward} that already caused a two-wheeler accident, and {duration} it has only gotten worse — someone could get seriously injured.",
            "A section of the road on {street} has completely caved in {duration}, exposing an open drain that is extremely dangerous for night-time traffic.",
            "The footbridge near {street} is showing large visible cracks and locals fear it could collapse {duration}.",
        ],
        "High": [
            "A large pothole has formed on {street} in {ward} {duration} and several vehicles have already been damaged trying to avoid it.",
            "The road surface on {street} is badly broken {duration}, causing heavy traffic jams every single morning.",
            "The road divider near {street} was knocked down {duration} and never replaced, creating a serious hazard for oncoming traffic.",
        ],
        "Medium": [
            "The road on {street} in {ward} has several small potholes that have been there {duration}.",
            "Footpath tiles on {street} are broken and uneven, making walking difficult, {duration}.",
            "Road markings near the {street} junction have faded {duration} and need repainting.",
        ],
        "Low": [
            "Just a minor observation — the road on {street} could use resurfacing whenever the department gets a chance.",
            "A small crack has appeared on the footpath along {street}; not urgent but worth noting.",
            "It would be nice if the traffic signage on {street} were repainted at some point.",
        ],
    },
    "Water Supply": {
        "Critical": [
            "There has been absolutely no water supply in {ward} {duration}; elderly residents are struggling without water for drinking or bathing.",
            "A major water pipeline has burst on {street}, flooding homes {duration} — this needs to be fixed immediately.",
            "Sewage appears to be mixing with the drinking water supply on {street} and several residents have already fallen ill.",
        ],
        "High": [
            "Water supply to {ward} has been extremely irregular {duration}, with taps running dry for most of the day.",
            "There is a visible leak in the main pipeline on {street} that has been wasting water {duration}.",
            "Water pressure in {ward} has dropped drastically {duration}, and upper floors are getting almost nothing.",
        ],
        "Medium": [
            "Water supply timing in {ward} has shifted without any notice {duration}, causing real inconvenience.",
            "The water in {ward} has had an unusual odor {duration}; residents would like it checked.",
            "A minor leak near the meter on {street} has been dripping {duration}.",
        ],
        "Low": [
            "Water pressure on {street} is a little lower than usual lately, nothing urgent.",
            "Would appreciate an update on the water supply schedule for {ward}.",
            "A borewell near {street} could use routine maintenance when convenient.",
        ],
    },
    "Electricity & Power": {
        "Critical": [
            "Live wires are hanging dangerously low on {street} in {ward} {duration}; a stray animal was electrocuted yesterday.",
            "There has been a complete power outage in {ward} {duration}, and it is affecting a nearby clinic's equipment.",
            "Sparking has been seen coming from the transformer near {street} and residents are worried about a fire.",
        ],
        "High": [
            "Frequent power cuts have been happening in {ward} {duration}, sometimes lasting several hours at a time.",
            "Voltage fluctuations on {street} have already damaged appliances in multiple homes {duration}.",
            "The transformer near {street} has been making a loud humming noise {duration} and residents are concerned.",
        ],
        "Medium": [
            "Power supply in {ward} briefly fluctuates {duration}, causing minor inconvenience.",
            "An electricity pole on {street} has been leaning slightly {duration} and should be checked.",
            "Streetside electric meters on {street} appear misaligned {duration}.",
        ],
        "Low": [
            "Power supply has mostly been fine, just the occasional flicker in {ward}.",
            "Would like to request a routine inspection of the wiring near {street}.",
            "Meter reading on {street} seems slightly off; not urgent.",
        ],
    },
    "Sanitation & Waste Management": {
        "Critical": [
            "Garbage has not been collected in {ward} {duration} and it is now overflowing, attracting rats and stray animals — a serious health hazard.",
            "Raw sewage is flowing openly onto {street} {duration}; the smell and health risk are unbearable.",
            "A blocked drain on {street} has caused sewage to back up into homes {duration}.",
        ],
        "High": [
            "Waste collection in {ward} has been missed {duration}, and the bins are now overflowing onto the street.",
            "The public toilet near {street} has been unusable {duration} due to a complete lack of cleaning.",
            "A dead animal has been lying uncollected on {street} {duration}.",
        ],
        "Medium": [
            "Garbage collection in {ward} has been irregular {duration}, sometimes skipping a day entirely.",
            "Litter has been steadily accumulating near the market on {street} {duration}.",
            "The community bin on {street} needs to be emptied more often; it has been full {duration}.",
        ],
        "Low": [
            "Street sweeping in {ward} could be a little more frequent, no rush.",
            "A minor litter issue near {street} could use attention when convenient.",
            "Requesting an additional dustbin be placed on {street}.",
        ],
    },
    "Street Lighting": {
        "Critical": [
            "All the streetlights on {street} have been out {duration}, and there was an attempted mugging in the dark last night.",
            "The street near the school on {street} has had no lighting {duration}, and children walk home in the dark — this is dangerous.",
            "A live wire is exposed at the base of a fallen streetlight pole on {street}.",
        ],
        "High": [
            "Several streetlights along {street} in {ward} have not worked {duration}, making the area unsafe at night.",
            "The streetlight pole near {street} has been leaning dangerously {duration} after last week's storm.",
            "Half the lights on the main road through {ward} have been flickering or dead {duration}.",
        ],
        "Medium": [
            "One streetlight near {street} has been out {duration}; the rest of the street is fine.",
            "Streetlights in {ward} have been switching on later than they should {duration}.",
            "A streetlight on {street} flickers intermittently {duration}.",
        ],
        "Low": [
            "Streetlighting on {street} is generally okay, just a little dim in one spot.",
            "Could the streetlight timing on {street} be adjusted slightly? Not urgent.",
            "Minor request to add one more streetlight near the {street} junction.",
        ],
    },
    "Public Safety & Law Enforcement": {
        "Critical": [
            "A group has been openly selling drugs near the {street} playground {duration}, and parents are afraid to let children outside.",
            "There was a violent altercation on {street} last night and residents fear for their safety {duration}.",
            "An unattended, suspicious package has been sitting near the {street} market since this morning.",
        ],
        "High": [
            "Reports of chain-snatching near {street} have increased {duration}, and residents are asking for more police patrols.",
            "A group of stray dogs has been aggressively chasing pedestrians on {street} {duration}.",
            "Illegal parking on {street} has been blocking emergency vehicle access {duration}.",
        ],
        "Medium": [
            "Loitering near the {street} market has increased slightly {duration}; more visible patrolling would help.",
            "Traffic violations at the {street} junction have gone unchecked {duration}.",
            "A broken boundary wall near {street} needs to be repaired for security.",
        ],
        "Low": [
            "General safety in {ward} has been fine, just requesting occasional patrol visibility on {street}.",
            "Would like a few more CCTV cameras considered for {street} eventually.",
            "Minor suggestion: better signage for the pedestrian crossing on {street}.",
        ],
    },
    "Parks & Public Property": {
        "Critical": [
            "A large tree in the {ward} park is uprooted and blocking the only exit path {duration}, blocking emergency access.",
            "The children's play equipment in the {ward} park is broken with exposed sharp edges {duration}; a child was already injured.",
            "A public building's ceiling near {street} partially collapsed {duration}.",
        ],
        "High": [
            "The park in {ward} has been locked and inaccessible {duration} despite being public property.",
            "Benches and play equipment in the {street} park are badly damaged {duration} and unsafe to use.",
            "A boundary fence around the {ward} park has fallen {duration}, letting stray animals in.",
        ],
        "Medium": [
            "The park in {ward} needs its grass cut; it has been overgrown {duration}.",
            "A few benches near {street} could use repainting and minor repair.",
            "The public restroom at the {ward} park needs better upkeep {duration}.",
        ],
        "Low": [
            "The {ward} park is nice overall, just suggesting more dustbins be added.",
            "Would be great to have a few more benches in the {street} park eventually.",
            "Minor suggestion to add some shade trees near the {ward} playground.",
        ],
    },
    "Noise & Pollution": {
        "Critical": [
            "A factory near {street} has been releasing thick black smoke {duration}, and several residents are reporting breathing difficulty.",
            "Extremely loud construction work has been running through the night on {street} {duration}, affecting patients at the nearby clinic.",
            "A strong chemical smell is coming from the drain on {street} and residents are feeling dizzy.",
        ],
        "High": [
            "A commercial generator near {street} runs loudly through the night {duration}, disturbing sleep for the whole block.",
            "Open burning of garbage has been happening on {street} {duration}, filling the area with thick smoke.",
            "Noise from the banquet hall near {street} regularly exceeds limits late into the night {duration}.",
        ],
        "Medium": [
            "Traffic noise on {street} has increased {duration}; a speed breaker or noise barrier might help.",
            "Dust from ongoing construction near {street} has been settling on homes {duration}.",
            "A loudspeaker near {street} plays louder than necessary during events.",
        ],
        "Low": [
            "Noise levels in {ward} are generally acceptable, just a minor observation about early morning deliveries.",
            "Would appreciate a reminder notice about noise limits sent to shops on {street}.",
            "Minor request to review the loudspeaker permit timing near {street}.",
        ],
    },
    "Health & Sanitation": {
        "Critical": [
            "Stagnant water near {street} has become a mosquito breeding ground and several dengue cases have already been reported in {ward}.",
            "The community health center near {street} has run out of essential medicines {duration}, putting patients at risk.",
            "A suspected contaminated food source near the {street} market has caused multiple cases of food poisoning.",
        ],
        "High": [
            "Mosquito fogging has not been carried out in {ward} {duration} despite rising complaints of bites.",
            "The public health clinic near {street} has been short-staffed {duration}, leading to very long patient wait times.",
            "Stray animal carcasses near {street} have not been disposed of {duration}, raising disease concerns.",
        ],
        "Medium": [
            "Vaccination camp schedules for {ward} have not been announced {duration}.",
            "The health clinic near {street} could use better signage and an appointment system.",
            "A minor pest-control follow-up is needed near {street}; the first treatment was {duration} ago.",
        ],
        "Low": [
            "General health services in {ward} have been adequate, just suggesting more awareness pamphlets.",
            "Would like more information on the health clinic hours near {street}.",
            "Minor suggestion to add a handwashing station near the {street} market.",
        ],
    },
    "Other / General Administration": {
        "Critical": [
            "Files related to an urgent property dispute on {street} appear to have been lost by the municipal office {duration}, and legal deadlines are at risk.",
            "A citizen was denied emergency documentation needed for hospital admission at the {ward} office {duration}.",
            "Suspected large-scale corruption in permit approvals near {street} is being reported by multiple residents.",
        ],
        "High": [
            "Applications submitted at the {ward} municipal office {duration} ago have still not been processed.",
            "Staff at the {street} administrative office have been consistently unhelpful and dismissive {duration}.",
            "The online portal for filing grievances has been down {duration}, forcing residents to travel in person.",
        ],
        "Medium": [
            "Response times from the {ward} municipal office have been slower than usual {duration}.",
            "Signage at the {street} government office is outdated and confusing.",
            "A request for a duplicate document at the {ward} office has been pending {duration}.",
        ],
        "Low": [
            "Overall service at the {ward} office has been fine, just a suggestion to extend counter hours slightly.",
            "Minor suggestion: add a token or queue system at the {street} office.",
            "Would appreciate an SMS update system for application status at the {ward} office.",
        ],
    },
}

# Roughly how urgency is distributed within each category (skews some
# categories towards being reported as more urgent than others, which is
# realistic: e.g. public-safety reports skew urgent, park suggestions don't).
CATEGORY_URGENCY_WEIGHTS = {
    "Roads & Infrastructure": {"Critical": 0.12, "High": 0.28, "Medium": 0.38, "Low": 0.22},
    "Water Supply": {"Critical": 0.15, "High": 0.30, "Medium": 0.35, "Low": 0.20},
    "Electricity & Power": {"Critical": 0.15, "High": 0.30, "Medium": 0.35, "Low": 0.20},
    "Sanitation & Waste Management": {"Critical": 0.18, "High": 0.27, "Medium": 0.33, "Low": 0.22},
    "Street Lighting": {"Critical": 0.10, "High": 0.25, "Medium": 0.40, "Low": 0.25},
    "Public Safety & Law Enforcement": {"Critical": 0.20, "High": 0.30, "Medium": 0.30, "Low": 0.20},
    "Parks & Public Property": {"Critical": 0.05, "High": 0.15, "Medium": 0.40, "Low": 0.40},
    "Noise & Pollution": {"Critical": 0.12, "High": 0.28, "Medium": 0.35, "Low": 0.25},
    "Health & Sanitation": {"Critical": 0.15, "High": 0.25, "Medium": 0.35, "Low": 0.25},
    "Other / General Administration": {"Critical": 0.05, "High": 0.15, "Medium": 0.40, "Low": 0.40},
}


@dataclass
class Complaint:
    text: str
    category: str
    urgency: str
    ward: str
    street: str
    submitted_at: str


def _compose_text(rng: random.Random, category: str, urgency: str) -> tuple[str, str, str]:
    ward = rng.choice(ZONES)
    street = rng.choice(STREET_NAMES)
    duration = rng.choice(DURATIONS_TIER[urgency])

    template = rng.choice(TEMPLATES[category][urgency])
    sentence = template.format(street=street, ward=ward, duration=duration)

    parts = []
    salutation = rng.choice(SALUTATIONS)
    if salutation:
        parts.append(salutation)
    parts.append(sentence)

    # Occasionally combine a second sentence from the same tier for variety.
    if rng.random() < 0.35:
        other_templates = [t for t in TEMPLATES[category][urgency] if t != template]
        if other_templates:
            second = rng.choice(other_templates).format(
                street=rng.choice(STREET_NAMES), ward=ward, duration=rng.choice(DURATIONS_TIER[urgency])
            )
            parts.append(second)

    # Occasionally mention this is a repeat complaint (feeds duplicate-boost narrative).
    if rng.random() < 0.15:
        parts.append(rng.choice(REPEAT_PHRASES))

    signoff = rng.choice(SIGNOFFS_TIER[urgency])
    if signoff:
        parts.append(signoff)

    text = " ".join(parts).strip()
    text = _corrupt_text(text, rng)
    return text, ward, street


def generate(n_per_category: int = N_PER_CATEGORY, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[Complaint] = []
    start_date = datetime(2025, 1, 1)

    for category in CATEGORIES:
        weights = CATEGORY_URGENCY_WEIGHTS[category]
        urgencies = list(weights.keys())
        probs = list(weights.values())

        seen_texts: set[str] = set()
        attempts = 0
        count = 0
        while count < n_per_category and attempts < n_per_category * 20:
            attempts += 1
            urgency = rng.choices(urgencies, weights=probs, k=1)[0]
            text, ward, street = _compose_text(rng, category, urgency)
            if text in seen_texts:
                continue
            seen_texts.add(text)
            day_offset = rng.randint(0, 270)
            submitted_at = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            rows.append(Complaint(text, category, urgency, ward, street, submitted_at))
            count += 1

    # Simulated annotator noise: perturb a small fraction of labels
    # independently of the text (see the constants' docstring above).
    urgency_order = ["Low", "Medium", "High", "Critical"]
    for r in rows:
        if rng.random() < URGENCY_LABEL_NOISE_RATE:
            idx = urgency_order.index(r.urgency)
            shift = rng.choice([-1, 1])
            r.urgency = urgency_order[max(0, min(3, idx + shift))]
        if r.category in CONFUSABLE_CATEGORIES and rng.random() < CATEGORY_LABEL_NOISE_RATE:
            r.category = CONFUSABLE_CATEGORIES[r.category]

    rng.shuffle(rows)
    df = pd.DataFrame([r.__dict__ for r in rows])
    df.insert(0, "complaint_id", [f"CMP{100000 + i}" for i in range(len(df))])
    return df


def main() -> None:
    df = generate()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Wrote {len(df)} synthetic complaints to {DATASET_PATH}")
    print(df["category"].value_counts())
    print(df["urgency"].value_counts())


if __name__ == "__main__":
    main()
