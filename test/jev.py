"""
Quick Jev API test — exercises all three primitives (Noul, Choice, Score)
in a single parallel request.

Setup:
    uv add typesafe-sdk python-dotenv      # or: pip install typesafe-sdk python-dotenv
    Set TYPESAFE_API_KEY in .env (project root)

Run:
    python test/jev.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = TypeSafeClient(
    api_key=os.environ["TYPESAFE_API_KEY"],
    model="jev-1.13.0", 
)


state = {
    "subject": "Duplicate charge on my account",
    "message": (
        "Hi, I was charged twice for order #A-104 ($49 each time). "
        "This is really frustrating — please refund the duplicate charge ASAP."
    ),
    "refund_policy": "Duplicate charges are eligible for a full refund.",
}

response = client.system_one(
    state=state,
    questions={
        "department": Choice(
            instructions="Which team should handle this ticket?",
            criteria={
                "billing":   "Payment, charges, invoices, or refund issues",
                "technical": "Bugs, errors, or integration problems",
                "sales":     "Pricing, plans, or account upgrades",
                "other":     "Anything that does not fit the above",
            },
        ),
        "frustration": Score(
            instructions="How frustrated does the customer appear?",
            criteria=[
                "Calm — just stating facts",
                "Mildly frustrated but polite",
                "Clearly angry, strong language or urgency",
            ],
        ),
        "refund_eligible": Noul(
            instructions="The stated refund policy covers this customer's situation."
        ),
        "is_urgent": Noul(
            instructions="The customer explicitly signals urgency or time-sensitivity."
        ),
    },
)


answers = response.answers

dept = answers["department"]
print(f"Department  : {dept.choice}  (confidence {dept.confidence:.2f})")
print(f"  probabilities: { {k: f'{v:.2f}' for k, v in dept.probabilities.items()} }")

frust = answers["frustration"]
print(f"\nFrustration : {frust.score:.2f} / 2  (confidence {frust.confidence:.2f})")
print(f"  level probabilities: {[f'{p:.2f}' for p in frust.probabilities]}")

print(f"\nRefund eligible : {answers['refund_eligible'].noul:.2f}  (1.0 = yes)")
print(f"Is urgent       : {answers['is_urgent'].noul:.2f}  (1.0 = yes)")


assert 0 <= answers["refund_eligible"].noul <= 1, "Noul must be a probability [0,1]"
print("\n✓ Self-check passed")
