"""Demo: surface past lessons for three new design issues.

Run:  python examples/demo.py
"""

from design_review_miner import KnowledgeIndex, keyword_profile
import pandas as pd

index = KnowledgeIndex.from_csv("data/design_review_records.csv")

new_issues = [
    "Laser weld penetration inconsistent on battery module busbar joints",
    "Plastic clip on center console fails to snap in during assembly",
    "Fasteners on skid plate corroding in coastal climate durability test",
]

for issue in new_issues:
    print("=" * 72)
    print(index.review(issue))
    print()

print("=" * 72)
print("KNOWLEDGE BASE TOP TERMS (TF-IDF weight):")
print(keyword_profile(pd.read_csv("data/design_review_records.csv")).to_string())
