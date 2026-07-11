"""Demo: surface past lessons for three new design issues.

Run:  python demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_review_miner import KnowledgeIndex, keyword_profile
import pandas as pd

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "design_review_records.csv")

index = KnowledgeIndex.from_csv(csv_path)

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
print(keyword_profile(pd.read_csv(csv_path)).to_string())
