import sys
import os
from pathlib import Path

BASE_DIR = Path("C:/Users/VIJAY/.gemini/antigravity/scratch/dr_screening")
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

import asyncio
from backend.database import async_sessionmaker_factory
from backend.routers.screenings import list_xai_eligible_cases, get_screening_detail
from backend.models.user import User

async def main():
    print("Connecting to database...", flush=True)
    async with async_sessionmaker_factory() as session:
        user = User(id="test-clinician", email="clinician@dr-screening.com", role="clinician", is_active=True)
        print("Calling list_xai_eligible_cases...", flush=True)
        cases = await list_xai_eligible_cases(limit=20, db=session, current_user=user)
        print(f"XAI Cases returned: {len(cases)}", flush=True)
        assert len(cases) > 0, "No XAI cases returned!"
        
        for c in cases[:5]:
            print(f"  Case: {c['case_number']} | Patient: {c['patient_name']} | Image: {c['image_url']} | Grade: {c['predicted_grade']} ({c['predicted_label']})", flush=True)
            
        print("\nCalling get_screening_detail on first case...", flush=True)
        first_id = cases[0]["id"]
        detail = await get_screening_detail(first_id, db=session, current_user=user)
        print(f"Detail case_number: {detail['encounter']['case_number']}", flush=True)
        print(f"Detail images count: {len(detail['images'])}", flush=True)
        print(f"Detail image 0: {detail['images'][0]['image_url']}", flush=True)
        if detail.get("prediction"):
            print(f"Detail prediction grade: {detail['prediction']['predicted_class']} ({detail['prediction']['predicted_label']})", flush=True)
            print(f"Detail gradcam_url: {detail['prediction']['gradcam_url']}", flush=True)

    print("\nALL DIRECT XAI ENDPOINT CHECKS PASSED!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
