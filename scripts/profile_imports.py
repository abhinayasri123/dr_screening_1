import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def check(name, fn):
    t0 = time.time()
    fn()
    print(f"[{time.time()-t0:.2f}s] {name}", flush=True)

check("1. backend.database", lambda: __import__("backend.database"))
check("2. backend.auth", lambda: __import__("backend.auth"))
check("3. backend.routers.auth", lambda: __import__("backend.routers.auth"))
check("4. backend.routers.patients", lambda: __import__("backend.routers.patients"))
check("5. backend.routers.predictions", lambda: __import__("backend.routers.predictions"))
check("6. backend.routers.reviews", lambda: __import__("backend.routers.reviews"))
check("7. backend.routers.federated", lambda: __import__("backend.routers.federated"))
check("8. backend.routers.metrics", lambda: __import__("backend.routers.metrics"))
check("9. backend.routers.screenings", lambda: __import__("backend.routers.screenings"))
check("10. backend.routers.models_admin", lambda: __import__("backend.routers.models_admin"))
check("11. backend.main", lambda: __import__("backend.main"))
