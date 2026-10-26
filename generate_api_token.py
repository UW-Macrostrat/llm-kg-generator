import os
from datetime import datetime, timedelta, timezone
from jose import jwt

secret = os.environ["SECRET_KEY"]
algorithm = os.environ["JWT_ENCRYPTION_ALGORITHM"]

payload = {
    "sub": "0009-0002-4083-2613",
    "role": "web_admin",
    "exp": datetime.now(timezone.utc) + timedelta(days=365),
}

token = jwt.encode(payload, secret, algorithm=algorithm)

print(token)