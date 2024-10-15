
https://requests.readthedocs.io/en/latest/user/advanced/

# Session Objects

# Request and Response Objects

# Prepared Requests

# SSL Cert Verification

# Client Side Certificates

# Body Content Workflow

# Keep-Alive

# Streaming Uploads

```python
with open('massive-body', 'rb') as f:
    requests.post('http://some.url/streamed', data=f)
```
# Chunk-Encoded Requests

# POST Multiple Multipart-Encoded Files

# Custom Authentication

# Event Hooks

# Streaming Requests

# Proxy

# SOCKS

# HTTP Verbs

# Example: Automatic Retries

```python
from urllib3.util import Retry
from requests import Session
from requests.adapters import HTTPAdapter

s = Session()
retries = Retry(
    total=3,
    backoff_factor=0.1,
    status_forcelist=[502, 503, 504],
    allowed_methods={'POST'},
)
s.mount('https://', HTTPAdapter(max_retries=retries))
```
