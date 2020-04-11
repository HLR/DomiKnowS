from urllib import request
from urllib.error import HTTPError
import time
import json

# Using github markdown API
# Document:
# https://developer.github.com/v3/markdown/
# Endpoint:
# https://api.github.com/markdown
# Example request:
# {
#   "text": "Hello world github/linguist#1 **cool**, and #1!",
#   "mode": "gfm",
#   "context": "github/gollum"
# }

def to_html(md, mode='markdown', context=None, encoding='utf-8'):
    data = {'text': md, 'mode': mode}
    if context is not None:
        data['context'] = context
    req = request.Request('https://api.github.com/markdown', json.dumps(data).encode(encoding))

    while True:
        try:
            with request.urlopen(req) as resp:
                return resp.read().decode(encoding)
        except HTTPError as e:
            # https://developer.github.com/v3/#rate-limiting
            reset = float(e.headers['X-RateLimit-Reset'])
            time.sleep((reset - time.time()) + 1)
