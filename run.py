#!/Users/jessedecker/miniconda3/envs/gis/bin/python

from viz import app
import os

app.secret_key = os.urandom(24)
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)