import sys
import os
import uvicorn

# Redirect execution to the new modular package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from colab_server_pkg.main import app

if __name__ == "__main__":
    uvicorn.run("colab_server_pkg.main:app", host="0.0.0.0", port=8001, reload=False)
