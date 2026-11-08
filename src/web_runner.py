import argparse
import time
import webbrowser
from threading import Thread
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn


def open_browser_when_ready(url):
    deadline = time.monotonic() + 30

    while time.monotonic() < deadline:
        try:
            with urlopen(f"{url}/api/health", timeout=1) as response:
                if response.status == 200:
                    if not webbrowser.open_new_tab(url):
                        print(f"Abra o navegador em {url}")
                    return
        except (OSError, URLError, TimeoutError):
            time.sleep(0.25)

    print(f"Servidor ainda não iniciou. Acesse {url} quando estiver disponível.")


def main():
    parser = argparse.ArgumentParser(
        description="Inicia a aplicação e abre a interface no navegador."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    Thread(
        target=open_browser_when_ready,
        args=(url,),
        daemon=True,
    ).start()

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
