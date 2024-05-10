import requests


def load_lottieurl(
    url="https://lottie.host/7af58fa9-62dc-4373-9464-40e087294535/pUrdD895QL.json",
):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
