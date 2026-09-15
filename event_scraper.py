import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re
import hashlib
import time as time_module
from urllib.parse import urlparse


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

OUT_CSV = "new_ufc_event.csv"

# TARGET_EVENT = "UFC 326: Holloway vs. Oliveira 2"
TIMEOUT =  20
_SESSION = None

# regex patterns
_NONCE_RE = re.compile(r'nonce="([0-9a-fA-F]+)"')   #capures nonce string
_DIFF_RE = re.compile(r"new Array\((\d+)\+1\)")     # captures difficulty number

SLEEP_BETWEEN_REQUESTS = 0.75

# force consistent column order
ORDERED_COLS = [
    "Event", "Fighter1", "Fighter2", "Winner", "Weightclass", "Method",
    "Round", "Time", "Format",
    "KD1", "KD2", "SIG_STR1", "SIG_STR2", "TOTAL_STR1", "TOTAL_STR2",
    "TD1", "TD2", "SUB_ATT1", "SUB_ATT2", "REV1", "REV2", "CTRL1", "CTRL2",
    "Head1", "Head2", "Body1", "Body2", "Leg1", "Leg2",
    "Distance1", "Distance2", "Clinch1", "Clinch2", "Ground1", "Ground2",
    "event_date", "f1_dob", "f1_height", "f1_reach", "f1_stance",
    "f2_dob", "f2_height", "f2_reach", "f2_stance",

]

# create one shared session
# reused
def get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(HEADERS)
    return _SESSION

# checks if real page or Pow page
def _is_challenge(resp):
    return "/__c" in resp.text and "nonce=" in resp.text

def _solve_challenge(session, url, resp):
    m_nonce = _NONCE_RE.search(resp.text)
    m_diff = _DIFF_RE.search(resp.text)
    if not m_nonce or not m_diff:
        return False

    nonce = m_nonce.group(1)
    target = "0" * int(m_diff.group(1))

    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(target):
        n += 1

    base = "{0.scheme}://{0.netloc}".format(urlparse(url))
    session.post(
        base + "/__c",
        data={"nonce": nonce, "n": n},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=TIMEOUT,
    )
    return True

# get a url and solve the PoW challenge if needed
# def get(url, max_retries=2):
#     session = get_session()
#     resp = session.get(url, timeout=TIMEOUT)

#     attempts = 0
#     while _is_challenge(resp) and attempts < max_retries:
#         if not _solve_challenge(session, url, resp):
#             break
#         resp = session.get(url, timeout=TIMEOUT)
#         attempts += 1

#     resp.raise_for_status()
#     return resp
def get(url, max_retries=2):
    session = get_session()

    for attempt in range(3):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            break
        except requests.exceptions.ConnectionError as e:
            print(f"    Connection error on attempt {attempt + 1}/3: {e}")
            if attempt == 2:
                raise
            time_module.sleep(3)

    attempts = 0
    while _is_challenge(resp) and attempts < max_retries:
        if not _solve_challenge(session, url, resp):
            break
        resp = session.get(url, timeout=TIMEOUT)
        attempts += 1

    resp.raise_for_status()
    return resp


def get_soup(url):
    return BeautifulSoup(get(url).text, "html.parser")


def get_completed_events():
    url = "http://ufcstats.com/statistics/events/completed?page=all"
    soup = get_soup(url)

    event_details = []

    table = soup.find("tbody")
    print("Found table: ", table is not None)
    if table:
        print("Number of <tr> rows:", len(table.find_all("tr")))
    if not table:
        return event_details

    for tr in table.find_all("tr"):
        a = tr.find("a", class_="b-link b-link_style_black")
        if not a:
            continue

        name = a.text.strip()

        span = tr.select_one("span")
        if not span:
            continue
        date_str = span.text.strip()
        event_date = datetime.strptime(date_str, "%B %d, %Y")

        # if target_event_name is not None and name != target_event_name:
        #     continue

        event_details.append({
            "name": name,
            "date": date_str,
            "date_dt": event_date,
            "url": a.get("href")
        })



    return event_details


def get_upcoming_events():
    url = "http://ufcstats.com/statistics/events/upcoming?page=all"
    soup = get_soup(url)

    event_details = []

    table = soup.find("tbody")
    print("Found table: ", table is not None)
    if table:
        print("Number of <tr> rows:", len(table.find_all("tr")))
    if not table:
        return event_details

    for tr in table.find_all("tr"):
        a = tr.find("a", class_="b-link b-link_style_black")
        if not a:
            continue

        name = a.text.strip()

        span = tr.select_one("span")
        if not span:
            continue
        date_str = span.text.strip()
        event_date = datetime.strptime(date_str, "%B %d, %Y")

        # if target_event_name is not None and name != target_event_name:
        #     continue

        event_details.append({
            "name": name,
            "date": date_str,
            "date_dt": event_date,
            "url": a.get("href")
        })



    return event_details

# scrape all fight links from on event page
def get_fight_links(event_url):

    soup = get_soup(event_url)
    table = soup.find("tbody")

    if not table:
        return []

    fight_links = []
    seen = set()

    # for a in table.find_all("a", class_="b-flag"):
    for a in table.find_all("a", attrs={f"data-link": True}):
        href = a.get("data-link")

        if href and href not in seen:
            seen.add(href)
            fight_links.append(href)

    return fight_links

# find a table by checking its headers
def find_table_by_headers(soup, required_headers):
    """
    Looks through all tables on the page and returns the one whose headers
    contain all the required keywords.

    Example:
    If we want the totals table, we search for headers like
    KD, SIG, TOTAL, TD, SUB, REV, CTRL.
    """
    required_headers = [h.lower() for h in required_headers]

    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]

        # Check whether every required header appears somewhere in this table
        if all(any(req in h for h in headers) for req in required_headers):
            return table

    return None


# Helper: split a UFCStats row into red-corner values and blue-corner values
def row_red_blue_values(row):

    ps = [p.get_text(" ", strip=True) for p in row.find_all("p")]

    # Start from index 0 and take every second value -> red fighter
    red = ps[0::2]

    # Start from index 1 and take every second value -> blue fighter
    blue = ps[1::2]

    return red, blue


# Scrape metadata from a fighter's profile page
def parse_fighter_profile(url):

    soup = get_soup(url)

    # Fighter info is usually inside this unordered list
    ul = soup.find("ul", class_="b-list__box-list")

    # If the profile layout is missing or changed, return empty values
    if not ul:
        return {
            "height": None,
            "reach": None,
            "stance": None,
            "dob": None
        }

    data = {}

    # Each list item contains something like:
    # Height: 5' 10"
    # Reach: 72"
    # DOB: Jan 01, 1990
    for li in ul.find_all("li", class_="b-list__box-list-item"):
        text = " ".join(li.get_text(" ", strip=True).split())

        # Split into key/value around the first colon
        if ":" in text:
            key, val = text.split(":", 1)
            data[key.strip().lower()] = val.strip()

    return {
        "height": data.get("height"),
        "reach": data.get("reach"),
        "stance": data.get("stance"),
        "dob": data.get("dob")
    }


# scrape one fight page. Return a row dict or None
def get_fight_with_profiles(fight_link, event_name, event_date, fighter_profile_cache):
    soup = get_soup(fight_link)

    winner = None
    draw_type = None

    # two fighter shown on page
    fighter_names = []
    fighter_urls = []

    #get fighter names, profile links and winner status
    for person in soup.find_all("div", class_="b-fight-details__person"):
        status_tag = person.find("i", class_="b-fight-details__person-status")
        name_tag = person.find("a", class_="b-link b-fight-details__person-link")

        name = name_tag.get_text(strip=True) if name_tag else ""
        url = name_tag.get("href") if name_tag else None
        status = status_tag.get_text(strip=True) if status_tag else ""

        fighter_names.append(name)
        fighter_urls.append(url)

        # W for winner, D for draw, NC for no contest
        if status == "W":
            winner = name
        elif status in ["D", "NC"]:
            draw_type = status

    # if no winner is marked
    if winner is None:
        if draw_type == "D":
            winner = "DRAW"
        elif draw_type == "NC":
            winner = "NO CONTEST"

    if len(fighter_names) < 2:
        return None

    # basic fight details__person
    title_tag = soup.find("h2", class_="b-content__title")
    title = title_tag.get_text(" ", strip=True) if title_tag else event_name

    weight_class_tag = soup.find("i", class_="b-fight-details__fight-title")
    weight_class = ( weight_class_tag.get_text(" ", strip=True).replace("Bout", "").strip()
        if weight_class_tag else "" )

    method = ""
    method_outer = soup.find("i", class_="b-fight-details__text-item_first")
    if method_outer:
        inner = method_outer.find_all("i")
        if len(inner) >= 2:
            method = inner[1].text.strip()

    round_ = ""
    time_ = ""
    format_ = ""

    items = soup.find_all("i", class_="b-fight-details__text-item")
    for item in items[:3]:
        label_tag = item.find("i", class_="b-fight-details__label")
        if not label_tag:
            continue
        label = label_tag.text.strip().rstrip(":")
        label_tag.extract()
        value = item.text.strip()

        if label == "Round":
            round_ = value
        elif  label == "Time":
            time_ = value
        elif label == "Time format":
            format_ = value.split()[0]

    # find main stat tables
    # Totals table contains things like KD, sig strikes, takedowns, etc.
    totals_table = find_table_by_headers(
        soup, ["KD", "SIG", "TOTAL", "TD", "SUB", "REV", "CTRL"]
    )

    # # This table contains head/body/leg plus distance/clinch/ground breakdowns
    target_table = find_table_by_headers(
        soup, ["HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]
    )

    def first_row_rb(table):
        """
        Helper to safely get the first row of a table and split it into
        red-corner values and blue-corner values.
        """
        if not table:
            return [], []

        tbody = table.find("tbody")
        if not tbody:
            return [], []

        row = tbody.find("tr")
        if not row:
            return [], []

        return row_red_blue_values(row)

    totals_red, totals_blue = first_row_rb(totals_table)
    target_red, target_blue = first_row_rb(target_table)

    # Default values in case tables are missing
    KD1 = KD2 = SIG1 = SIG2 = TOT1 = TOT2 = TD1 = TD2 = ""
    SUB1 = SUB2 = REV1 = REV2 = CTRL1 = CTRL2 = ""
    Head1 = Head2 = Body1 = Body2 = Leg1 = Leg2 = ""
    Distance1 = Distance2 = Clinch1 = Clinch2 = Ground1 = Ground2 = ""

    # Parse totals table values
    if len(totals_red) >= 10 and len(totals_blue) >= 10:
        # First entry is usually fighter name, so remove it
        totals_red = totals_red[1:]
        totals_blue = totals_blue[1:]

        # Order on UFCStats is usually:
        # KD, SIG STR, SIG STR %, TOTAL STR, TD, TD %, SUB ATT, REV, CTRL
        KD1, SIG1, _, TOT1, TD1, _, SUB1, REV1, CTRL1 = totals_red[:9]
        KD2, SIG2, _, TOT2, TD2, _, SUB2, REV2, CTRL2 = totals_blue[:9]

    # Parse target/position breakdown values
    if len(target_red) >= 6 and len(target_blue) >= 6:
        # Last 6 values are usually:
        # HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND
        Head1, Body1, Leg1, Distance1, Clinch1, Ground1 = target_red[-6:]
        Head2, Body2, Leg2, Distance2, Clinch2, Ground2 = target_blue[-6:]


    # Scrape fighter profiles, but use cache so repeated fighters are not re-downloaded
    f1_url = fighter_urls[0]
    f2_url = fighter_urls[1]

    if f1_url and f1_url not in fighter_profile_cache:
        fighter_profile_cache[f1_url] = parse_fighter_profile(f1_url)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if f2_url and f2_url not in fighter_profile_cache:
        fighter_profile_cache[f2_url] = parse_fighter_profile(f2_url)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    f1_meta = fighter_profile_cache.get(f1_url, {})
    f2_meta = fighter_profile_cache.get(f2_url, {})


    # Return one fully enriched fight row
    return {
        "Event": event_name,
        "Fighter1": fighter_names[0],
        "Fighter2": fighter_names[1],
        "Winner": winner,
        "Weightclass": weight_class,
        "Method": method,
        "Round": round_,
        "Time": time_,
        "Format": format_,
        "KD1": KD1,
        "KD2": KD2,
        "SIG_STR1": SIG1,
        "SIG_STR2": SIG2,
        "TOTAL_STR1": TOT1,
        "TOTAL_STR2": TOT2,
        "TD1": TD1,
        "TD2": TD2,
        "SUB_ATT1": SUB1,
        "SUB_ATT2": SUB2,
        "REV1": REV1,
        "REV2": REV2,
        "CTRL1": CTRL1,
        "CTRL2": CTRL2,
        "Head1": Head1,
        "Head2": Head2,
        "Body1": Body1,
        "Body2": Body2,
        "Leg1": Leg1,
        "Leg2": Leg2,
        "Distance1": Distance1,
        "Distance2": Distance2,
        "Clinch1": Clinch1,
        "Clinch2": Clinch2,
        "Ground1": Ground1,
        "Ground2": Ground2,
        # "event_date": event_date,
        "event_date": event_date.strftime("%B %d, %Y"),
        "f1_dob": f1_meta.get("dob"),
        "f1_height": f1_meta.get("height"),
        "f1_reach": f1_meta.get("reach"),
        "f1_stance": f1_meta.get("stance"),
        "f2_dob": f2_meta.get("dob"),
        "f2_height": f2_meta.get("height"),
        "f2_reach": f2_meta.get("reach"),
        "f2_stance": f2_meta.get("stance"),
        # "fight_url": fight_link,
        # "f1_url": f1_url,
        # "f2_url": f2_url,
    }



def scrape_event(event_dict, fighter_profile_cache=None):

    # store fighter profile results so repeated fighters do not cause repeat scraping
    if fighter_profile_cache is None:
        fighter_profile_cache = {}

    # store all fight rows here before converting to df
    rows = []
    fight_links = get_fight_links(event_dict["url"])

    for i, fight_link in enumerate(fight_links, start=1):
        try:
            row = get_fight_with_profiles(
                fight_link=fight_link,
                event_name=event_dict["name"],
                event_date=event_dict["date_dt"],
                fighter_profile_cache=fighter_profile_cache,
            )
            if row is not None:
                rows.append(row)

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"    Error scraping fight: {fight_link}")
            print(f"    {e}")

    return rows

def main():

    events = get_completed_events()
    print(f"found {len(events)} completed events")

    if not events:
        return

    test_event = events[0]

    print(f"scraping test event: {test_event['name']} | {test_event['date']}")

    rows = scrape_event(test_event)
    df = pd.DataFrame(rows)
    cols_present = [c for c in ORDERED_COLS if c in df.columns]
    df = df[cols_present]
    print(f"scraped {len(df)} rows")
    print(f"df shape: {df.shape}")
    print(df.head())



if __name__ == "__main__":
    main()
