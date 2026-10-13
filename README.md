# EdgeLab forecasts

A free website with Premier League match probabilities from the EdgeLab model
(Dixon-Coles), every bookmaker's odds, and value bets checked against the betting
exchange. GitHub rebuilds it twice a day (09:00 and 20:00 Cyprus time). It is hosted free on
GitHub Pages at `https://<your-username>.github.io/<repo-name>/`.

## Set it up (about 10 minutes, once)

1. Create a free account at github.com.
2. Click **New repository**. Name it, for example, `edgelab`. Choose **Public**
   (GitHub Pages is free for public repositories). Create it.
3. On the new repository page, click **uploading an existing file** and drag in
   everything from this folder, including the hidden `.github` folder. (On a Mac,
   press Cmd+Shift+. in Finder to show hidden folders.) Click **Commit changes**.
4. Go to **Settings > Pages**. Under **Source**, choose **GitHub Actions**.
5. Go to **Settings > Actions > General > Workflow permissions**, choose
   **Read and write permissions**, and save.
6. Go to the **Actions** tab, open **Update forecasts**, and click **Run workflow**.
   After about two minutes the site is live at the address shown in the run.

Share that address with your friends. It works on any phone; on iPhone use
Share > Add to Home Screen to make it open like an app.

## What runs

`build.py` downloads every season in `config.json` from football-data.co.uk, fits the
model on all results so far, predicts the upcoming round from Football-Data's
`fixtures.csv` (with bookmakers' odds) or, if that has no Premier League games yet,
from `fixtures_manual.csv`, rebuilds this season's track record, and writes
`public/index.html`. Each new prediction is logged once in `history/predictions.csv`
(committed back to the repository) so the live record cannot be edited after the fact.

## Weekly upkeep

Nothing, normally. If Football-Data's fixtures file is late, type the round into
`fixtures_manual.csv` on GitHub (team names exactly as Football-Data spells them,
e.g. `Man United`, `Nott'm Forest`) and click **Run workflow**.

## Once a season

Edit `config.json`: add the new season (`code` like `2728`, dates, and the three
promoted teams), set `current_season`, and drop the oldest season if you like.

## Notes

- GitHub pauses scheduled runs in a repository with no activity for 60 days. The
  prediction log commits usually prevent this; if it happens, click **Enable workflow**
  in the Actions tab.
- Test locally: `pip install -r requirements.txt`, put `E0_2324.csv` ... `E0_2627.csv`
  in a folder, then `python build.py --local that-folder`. Open `public/index.html`.

## Daily odds checks and alerts

With the `ODDS_API_KEY` secret set, every run (twice a day) pulls live prices for all
upcoming Premier League matches from The Odds API, including matches football-data has
not listed yet, so value shows up early in the week when prices are softest. Each run
uses 2 API credits (about 120 a month).

When a run finds a value bet for the first time, it opens a GitHub issue titled
"EdgeLab: N new value bets". GitHub emails the repository owner and notifies the GitHub
mobile app. To change this, edit the "Alert on new value bets" step in
`.github/workflows/build.yml`.
