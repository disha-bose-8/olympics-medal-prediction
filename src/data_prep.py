import pandas as pd

RAW = "data/raw/"
OUT = "data/processed/olympics_merged.csv"

# ---------- 1. Olympic data: medals and athletes per country per Games ----------
ev = pd.read_csv(RAW + "athlete_events.csv")
ev = ev[ev["Season"] == "Summer"]

medals = (ev.dropna(subset=["Medal"])
            .drop_duplicates(["Year", "NOC", "Event", "Medal"])   # one medal per event
            .groupby(["Year", "NOC"]).size().rename("Medals"))
athletes = ev.groupby(["Year", "NOC"])["ID"].nunique().rename("Athletes")

oly = athletes.to_frame().join(medals).fillna({"Medals": 0}).reset_index()
oly["Medals"] = oly["Medals"].astype(int)

# ---------- 2. Medals at the previous Games (do this BEFORE cutting off at 1988) ----------
prev = oly[["Year", "NOC", "Medals"]].copy()
prev["Year"] += 4
prev = prev.rename(columns={"Medals": "Medals_Last_Games"})
oly = oly.merge(prev, on=["Year", "NOC"], how="left")
oly["Medals_Last_Games"] = oly["Medals_Last_Games"].fillna(0)

oly = oly[oly["Year"] >= 1988]
oly["Total_Medals_Year"] = oly.groupby("Year")["Medals"].transform("sum")
oly["Athletes_pct"] = oly["Athletes"] / oly.groupby("Year")["Athletes"].transform("sum") * 100

# ---------- 3. Olympic codes to World Bank codes ----------
NOC_TO_ISO = {
    "GER": "DEU", "NED": "NLD", "SUI": "CHE", "DEN": "DNK", "GRE": "GRC", "POR": "PRT",
    "CRO": "HRV", "BUL": "BGR", "LAT": "LVA", "SLO": "SVN", "INA": "IDN", "IRI": "IRN",
    "KSA": "SAU", "UAE": "ARE", "NGR": "NGA", "RSA": "ZAF", "CHI": "CHL", "PUR": "PRI",
    "HAI": "HTI", "ZIM": "ZWE", "ZAM": "ZMB", "ALG": "DZA", "BAH": "BHS", "MAS": "MYS",
    "PHI": "PHL", "SIN": "SGP", "SRI": "LKA", "VIE": "VNM", "MGL": "MNG", "KUW": "KWT",
    "LIB": "LBN", "CRC": "CRI", "GUA": "GTM", "HON": "HND", "PAR": "PRY", "TAN": "TZA",
    "MRI": "MUS", "BUR": "BFA", "MAD": "MDG", "MAW": "MWI", "NEP": "NPL", "OMA": "OMN",
    "MYA": "MMR", "CAM": "KHM", "ANG": "AGO", "BOT": "BWA", "BAR": "BRB", "BER": "BMU",
    "BRU": "BRN", "ESA": "SLV", "GAM": "GMB", "GRN": "GRD", "GUI": "GIN", "ISV": "VIR",
    "LBA": "LBY", "LES": "LSO", "MTN": "MRT", "NCA": "NIC", "NIG": "NER", "SAM": "WSM",
    "SEY": "SYC", "SKN": "KNA", "SOL": "SLB", "SUD": "SDN", "TGA": "TON", "TOG": "TGO",
    "VAN": "VUT", "VIN": "VCT", "ARU": "ABW", "ASA": "ASM", "BHU": "BTN", "BIZ": "BLZ",
    "CAY": "CYM", "CGO": "COG", "GBS": "GNB", "GEQ": "GNQ", "IVB": "VGB", "ANT": "ATG",
    "MON": "MCO", "PLE": "PSE", "URU": "URY", "FIJ": "FJI", "BAN": "BGD", "CHA": "TCD", "KOS": "XKX",
}
oly["Code"] = oly["NOC"].replace(NOC_TO_ISO)

# ---------- 4. World Bank data plus "% of world" features ----------
wdi = pd.read_csv(RAW + "wdi_selected.csv")
wdi["GDP_pct_World"] = wdi["GDP"] / wdi.groupby("Year")["GDP"].transform("sum") * 100
wdi["Pop_pct_World"] = wdi["Pop"] / wdi.groupby("Year")["Pop"].transform("sum") * 100

# ---------- 5. Merge ----------
df = oly.merge(wdi, on=["Code", "Year"], how="left", indicator=True)

# check what failed to match BEFORE dropping the _merge column
bad = df[df["_merge"] == "left_only"]
print("Unmatched codes:", sorted(bad["NOC"].unique()))
print("Rows lost to no match:", len(bad))
matched = df[df["_merge"] == "both"]
print("Rows lost to missing values after matching:", len(matched) - len(matched.dropna()))

df = df.drop(columns="_merge").dropna()      # the paper also drops rows with nulls
df = df.rename(columns={"NOC": "Nation"})

cols = ["Nation", "Year", "GDP", "GDP_Per_Capita", "GDP_Growth", "GDP_pct_World",
        "Pop", "Pop_pct_World", "Pop_Growth", "Area", "Athletes", "Athletes_pct",
        "Medals_Last_Games", "Total_Medals_Year", "Medals"]
df = df[cols]
df.to_csv(OUT, index=False)

print(df.shape)
print(df.groupby("Year").size())
print(df[df["Year"] == 2016].sort_values("Medals", ascending=False).head(5)[["Nation", "Medals"]])
print("Share of rows with at least one medal:", (df["Medals"] > 0).mean())

m = oly.merge(wdi, on=["Code", "Year"], how="inner")
lost = m[m.isna().any(axis=1)].copy()
lost["missing"] = lost.isna().apply(lambda r: ", ".join(r.index[r]), axis=1)
print(lost[lost["Medals"] > 0][["Year", "NOC", "Medals", "missing"]]
        .sort_values("Medals", ascending=False).head(15).to_string())