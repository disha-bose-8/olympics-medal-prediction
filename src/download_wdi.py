import wbgapi as wb

indicators = {
    "NY.GDP.MKTP.CD":    "GDP",
    "NY.GDP.PCAP.CD":    "GDP_Per_Capita",
    "NY.GDP.MKTP.KD.ZG": "GDP_Growth",
    "SP.POP.TOTL":       "Pop",
    "SP.POP.GROW":       "Pop_Growth",
    "AG.LND.TOTL.K2":    "Area",
}

years = [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016]

df = wb.data.DataFrame(
    list(indicators),
    time=years,
    columns="series",
    skipAggs=True,          # leaves out World, regions, income groups
    numericTimeKeys=True,   # years come back as numbers
)

df = df.reset_index().rename(columns={"economy": "Code", "time": "Year"})
df = df.rename(columns=indicators)

df.to_csv("data/raw/wdi_selected.csv", index=False)
print(df.shape)
print(df.head())