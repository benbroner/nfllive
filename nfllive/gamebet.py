from nfllive.modelling import scikitmodels
from  nfl_ref import full_package as fp 

model = scikitmodels.oneline_win()

df = fp.scraper_main()
print(df)